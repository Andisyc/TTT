"""Record one fixed-command circle trial with an optional online payload protocol."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory
from typing import Any, cast

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.algos.torch.distill.fada.adaptation import fada_adapter_named_parameters
from unilab.training import ensure_registries
from unilab.training.seed import apply_training_seed
from unilab.visualization.circle_trial import finalize_circle_trial, run_circle_trial
from unilab.visualization.experiment_common import GROUP_MODES
from unilab.visualization.experiment_recording import experiment_recording_run
from unilab.visualization.fada_recording import FADAVideoRecordingConfig, capture_fada_trial
from unilab.visualization.fada_rendering import FADADemoRenderConfig
from unilab.visualization.playback_cli import _build_play_args
from unilab.visualization.playback_controls import _build_playback_config
from unilab.visualization.playback_distill_sessions import create_fada_playback_session

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "model/new_best/student_iter_0031.pt")
    parser.add_argument(
        "--scene-model-file", type=Path,
        default=ROOT / "model/straight_mix_50k_assets/g1/scene_flat.xml",
        help="Scene from the XML asset tree matched to the checkpoint",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Matching unloaded frozen trajectory.npz for E_shape; omitted means E_shape_NA")
    parser.add_argument("--duration-s", type=float, default=75.0, help="Maximum simulation seconds per measured lap (default: 75)")
    parser.add_argument("--approach-s", type=float, default=0.0, help="Seconds of circular walking before measurement (default: 0)")
    parser.add_argument("--startup-s", type=float, default=0.0, help="Initial portion of circle adaptation without online updates")
    parser.add_argument("--record-from-start", action="store_true", help="Include adaptation in the video; default records only the test")
    parser.add_argument("--keep-updating", action="store_true", help="Keep online updates active during the measured circle")
    parser.add_argument(
        "--forward-m-s", type=float, default=0.33,
        help="Forward speed during adaptation and, unless overridden, measurement (m/s; default: 0.33)",
    )
    parser.add_argument(
        "--test-forward-m-s", type=float,
        help="Forward speed during measurement in m/s (default: same as --forward-m-s)",
    )
    parser.add_argument(
        "--yaw-rate-rad-s", type=float, default=0.20,
        help="Test yaw rate in rad/s (default: 0.20); also used during adaptation unless overridden",
    )
    parser.add_argument(
        "--approach-yaw-rate-rad-s", type=float,
        help="Adaptation yaw rate in rad/s (default: same as --yaw-rate-rad-s); 0 goes straight",
    )
    parser.add_argument("--method", choices=("frozen", "full_oat", "fada_ttt"), default="frozen")
    parser.add_argument("--gamma", type=float, default=0.5, help="Full OAT correction gain (default: %(default)s); unused by FADA-TTT and frozen")
    parser.add_argument("--payload-kg", type=float, default=0.0, help="Added payload mass (default: 0)")
    parser.add_argument("--payload-side", choices=("left", "right"), default="left", help="Shoulder for the payload (default: left)")
    parser.add_argument("--payload-attach-s", type=float, default=0.0, help="Payload attachment time from trial start, in simulation seconds")
    parser.add_argument("--payload-per-lap-kg", type=float, help="Add this mass after each completed lap, starting unloaded")
    parser.add_argument("--max-payload-kg", type=float, help="Final load to complete a lap at in per-lap mode")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: %(default)s)")
    parser.add_argument("--update-interval", type=int, help="Online update interval T in control steps; window size follows T (default: protocol setting)")
    parser.add_argument("--camera-distance", type=float, help="Fixed video camera distance in metres (default: recording profile); larger values widen the view")
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be an integer in [0, 2**32)")
    if args.update_interval is not None and args.update_interval <= 0:
        parser.error("--update-interval must be a positive integer")
    duration_s = args.duration_s
    if not math.isfinite(duration_s) or duration_s <= 0:
        parser.error("--duration-s must be finite and positive")
    if not math.isfinite(args.approach_s) or args.approach_s < 0:
        parser.error("--approach-s must be finite and nonnegative")
    if not math.isfinite(args.startup_s) or not 0 <= args.startup_s <= args.approach_s:
        parser.error("--startup-s must be finite and between 0 and --approach-s")
    if not math.isfinite(args.payload_attach_s) or not 0 <= args.payload_attach_s < args.approach_s + duration_s:
        parser.error("--payload-attach-s must be finite and fall within the trial")
    if not math.isfinite(args.forward_m_s) or args.forward_m_s <= 0:
        parser.error("--forward-m-s must be finite and positive")
    if args.test_forward_m_s is not None and (not math.isfinite(args.test_forward_m_s) or args.test_forward_m_s <= 0):
        parser.error("--test-forward-m-s must be finite and positive")
    if not math.isfinite(args.yaw_rate_rad_s) or args.yaw_rate_rad_s == 0:
        parser.error("--yaw-rate-rad-s must be finite and nonzero")
    if args.approach_yaw_rate_rad_s is not None and not math.isfinite(args.approach_yaw_rate_rad_s):
        parser.error("--approach-yaw-rate-rad-s must be finite")
    if not math.isfinite(args.payload_kg) or args.payload_kg < 0:
        parser.error("--payload-kg must be finite and nonnegative")
    mass_levels: list[float] = []
    if args.payload_per_lap_kg is not None or args.max_payload_kg is not None:
        increment, maximum = args.payload_per_lap_kg, args.max_payload_kg
        if increment is None or maximum is None or not all(math.isfinite(x) and x > 0 for x in (increment, maximum)):
            parser.error("per-lap mode requires positive finite --payload-per-lap-kg and --max-payload-kg")
        ratio = maximum / increment
        if not math.isfinite(ratio) or ratio < 1 or not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-9):
            parser.error("--max-payload-kg must be a positive multiple of --payload-per-lap-kg")
        if (args.payload_kg or args.payload_attach_s or args.approach_s or args.reference is not None
                or args.test_forward_m_s is not None or args.approach_yaw_rate_rad_s is not None):
            parser.error("per-lap mode requires a fixed command, no adaptation prefix, no timed payload and no reference")
        mass_levels = [i * increment for i in range(1, round(ratio))] + [maximum]
    if not math.isfinite(args.gamma) or args.gamma < 0:
        parser.error("--gamma must be finite and nonnegative")
    if args.camera_distance is not None and (not math.isfinite(args.camera_distance) or args.camera_distance <= 0):
        parser.error("--camera-distance must be finite and positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")
    source_scene = args.scene_model_file.expanduser().resolve()
    if not source_scene.is_file():
        parser.error(f"Checkpoint-matched scene not found: {source_scene}")
    reference = args.reference.expanduser().resolve() if args.reference is not None else None
    if reference is not None and not reference.is_file():
        parser.error(f"Unloaded reference trajectory not found: {reference}")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    if args.device == "cpu":
        torch.set_num_threads(1)
    ensure_registries()
    apply_training_seed(args.seed)
    with TemporaryDirectory(prefix="circle_g1_scene_") as tmp:
        scene = copytree(source_scene.parent, Path(tmp) / "g1") / source_scene.name
        with initialize_config_dir(config_dir=str(ROOT / "conf/distill"), version_base="1.3"):
            cfg = compose(config_name="config", overrides=[
                "task=g1_walk_flat/mujoco", "recording_profile=figure_eight", f"algo.seed={args.seed}",
                "demo.figure_eight.enabled=false", "demo.rendering.show_reference_route=false",
                "demo.footprints.max_records=3750", "demo.rendering.max_footprint_markers=3750",
                "online_adaptation.enabled=false", "env.runtime_payload.enabled=false",
                "experiment_recording.enabled=true", "video_recording.show_status_overlay=false",
            ])
        if args.camera_distance is not None:
            cfg.interactive.camera_distance = args.camera_distance
        if mass_levels or args.payload_kg > 0:
            cfg.env.runtime_payload.enabled = True
            cfg.env.runtime_payload.attach_time_s = args.payload_attach_s
            cfg.env.runtime_payload.added_mass_kg = args.payload_kg
            if mass_levels:
                OmegaConf.update(cfg, "env.runtime_payload.requested_mass_levels_kg", mass_levels, force_add=True)
            x, y, z = cfg.env.runtime_payload.attachment_pos_m
            cfg.env.runtime_payload.attachment_pos_m = [x, abs(y) if args.payload_side == "left" else -abs(y), z]
        if args.method != "frozen":
            online = OmegaConf.load(ROOT / "conf/distill/oat_component_ablation.yaml").protocol.online
            if args.update_interval is not None:
                online.update_interval = args.update_interval
            cfg.online_adaptation = OmegaConf.merge(cfg.online_adaptation, online)
            cfg.online_adaptation.enabled = True
            cfg.online_adaptation.mode = GROUP_MODES[args.method]
            if args.method == "fada_ttt":
                cfg.online_adaptation.sigma = 0.0
            else:
                cfg.online_adaptation.gamma = args.gamma
        cfg.training.play_checkpoint_path = str(checkpoint)
        cfg.training.device = args.device
        OmegaConf.update(cfg, "training.fada.scene_model_file", str(scene), force_add=True)
        cfg.experiment_recording.output_dir = str(output / "raw")
        cfg.video_recording.output_path = str(output / "trial.mp4")
        trial_budget_s = (len(mass_levels) + 1) * duration_s
        cfg.video_recording.duration_s = trial_budget_s
        if mass_levels:
            cfg.interactive.camera_follow_body = True
            cfg.video_recording.show_status_overlay = True
        if args.record_from_start and args.approach_s > 0:
            cfg.video_recording.duration_s += args.approach_s
            cfg.interactive.camera_follow_body = True
        cfg.env.max_episode_seconds = max(float(cfg.env.max_episode_seconds), args.approach_s + trial_budget_s + 1.0)
        play_args = _build_play_args(cfg, algo="fada")
        video = FADAVideoRecordingConfig(**cast(dict[str, Any], OmegaConf.to_container(cfg.video_recording, resolve=True)))
        render = FADADemoRenderConfig(**cast(dict[str, Any], OmegaConf.to_container(cfg.demo.rendering, resolve=True)))
        session, _, _ = create_fada_playback_session(
            playback_cfg=_build_playback_config(play_args, num_envs=1),
            cfg=cfg, root_dir=ROOT, device=args.device,
        )
        try:
            with experiment_recording_run(session):
                controller = session.controller
                if controller is None or (args.method == "frozen" and any(fada_adapter_named_parameters(controller.policy))):
                    raise ValueError("Frozen circle runs require a base checkpoint without LoRA adapters")

                def capture_trial():
                    return capture_fada_trial(session, play_args, video, label=f"Circle | {args.method}", render_config=render)

                result = run_circle_trial(
                    session, forward_m_s=args.forward_m_s, yaw_rate_rad_s=args.yaw_rate_rad_s, duration_s=duration_s,
                    test_forward_m_s=args.test_forward_m_s,
                    approach_yaw_rate_rad_s=args.approach_yaw_rate_rad_s,
                    approach_s=args.approach_s, keep_updating=args.keep_updating,
                    startup_s=args.startup_s,
                    record_from_start=args.record_from_start,
                    trajectory_path=output / "trajectory.npz", capture_trial=capture_trial,
                )
        finally:
            session.wrapped_env.close()
    result = finalize_circle_trial(result, output, reference=reference)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
