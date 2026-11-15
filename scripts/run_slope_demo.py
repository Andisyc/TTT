"""Record one fixed-command uphill trial with uninterrupted online adaptation."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory
from typing import Any, cast

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.algos.torch.distill.fada.adaptation import fada_adapter_named_parameters
from unilab.terrains.heightfield_terrains import HfSmoothRampTerrainCfg
from unilab.training import ensure_registries
from unilab.training.seed import apply_training_seed
from unilab.visualization.experiment_common import GROUP_MODES
from unilab.visualization.experiment_evaluation import evaluate_experiment
from unilab.visualization.experiment_recording import experiment_recording_run
from unilab.visualization.fada_recording import FADAVideoRecordingConfig, capture_fada_trial
from unilab.visualization.fada_rendering import FADADemoRenderConfig
from unilab.visualization.playback_cli import _build_play_args
from unilab.visualization.playback_controls import _build_playback_config
from unilab.visualization.playback_distill_sessions import create_fada_playback_session
from unilab.visualization.slope_trial import run_slope_trial

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "model/new_best/student_iter_0031.pt")
    parser.add_argument("--scene-model-file", type=Path, default=ROOT / "model/straight_mix_50k_assets/g1/scene_flat.xml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("frozen", "full_oat", "fada_ttt"), default="full_oat")
    parser.add_argument("--forward-m-s", type=float, default=0.4)
    parser.add_argument("--yaw-rate-rad-s", type=float, default=0.0, help="Fixed yaw command for open-loop calibration (positive: left, negative: right)")
    parser.add_argument("--approach-s", type=float, default=0.0, help="Additional flat approach time; updates continue on the slope")
    parser.add_argument("--duration-s", type=float, default=60.0, help="Timeout after the optional approach")
    parser.add_argument("--length-m", type=float, default=4.0, help="Horizontal ramp or flat test segment length, excluding the lead-in (default: 4 m)")
    parser.add_argument("--rise-m", type=float, default=0.175, help="Endpoint height above the starting flat; 0 selects flat ground (default: 0.175 m)")
    parser.add_argument("--width-m", type=float, default=2.0)
    parser.add_argument("--payload-kg", type=float, default=0.0)
    parser.add_argument("--payload-side", choices=("left", "right"), default="left")
    parser.add_argument("--payload-attach-s", type=float, default=5.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    for name in ("forward_m_s", "duration_s", "length_m", "width_m"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.yaw_rate_rad_s):
        parser.error("--yaw-rate-rad-s must be finite")
    if not math.isfinite(args.rise_m) or not 0 <= args.rise_m < args.length_m:
        parser.error("--rise-m must be finite, nonnegative and less than --length-m for a nonvertical circular arc")
    for name in ("approach_s", "payload_kg", "payload_attach_s", "gamma"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.width_m <= 0.5:
        parser.error("--width-m must exceed the 0.5 m body safety margin")
    if args.payload_kg and args.payload_attach_s >= args.approach_s + args.duration_s:
        parser.error("payload attachment must be within the trial")
    checkpoint = args.checkpoint.expanduser().resolve()
    source_scene = args.scene_model_file.expanduser().resolve()
    if not checkpoint.is_file() or not source_scene.is_file():
        parser.error("checkpoint and checkpoint-matched source scene must exist")
    ramp = HfSmoothRampTerrainCfg(
        horizontal_length_m=args.length_m, rise_m=args.rise_m,
        approach_length_m=1.0 + args.forward_m_s * args.approach_s,
    )
    # The arc ends at finish_x; the equal-height margin beyond it is not scored.
    length = math.ceil((2 * (ramp.finish_x + 1.0)) / ramp.horizontal_scale) * ramp.horizontal_scale
    width = math.ceil(args.width_m / ramp.horizontal_scale) * ramp.horizontal_scale
    ramp.size = (length, width)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.device == "cpu":
        torch.set_num_threads(1)
    ensure_registries()
    apply_training_seed(args.seed)
    with TemporaryDirectory(prefix="slope_g1_") as tmp:
        scene = copytree(source_scene.parent, Path(tmp) / "g1") / source_scene.name
        with initialize_config_dir(config_dir=str(ROOT / "conf/distill"), version_base="1.3"):
            cfg = compose(config_name="config", overrides=[
                "task=g1_walk_flat/mujoco", "recording_profile=figure_eight", f"algo.seed={args.seed}",
                "demo.figure_eight.enabled=false", "online_adaptation.enabled=false",
                "env.runtime_payload.enabled=false", "experiment_recording.enabled=true",
                "demo.rendering.show_reference_route=false", "video_recording.show_status_overlay=false",
                "interactive.camera_distance=null", "interactive.camera_elevation=-40.0",
                "interactive.camera_azimuth=90.0",
            ])
        OmegaConf.update(cfg, "env.slope_evaluation", asdict(ramp), force_add=True)
        cfg.env.commands.vel_limit = [[args.forward_m_s, 0.0, args.yaw_rate_rad_s]] * 2
        cfg.env.runtime_payload.enabled = args.payload_kg > 0
        cfg.env.runtime_payload.added_mass_kg = args.payload_kg
        cfg.env.runtime_payload.attach_time_s = args.payload_attach_s
        x, y, z = cfg.env.runtime_payload.attachment_pos_m
        cfg.env.runtime_payload.attachment_pos_m = [x, abs(y) if args.payload_side == "left" else -abs(y), z]
        if args.method != "frozen":
            online = OmegaConf.load(ROOT / "conf/distill/oat_component_ablation.yaml").protocol.online
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
        cfg.env.max_episode_seconds = max(float(cfg.env.max_episode_seconds), args.approach_s + args.duration_s + 1)
        cfg.experiment_recording.output_dir = str(output / "raw")
        cfg.video_recording.output_path = str(output / "trial.mp4")
        cfg.video_recording.duration_s = args.approach_s + args.duration_s
        play_args = _build_play_args(cfg, algo="fada")
        video = FADAVideoRecordingConfig(**cast(dict[str, Any], OmegaConf.to_container(cfg.video_recording, resolve=True)))
        render = FADADemoRenderConfig(**cast(dict[str, Any], OmegaConf.to_container(cfg.demo.rendering, resolve=True)))
        session, _, _ = create_fada_playback_session(
            playback_cfg=_build_playback_config(play_args, num_envs=1), cfg=cfg, root_dir=ROOT, device=args.device,
        )
        try:
            with experiment_recording_run(session):
                controller = session.controller
                if controller is None or (args.method == "frozen" and any(fada_adapter_named_parameters(controller.policy))):
                    raise ValueError("frozen baseline requires a base checkpoint without LoRA")
                if session.env.cfg.slope_evaluation is None:
                    raise RuntimeError("playback did not materialize the slope evaluation configuration")
                def capture_trial():
                    return capture_fada_trial(session, play_args, video, label=f"Arc ramp, {args.length_m:g} m long, {args.rise_m:g} m rise | {args.method}", render_config=render)
                result = run_slope_trial(
                    session, ramp=ramp, forward_m_s=args.forward_m_s, approach_s=args.approach_s,
                    yaw_rate_rad_s=args.yaw_rate_rad_s,
                    duration_s=args.duration_s, output=output, capture_trial=capture_trial,
                )
        finally:
            session.wrapped_env.close()
    if "arc_finished" in result["events_s"]:
        velocity = evaluate_experiment({
            "run_dir": str(output / "raw"), "output_dir": str(output / "velocity"),
            "linear_frame": "body_xy", "sample_phase": "completed", "relative_to": "run_start",
            "start_s": result["events_s"]["arc_started"],
            "end_s": result["events_s"]["arc_finished"],
        })
        result["velocity"] = velocity["episodes"][0]["velocity_observed_segment"]
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
