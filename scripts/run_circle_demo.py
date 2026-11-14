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
from unilab.visualization.circle_trial import run_circle_trial
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
    parser.add_argument("--duration-s", type=float, default=75.0, help="Maximum simulation seconds to finish one lap (default: 75)")
    parser.add_argument(
        "--forward-m-s", type=float, default=0.33,
        help="Fixed forward speed in m/s (default: 0.33)",
    )
    parser.add_argument(
        "--yaw-rate-rad-s", type=float, default=0.20,
        help="Fixed yaw rate in rad/s: positive for left, negative for right (default: 0.20)",
    )
    parser.add_argument("--method", choices=("frozen", "full_oat", "fada_ttt"), default="frozen")
    parser.add_argument("--payload-kg", type=float, default=0.0, help="Payload mass applied on the first physics step (default: 0)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    duration_s = args.duration_s
    if not math.isfinite(duration_s) or duration_s <= 0:
        parser.error("--duration-s must be finite and positive")
    if not math.isfinite(args.forward_m_s) or args.forward_m_s <= 0:
        parser.error("--forward-m-s must be finite and positive")
    if not math.isfinite(args.yaw_rate_rad_s) or args.yaw_rate_rad_s == 0:
        parser.error("--yaw-rate-rad-s must be finite and nonzero")
    if not math.isfinite(args.payload_kg) or args.payload_kg < 0:
        parser.error("--payload-kg must be finite and nonnegative")
    if args.method != "frozen" and args.payload_kg == 0:
        parser.error("online circle methods require a positive --payload-kg")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")
    source_scene = args.scene_model_file.expanduser().resolve()
    if not source_scene.is_file():
        parser.error(f"Checkpoint-matched scene not found: {source_scene}")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    if args.device == "cpu":
        torch.set_num_threads(1)
    ensure_registries()
    apply_training_seed(0)
    with TemporaryDirectory(prefix="circle_g1_scene_") as tmp:
        scene = copytree(source_scene.parent, Path(tmp) / "g1") / source_scene.name
        with initialize_config_dir(config_dir=str(ROOT / "conf/distill"), version_base="1.3"):
            cfg = compose(config_name="config", overrides=[
                "task=g1_walk_flat/mujoco", "recording_profile=figure_eight", "algo.seed=0",
                "demo.figure_eight.enabled=false", "demo.rendering.show_reference_route=false",
                "demo.footprints.max_records=3750", "demo.rendering.max_footprint_markers=3750",
                "online_adaptation.enabled=false", "env.runtime_payload.enabled=false",
                "experiment_recording.enabled=true", "video_recording.show_status_overlay=false",
            ])
        if args.payload_kg > 0:
            cfg.env.runtime_payload.enabled = True
            cfg.env.runtime_payload.attach_time_s = 0.0
            cfg.env.runtime_payload.added_mass_kg = args.payload_kg
        if args.method != "frozen":
            online = OmegaConf.load(ROOT / "conf/distill/oat_component_ablation.yaml").protocol.online
            cfg.online_adaptation = OmegaConf.merge(cfg.online_adaptation, online)
            cfg.online_adaptation.enabled = True
            cfg.online_adaptation.mode = GROUP_MODES[args.method]
            if args.method == "fada_ttt":
                cfg.online_adaptation.sigma = 0.0
        cfg.training.play_checkpoint_path = str(checkpoint)
        cfg.training.device = args.device
        OmegaConf.update(cfg, "training.fada.scene_model_file", str(scene), force_add=True)
        cfg.experiment_recording.output_dir = str(output / "raw")
        cfg.video_recording.output_path = str(output / "trial.mp4")
        cfg.video_recording.duration_s = duration_s
        cfg.env.max_episode_seconds = max(float(cfg.env.max_episode_seconds), duration_s + 1.0)
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
                    trajectory_path=output / "trajectory.npz", capture_trial=capture_trial,
                )
        finally:
            session.wrapped_env.close()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
