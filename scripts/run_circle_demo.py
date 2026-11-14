"""Record one unloaded FADA student trial under a fixed body-frame turn command."""
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
    parser.add_argument("--duration-s", type=float, default=75.0, help="Recording duration in simulation seconds (default: 75)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    duration_s = args.duration_s
    if not math.isfinite(duration_s) or duration_s <= 0:
        parser.error("--duration-s must be finite and positive")
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
                if controller is None or any(fada_adapter_named_parameters(controller.policy)):
                    raise ValueError("Circle Demo requires a base checkpoint without LoRA adapters")
                if session.env.cfg.runtime_payload.enabled:
                    raise RuntimeError("Circle Demo requires an unloaded environment")

                def capture_trial():
                    return capture_fada_trial(session, play_args, video, label="Unloaded circle", render_config=render)

                result = run_circle_trial(
                    session, forward_m_s=0.33, yaw_rate_rad_s=0.20, duration_s=duration_s,
                    settle_s=5.0, trajectory_path=output / "trajectory.npz", capture_trial=capture_trial,
                )
        finally:
            session.wrapped_env.close()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
