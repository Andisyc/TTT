"""Record and score frozen straight-line walking after online payload adaptation."""
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
from unilab.visualization.experiment_common import GROUP_MODES
from unilab.visualization.experiment_evaluation import evaluate_experiment
from unilab.visualization.experiment_recording import experiment_recording_run
from unilab.visualization.fada_recording import FADAVideoRecordingConfig, capture_fada_trial
from unilab.visualization.fada_rendering import FADADemoRenderConfig
from unilab.visualization.playback_cli import _build_play_args
from unilab.visualization.playback_controls import _build_playback_config
from unilab.visualization.playback_distill_sessions import create_fada_playback_session
from unilab.visualization.straight_line_trial import run_straight_line_trial

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "model/new_best/student_iter_0031.pt")
    parser.add_argument("--scene-model-file", type=Path, default=ROOT / "model/straight_mix_50k_assets/g1/scene_flat.xml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("frozen", "full_oat", "fada_ttt"), default="frozen")
    parser.add_argument("--forward-m-s", type=float, default=0.4)
    parser.add_argument("--approach-s", type=float, default=20.0, help="Straight-line adaptation time before test entry")
    parser.add_argument("--duration-s", type=float, default=12.5, help="Fixed duration of the test phase")
    parser.add_argument("--record-from-start", action="store_true", help="Record from simulation time zero; default records only the test")
    parser.add_argument("--keep-updating", action="store_true", help="Continue online adaptation during the test instead of freezing at entry")
    parser.add_argument("--payload-kg", type=float, default=0.0)
    parser.add_argument("--payload-side", choices=("left", "right"), default="left")
    parser.add_argument("--payload-attach-s", type=float, default=5.0)
    parser.add_argument("--gamma", type=float, default=0.5, help="OAT correction gain; unused by FADA-TTT")
    parser.add_argument(
        "--update-interval", type=int, default=5,
        help="T: update-check interval in control steps (default: 5); window capacity also follows T",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    if args.update_interval <= 0:
        parser.error("--update-interval must be a positive integer")
    for name in ("forward_m_s", "duration_s"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.keep_updating and args.method == "frozen":
        parser.error("--keep-updating requires --method full_oat or fada_ttt")
    for name in ("approach_s", "payload_kg", "payload_attach_s", "gamma"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.payload_kg > 0 and args.payload_attach_s >= args.approach_s:
        parser.error("payload attachment must precede the frozen test")
    if args.method != "frozen" and (args.payload_kg == 0 or args.approach_s == 0):
        parser.error("online methods require a positive payload and adaptation time")
    checkpoint = args.checkpoint.expanduser().resolve()
    source_scene = args.scene_model_file.expanduser().resolve()
    if not checkpoint.is_file() or not source_scene.is_file():
        parser.error("checkpoint and checkpoint-matched scene must exist")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.device == "cpu":
        torch.set_num_threads(1)
    ensure_registries()
    apply_training_seed(args.seed)
    with TemporaryDirectory(prefix="straight_line_g1_") as tmp:
        scene = copytree(source_scene.parent, Path(tmp) / "g1") / source_scene.name
        with initialize_config_dir(config_dir=str(ROOT / "conf/distill"), version_base="1.3"):
            cfg = compose(config_name="config", overrides=[
                "task=g1_walk_flat/mujoco", "recording_profile=straight_payload", f"algo.seed={args.seed}",
                "online_adaptation.enabled=false", "env.runtime_payload.enabled=false",
                "experiment_recording.enabled=true", "demo.footprints.enabled=true", "demo.rendering.enabled=true",
                "demo.rendering.show_reference_route=false", "video_recording.show_status_overlay=false",
                "interactive.camera_follow_body=false", "interactive.camera_distance=null",
                "interactive.camera_elevation=-70.0", "interactive.camera_azimuth=90.0",
            ])
        cfg.env.commands.vel_limit = [[args.forward_m_s, 0.0, 0.0]] * 2
        cfg.env.runtime_payload.enabled = args.payload_kg > 0
        cfg.env.runtime_payload.added_mass_kg = args.payload_kg
        cfg.env.runtime_payload.attach_time_s = args.payload_attach_s
        x, y, z = cfg.env.runtime_payload.attachment_pos_m
        cfg.env.runtime_payload.attachment_pos_m = [x, abs(y) if args.payload_side == "left" else -abs(y), z]
        if args.method != "frozen":
            online = OmegaConf.load(ROOT / "conf/distill/oat_component_ablation.yaml").protocol.online
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
        cfg.env.max_episode_seconds = max(float(cfg.env.max_episode_seconds), args.approach_s + args.duration_s + 1.0)
        cfg.experiment_recording.output_dir = str(output / "raw")
        cfg.video_recording.output_path = str(output / "trial.mp4")
        cfg.video_recording.duration_s = args.duration_s + (args.approach_s if args.record_from_start else 0.0)
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
                    raise ValueError("frozen baseline requires a checkpoint without LoRA adapters")

                def capture_trial():
                    return capture_fada_trial(session, play_args, video, label=f"Straight Line | {args.method}", render_config=render)

                result = run_straight_line_trial(
                    session, forward_m_s=args.forward_m_s, approach_s=args.approach_s,
                    test_duration_s=args.duration_s, trajectory_path=output / "trajectory.npz", capture_trial=capture_trial,
                    record_from_start=args.record_from_start,
                    keep_updating=args.keep_updating,
                )
        finally:
            session.wrapped_env.close()
    if result["completed_test"]:
        start = result["test_start_step"] * result["ctrl_dt"]
        velocity = evaluate_experiment({
            "run_dir": str(output / "raw"), "output_dir": str(output / "velocity"),
            "linear_frame": "body_xy", "sample_phase": "completed", "relative_to": "run_start",
            "start_s": start, "end_s": start + result["requested_test_duration_s"],
        })
        episode, = velocity["episodes"]
        if not episode["full_requested_window"]:
            raise RuntimeError("velocity recording does not cover the complete test")
        result["velocity"] = episode["velocity_observed_segment"]
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("\n=== Straight Line metrics ===")
    print(f"Status: {result['stop_reason']} | Simulation time: {result['duration_s']:.2f} s")
    if result["completed_test"]:
        geometry = result["geometry"]
        velocity_metrics = result["velocity"]
        if geometry["line_rmse_m"] is None:
            print("E_line: x (no horizontal travel; straightness undefined)")
        else:
            print(f"E_line (line-fit RMSE):     {geometry['line_rmse_m']:.6f} m")
            print(f"Maximum line deviation:    {geometry['max_line_deviation_m']:.6f} m")
        if geometry["direction_error_deg"] is None:
            print("E_dir: x (global trajectory direction undefined)")
        else:
            print(f"E_dir (global direction):  {geometry['direction_error_deg']:.6f} deg")
        print(f"Forward distance:          {geometry['forward_distance_m']:.6f} m")
        print(f"Endpoint lateral offset:   {geometry['endpoint_lateral_m']:.6f} m")
        print(f"Linear velocity RMSE:      {velocity_metrics['E_lin_m_s']:.6f} m/s")
        print(f"Angular velocity RMSE:     {velocity_metrics['E_ang_rad_s']:.6f} rad/s")
    else:
        print("E_line: x (test incomplete; no full-test metrics)")
        print("E_dir: x (test incomplete; no full-test metrics)")
    print(f"Full results: {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
