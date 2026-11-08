"""Record a frozen motion-quality SAC Oracle following figure-eight commands."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, cast

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import hydra
from omegaconf import DictConfig, OmegaConf

from unilab.visualization.demo_commands import FigureEightCommandConfig
from unilab.visualization.fada_recording import FADAVideoRecordingConfig, record_fada_playback
from unilab.visualization.fada_rendering import FADADemoRenderConfig
from unilab.visualization.oracle_recording import create_oracle_recording_session


@hydra.main(version_base="1.3", config_path="../conf", config_name="record_oracle")
def main(cfg: DictConfig) -> None:
    checkpoint_path = cfg.training.play_checkpoint_path
    figure_eight = FigureEightCommandConfig(**cast(
        dict[str, Any], OmegaConf.to_container(cfg.demo.figure_eight, resolve=True),
    ))
    recording = FADAVideoRecordingConfig(**cast(
        dict[str, Any], OmegaConf.to_container(cfg.video_recording, resolve=True),
    ))
    rendering = FADADemoRenderConfig(**cast(
        dict[str, Any], OmegaConf.to_container(cfg.demo.rendering, resolve=True),
    ))
    session = create_oracle_recording_session(
        checkpoint_path,
        figure_eight=figure_eight,
        device=str(cfg.training.device),
        scene_model_file=cfg.env.scene.model_file,
        max_episode_seconds=float(cfg.env.max_episode_seconds),
        footprints=bool(cfg.demo.footprints.enabled),
    )
    try:
        print(
            f"[record_oracle] Device: {session.device}; "
            f"sim_dt={session.env.cfg.sim_dt:g}s; ctrl_dt={session.env.cfg.ctrl_dt:g}s",
            flush=True,
        )
        record_fada_playback(
            session, cfg.interactive, recording,
            render_config=rendering, policy_kind="oracle",
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
