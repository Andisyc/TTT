"""Measure a frozen FADA checkpoint, fit Gerono commands, or verify frozen playback."""
from __future__ import annotations

import os
import sys
from pathlib import Path

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

import hydra
from omegaconf import DictConfig

from unilab.visualization.gerono_measurement import run_gerono_calibration


@hydra.main(version_base="1.3", config_path="../conf/distill", config_name="config")
def main(cfg: DictConfig) -> None:
    run_gerono_calibration(cfg, ROOT_DIR, log=lambda message: print(message, flush=True))


if __name__ == "__main__":
    main()
