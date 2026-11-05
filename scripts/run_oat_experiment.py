"""Run one OAT/FADA component trial headlessly, without video rendering."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from unilab.visualization.experiment_execution import run_oat_experiment


@hydra.main(version_base="1.3", config_path="../conf/distill", config_name="oat_experiment")
def main(cfg: DictConfig) -> None:
    result = run_oat_experiment(cfg, root_dir=ROOT_DIR)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
