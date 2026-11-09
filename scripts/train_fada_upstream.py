"""Run the pinned original FADA trainer with a UniLab Oracle source.

The existing train_distill.py / oracle_fada routes are not used or modified.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--upstream-root", required=True, help="Unmodified pinned LeCAR-Lab/FADA checkout")
    parser.add_argument("--oracle-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True, help="New, independent run directory")
    parser.add_argument("--scene-model-file")
    args, overrides = parser.parse_known_args()
    from unilab.algos.torch.distill.fada.upstream_training import run_upstream_training
    result = run_upstream_training(
        upstream_root=args.upstream_root, oracle_checkpoint=args.oracle_checkpoint,
        output_dir=args.output_dir, scene_model_file=args.scene_model_file,
        overrides=overrides[1:] if overrides[:1] == ["--"] else overrides,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
