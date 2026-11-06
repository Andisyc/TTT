#!/usr/bin/env python3
"""Fixed-command Oracle acceptance; no training or online adaptation.

Example:
  uv run --frozen --no-sync python scripts/evaluate_motion_quality_oracle.py \
    --checkpoint /path/to/model_5000.pt --device cuda --record-video

Writes run.json, timeseries.csv, outcome.json and optional rollout.mp4. Existing
output directories are never overwritten. Completion is not an automatic gait
quality verdict; inspect the fixed-command measurements and optional video.
Protocol v2 gives all cases the same nominal gait period and initial phase pair
[0, pi], records actual actor command/phase inputs, and exports angular velocity
in explicitly corrected body and world frames.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from unilab.training.oracle.evaluation import evaluate_motion_quality_oracle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--record-video", action="store_true")
    args = parser.parse_args()
    output = args.output_dir
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        output = args.checkpoint.expanduser().resolve().parent / "acceptance" / stamp
    result = evaluate_motion_quality_oracle(
        args.checkpoint, output, duration_s=args.duration_s, seed=args.seed,
        device=args.device, record_video=args.record_video,
    )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
