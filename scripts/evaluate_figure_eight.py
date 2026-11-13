"""Score recorded figure-eight geometry without running simulation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from unilab.visualization.figure_eight_evaluation import evaluate_figure_eight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory', type=Path, required=True, help='Recorded trial trajectory.npz')
    parser.add_argument('--reference', type=Path, required=True,
                        help='Measured unloaded trajectory.npz from the same command table')
    args = parser.parse_args()
    try:
        result = evaluate_figure_eight(args.trajectory, args.reference)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == '__main__':
    main()
