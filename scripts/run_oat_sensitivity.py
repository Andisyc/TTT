"""Run only the requested OAT sensitivity values with per-trial video and metrics."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Filled oat_experiment.yaml protocol")
    parser.add_argument("--output-dir", required=True, type=Path, help="New batch directory")
    parser.add_argument("--param", required=True, choices=("T", "sigma", "gamma"))
    parser.add_argument("--values", required=True, nargs="+", type=float)
    parser.add_argument("--repeats", type=int, default=10, help="Trials per value (default: 10)")
    parser.add_argument("--first-seed", type=int, default=0, help="First seed, reused across values (default: 0)")
    parser.add_argument("--dry-run", action="store_true", help="Print trial paths without creating an environment")
    args = parser.parse_args()

    if sys.platform == "linux":
        os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(ROOT_DIR / "src"))
    from omegaconf import OmegaConf

    from unilab.visualization.experiment_sensitivity import plan_sensitivity_trials, run_sensitivity_trials

    output = args.output_dir.expanduser().resolve()
    protocol = OmegaConf.load(args.config.expanduser())
    trials = plan_sensitivity_trials(
        protocol, param=args.param, values=args.values, repeats=args.repeats,
        first_seed=args.first_seed, output_dir=output,
    )
    if args.dry_run:
        for trial in trials:
            print(trial["output_dir"])
        print(f"Planned trials: {len(trials)}")
        return
    run_sensitivity_trials(trials, param=args.param, repeats=args.repeats, output_dir=output, root_dir=ROOT_DIR)
    print(f"Summary: {output / 'summary.json'}")


if __name__ == "__main__":
    main()
