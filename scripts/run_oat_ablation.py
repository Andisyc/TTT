"""Run Table 3 ablations in isolated processes; never overwrite partial trials."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT_DIR / "conf/distill/oat_component_ablation.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--first-seed", type=int)
    parser.add_argument("--methods", nargs="+", choices=("fada_ttt", "online_reference", "self_supervised_correction", "full_oat"))
    parser.add_argument("--payloads-kg", nargs="+", type=float)
    parser.add_argument("--reference-run-dir", help="Explicit common reference template; supports {payload_kg} and {seed}")
    parser.add_argument("--resume", action="store_true", help="Skip verified completed trials from the identical batch plan")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without creating files or launching workers")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT_DIR / "src"))
    from omegaconf import DictConfig, OmegaConf

    from unilab.visualization.experiment_ablation import plan_ablation_trials, run_ablation_batch

    config = OmegaConf.load(args.config.expanduser())
    if not isinstance(config, DictConfig):
        raise ValueError("ablation config must be a mapping")
    for field in ("repeats", "first_seed", "methods", "payloads_kg"):
        value = getattr(args, field)
        if value is not None:
            config[field] = value
    if args.reference_run_dir is not None:
        config.protocol.evaluation.reference_run_dir = args.reference_run_dir
    output = args.output_dir.expanduser().resolve()
    trials = plan_ablation_trials(config, output_dir=output)
    if args.dry_run:
        for method in config.methods:
            for mass in config.payloads_kg:
                print(f"{method} payload={mass:g}kg seeds={config.first_seed}.."
                      f"{config.first_seed + config.repeats - 1}")
        print(f"Planned trials: {len(trials)}; each trial uses a fresh process")
        print(f"Output: {output}")
        return
    run_ablation_batch(trials, output_dir=output, root_dir=ROOT_DIR, resume=args.resume)


if __name__ == "__main__":
    main()
