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
    parser.add_argument("--update-interval", type=int, help="T in control steps; also sets the update batch size")
    parser.add_argument("--duration-s", type=float, help="Total simulation and video duration in seconds")
    parser.add_argument("--export-adapted-after-payload-s", type=float,
                        help="Run this many seconds under load and save a frozen inference checkpoint")
    parser.add_argument("--forward-m-s", type=float, help="Override the straight-line forward command")
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
    if args.forward_m_s is not None:
        import math

        if not math.isfinite(args.forward_m_s) or args.forward_m_s <= 0:
            parser.error("--forward-m-s must be finite and positive")
        config.protocol.command = [args.forward_m_s, 0.0, 0.0]
    output = args.output_dir.expanduser().resolve()
    trials = plan_ablation_trials(
        config, output_dir=output, update_interval=args.update_interval, duration_s=args.duration_s,
        export_adapted_after_payload_s=args.export_adapted_after_payload_s,
    )
    if args.dry_run:
        for method in config.methods:
            for mass in config.payloads_kg:
                print(f"{method} payload={mass:g}kg seeds={config.first_seed}.."
                      f"{config.first_seed + config.repeats - 1}")
        trial = trials[0]
        print(f"Checkpoint: {trial['checkpoint_path']}")
        print(f"T={trial['online']['update_interval']} window_size={trial['online']['window_size']} "
              f"duration={trial['duration_s']:g}s attach={trial['payload']['attach_time_s']:g}s "
              f"episode_limit={trial['env_overrides'].get('max_episode_seconds')} "
              f"evaluation={trial['evaluation']['start_s']:g}..{trial['evaluation']['end_s']:g}s after attachment")
        if trial.get("export_adapted_after_payload_s") is not None:
            print(f"Export adapted.pt after {trial['export_adapted_after_payload_s']:g}s under load; command={trial['command']}")
        print(f"Planned trials: {len(trials)}; each trial uses a fresh process")
        print(f"Output: {output}")
        return
    run_ablation_batch(trials, output_dir=output, root_dir=ROOT_DIR, resume=args.resume)


if __name__ == "__main__":
    main()
