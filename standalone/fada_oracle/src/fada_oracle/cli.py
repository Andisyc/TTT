"""Independent command entry; environment and trainer imports stay lazy."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Official FADA with an independently loaded Oracle")
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("setup", help="Prepare the private pinned official source (no simulation)")
    setup.add_argument("--workdir", type=Path, default=Path.cwd())
    training = commands.add_parser(
        "train", help="Start a new official Planner–IDM run",
        epilog="Additional arguments are passed to the pinned official FADA parser unchanged.",
    )
    training.add_argument("--workdir", type=Path, default=Path.cwd())
    training.add_argument("--oracle-checkpoint", type=Path, required=True)
    training.add_argument("--scene-model-file", type=Path, required=True)
    training.add_argument("--output-dir", type=Path)
    training.add_argument("--backend", choices=("classic", "warp"), default="warp")
    args, official_arguments = parser.parse_known_args(argv)
    if args.command == "setup":
        if official_arguments:
            parser.error("setup accepts no training arguments")
        from .upstream import prepare_upstream
        print(prepare_upstream(args.workdir))
        return 0
    from .integration import train
    output = args.output_dir or args.workdir / "outputs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Training output: {output.resolve()}", flush=True)
    result = train(
        project_root=args.workdir, checkpoint=args.oracle_checkpoint,
        scene=args.scene_model_file, output=output, backend=args.backend,
        official_arguments=official_arguments[1:] if official_arguments[:1] == ["--"] else official_arguments,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
