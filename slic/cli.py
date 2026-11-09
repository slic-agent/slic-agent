"""slic command line: check | solve | eval | train | data | tasks | serve."""
from __future__ import annotations

import sys
from typing import List

from . import __version__

COMMANDS = {
    "check": ("slic.check", "pre-run check of the installation (packages, GPUs, simulator); --dumps checks a run's dumps"),
    "solve": ("slic.solve", "run one optimization episode and stream every turn"),
    "eval": ("slic.evaluate", "evaluate a checkpoint on a task set (running Pass@1)"),
    "train": ("slic.train", "launch an RL (SLIC and its baselines) or SFT run"),
    "data": ("slic.data", "download the released task sets and technology cards"),
    "tasks": ("slic.tasks", "build a task parquet from your own circuits / technology card"),
    "serve": ("slic.serve", "start the browser UI"),
}


def usage() -> str:
    lines = [f"slic {__version__}", "usage: slic <command> [--config FILE] [options]", ""]
    lines += [f"  {name:<7} {desc}" for name, (_, desc) in COMMANDS.items()]
    lines += ["", "run `slic <command> --help` for the command's options"]
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0
    if argv[0] in ("-V", "--version"):
        print(__version__)
        return 0
    cmd = argv[0]
    if cmd not in COMMANDS:
        print(usage(), file=sys.stderr)
        print(f"\nunknown command: {cmd}", file=sys.stderr)
        return 2
    import importlib

    mod = importlib.import_module(COMMANDS[cmd][0])
    return int(mod.main(argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
