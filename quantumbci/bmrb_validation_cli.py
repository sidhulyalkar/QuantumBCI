"""Installed CLI for known-ground-truth BMRB validation."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from typing import Sequence

from .bmrb_validation import run_bmrb_validation_suite


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantumbci-bmrb-validate",
        description="Run known-ground-truth adversarial validation of BMRB decision behavior.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {version('quantum-bci')}")
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--participants", type=int, default=8)
    parser.add_argument("--bootstrap-resamples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1901)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-qualified",
        action="store_true",
        help="Exit non-zero when the known-truth suite does not satisfy its software QA contract.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_bmrb_validation_suite(
        replicates=args.replicates,
        seed=args.seed,
        participants=args.participants,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(args.output)
    if args.require_qualified and not result["qualified"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
