"""Installed CLI for the real Kumar2024 equivalence-first E001 study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .kumar2024 import Kumar2024StudyConfig, run_kumar2024_study


def _int_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _text_list(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantumbci-kumar2024",
        description=(
            "Run the real Kumar2024 E001 study under merged neurOS longitudinal authority. "
            "This command downloads public MOABB data only when explicitly invoked."
        ),
    )
    parser.add_argument("--subjects", type=_int_list, default=(1, 10))
    parser.add_argument(
        "--held-out-sessions",
        type=_text_list,
        default=("5",),
        help="comma-separated targets; use --all-target-sessions for sessions 1..5",
    )
    parser.add_argument("--all-target-sessions", action="store_true")
    parser.add_argument("--budgets", type=_int_list, default=(0, 1, 2, 5, 10))
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--evaluation-fraction", type=float, default=0.5)
    parser.add_argument("--fmin", type=float, default=8.0)
    parser.add_argument("--fmax", type=float, default=30.0)
    parser.add_argument("--resample", type=float, default=None)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--covariance-regularization", type=float, default=1e-6)
    parser.add_argument("--quantumbci-source-sha", required=True)
    parser.add_argument("--neuros-source-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Kumar2024StudyConfig(
        subjects=tuple(args.subjects),
        held_out_sessions=None if args.all_target_sessions else tuple(args.held_out_sessions),
        budgets_per_class=tuple(args.budgets),
        split_seed=int(args.split_seed),
        evaluation_fraction=float(args.evaluation_fraction),
        fmin=float(args.fmin),
        fmax=float(args.fmax),
        resample_hz=args.resample,
        ridge=float(args.ridge),
        covariance_regularization=float(args.covariance_regularization),
    )
    result = run_kumar2024_study(
        args.output,
        config=config,
        quantumbci_source_sha=str(args.quantumbci_source_sha),
        neuros_source_sha=str(args.neuros_source_sha),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
