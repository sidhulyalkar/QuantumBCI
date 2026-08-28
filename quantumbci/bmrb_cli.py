"""Command-line entry point for Brain Mechanism Recapitulation Benchmark bundles."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from .bmrb import (
    DEFAULT_E002_RELIABILITY_ESTIMATES,
    build_bmrb_dynamics_bundle,
    write_bmrb_dynamics_bundle,
)
from .reliability import (
    DEFAULT_RELIABILITY_BOOTSTRAP_RESAMPLES,
    DEFAULT_RELIABILITY_SEED,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantumbci-bmrb",
        description=(
            "Build mechanism-necessity evidence bundles from independently qualified "
            "QuantumBCI case artifacts."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dynamics = subparsers.add_parser(
        "dynamics",
        help="build BMRB-Dynamics repeated-case reliability and mechanism profile",
    )
    dynamics.add_argument("manifest", help="JSON manifest of participant/occasion stability artifacts")
    dynamics.add_argument("--output-dir", default="bmrb-dynamics")
    dynamics.add_argument(
        "--resamples",
        type=int,
        default=DEFAULT_RELIABILITY_BOOTSTRAP_RESAMPLES,
        help="participant-primary hierarchical bootstrap resamples",
    )
    dynamics.add_argument("--seed", type=int, default=DEFAULT_RELIABILITY_SEED)
    dynamics.add_argument(
        "--estimate",
        action="append",
        dest="estimates",
        help=(
            "point-estimate name to include; repeat for multiple names. Defaults to the "
            "canonical E002 mechanism and predictive comparison set."
        ),
    )
    dynamics.add_argument("--json", action="store_true", help="print full machine-readable bundle")
    return parser


def _run_dynamics(args: argparse.Namespace) -> int:
    estimates: Sequence[str] = (
        tuple(args.estimates) if args.estimates else DEFAULT_E002_RELIABILITY_ESTIMATES
    )
    bundle = build_bmrb_dynamics_bundle(
        args.manifest,
        estimate_names=estimates,
        n_resamples=args.resamples,
        seed=args.seed,
    )
    json_path, html_path = write_bmrb_dynamics_bundle(bundle, args.output_dir)
    if args.json:
        print(json.dumps(bundle.to_mapping(), indent=2, sort_keys=True))
    else:
        profile = bundle.profile
        promotion = (
            "none"
            if profile.promotion_ceiling is None
            else profile.promotion_ceiling.name.lower()
        )
        print("BMRB-Dynamics: completed")
        print(f"study: {bundle.study_id}")
        print(f"cases: {len(bundle.case_specs)}")
        print(f"participants: {bundle.reliability.participant_count}")
        print(f"evidence coverage: {profile.evidence_coverage_tier.name.lower()}")
        print(f"promotion ceiling: {promotion}")
        print(f"first failing gate: {profile.first_failing_gate or 'none'}")
        print(f"bundle: {json_path}")
        print(f"report: {html_path}")
        print("Claim ceiling: quantum_inspired; necessity requires causal/ablation evidence")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "dynamics":
        return _run_dynamics(args)
    raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
