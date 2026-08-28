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
from .bmrb_causal import build_bmrb_causal_bundle, write_bmrb_causal_bundle
from .bmrb_representation import (
    build_bmrb_representation_bundle,
    write_bmrb_representation_bundle,
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
            "QuantumBCI and neuros-mechint artifacts."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dynamics = subparsers.add_parser(
        "dynamics",
        help="build BMRB-Dynamics repeated-case reliability and mechanism profile",
    )
    dynamics.add_argument(
        "manifest", help="JSON manifest of participant/occasion stability artifacts"
    )
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
    dynamics.add_argument(
        "--json", action="store_true", help="print full machine-readable bundle"
    )

    representation = subparsers.add_parser(
        "representation",
        help=(
            "build BMRB-Representation conservation evidence from exact-paired frozen E001 "
            "representation lanes"
        ),
    )
    representation.add_argument(
        "manifest",
        help=(
            "JSON representation manifest binding a preregistered policy and two or more "
            "verified E001 representation-lane artifact directories"
        ),
    )
    representation.add_argument("--output-dir", default="bmrb-representation")
    representation.add_argument(
        "--json", action="store_true", help="print full machine-readable representation bundle"
    )

    causal = subparsers.add_parser(
        "causal",
        help=(
            "attach participant-balanced intervention, ablation and matched-classical "
            "recovery evidence to a BMRB-Dynamics bundle"
        ),
    )
    causal.add_argument(
        "manifest",
        help=(
            "JSON causal manifest binding the upstream BMRB bundle, policy, neuros-mechint "
            "dose-response/evidence-pack artifacts, and matched classical recovery records"
        ),
    )
    causal.add_argument("--output-dir", default="bmrb-causal")
    causal.add_argument(
        "--json", action="store_true", help="print full machine-readable causal bundle"
    )
    return parser


def _promotion_label(profile: object) -> str:
    ceiling = getattr(profile, "promotion_ceiling", None)
    return "none" if ceiling is None else ceiling.name.lower()


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
        print("BMRB-Dynamics: completed")
        print(f"study: {bundle.study_id}")
        print(f"cases: {len(bundle.case_specs)}")
        print(f"participants: {bundle.reliability.participant_count}")
        print(f"evidence coverage: {profile.evidence_coverage_tier.name.lower()}")
        print(f"promotion ceiling: {_promotion_label(profile)}")
        print(f"first failing gate: {profile.first_failing_gate or 'none'}")
        print(f"bundle: {json_path}")
        print(f"report: {html_path}")
        print("Claim ceiling: quantum_inspired; necessity requires causal/ablation evidence")
    return 0


def _run_representation(args: argparse.Namespace) -> int:
    bundle = build_bmrb_representation_bundle(args.manifest)
    json_path, html_path = write_bmrb_representation_bundle(bundle, args.output_dir)
    if args.json:
        print(json.dumps(bundle.to_mapping(), indent=2, sort_keys=True))
    else:
        result = bundle.conservation
        profile = bundle.profile
        print("BMRB-Representation: completed")
        print(f"study: {bundle.study_id}")
        print(f"mechanism: {bundle.mechanism_id}")
        print(f"representations: {result.representation_count}")
        print(f"representation families: {result.representation_family_count}")
        print(f"participants: {result.participant_count}")
        print(f"direction match: {result.direction_match_fraction:.6g}")
        print(
            "information-novel representation fraction: "
            f"{result.information_novel_representation_fraction:.6g}"
        )
        print(f"conservation criteria passed: {str(result.conservation_criteria_passed).lower()}")
        print(f"adversary survival passed: {str(result.adversary_survival_passed).lower()}")
        print(f"policy preregistered: {str(bundle.policy.preregistered).lower()}")
        print(f"representation promotion eligible: {str(result.promotion_eligible).lower()}")
        print(f"evidence coverage: {profile.evidence_coverage_tier.name.lower()}")
        print(f"promotion ceiling: {_promotion_label(profile)}")
        print(f"first failing gate: {profile.first_failing_gate or 'none'}")
        print(f"bundle: {json_path}")
        print(f"report: {html_path}")
        print("Physical-quantum promotion: locked; independent witness evidence is required")
    return 0


def _run_causal(args: argparse.Namespace) -> int:
    bundle = build_bmrb_causal_bundle(args.manifest)
    json_path, html_path = write_bmrb_causal_bundle(bundle, args.output_dir)
    if args.json:
        print(json.dumps(bundle.to_mapping(), indent=2, sort_keys=True))
    else:
        profile = bundle.profile
        result = bundle.causal_result
        print("BMRB causal necessity: completed")
        print(f"study: {bundle.study_id}")
        print(f"participants: {len(result.participants)}")
        print(f"cases: {len(result.cases)}")
        print(f"scientific criteria passed: {str(result.scientific_criteria_passed).lower()}")
        print(f"policy preregistered: {str(bundle.policy.preregistered).lower()}")
        print(f"causal promotion eligible: {str(result.promotion_eligible).lower()}")
        print(f"mean necessity: {result.mean_necessity_fraction:.6g}")
        print(
            "mean matched-classical recovery: "
            f"{result.mean_classical_recovery_fraction:.6g}"
        )
        print(f"evidence coverage: {profile.evidence_coverage_tier.name.lower()}")
        print(f"promotion ceiling: {_promotion_label(profile)}")
        print(f"first failing gate: {profile.first_failing_gate or 'none'}")
        print(f"bundle: {json_path}")
        print(f"report: {html_path}")
        print("Physical-quantum promotion: locked; independent witness evidence is required")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "dynamics":
        return _run_dynamics(args)
    if args.command == "representation":
        return _run_representation(args)
    if args.command == "causal":
        return _run_causal(args)
    raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
