"""Installed CLI for publication-grade confirmatory representation studies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .confirmatory_representation import (
    CONFIRMATORY_REPRESENTATION_BENCHMARK,
    ConfirmatoryRepresentationPolicy,
    write_confirmatory_representation_bundle,
)


def _load(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON input must contain an object")
    return payload


def _policy_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = payload.get("policy")
    if nested is None:
        return payload
    if not isinstance(nested, Mapping):
        raise ValueError("policy must be a JSON object")
    return nested


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantumbci-confirmatory",
        description=(
            "Fingerprint preregistration policies and run BMRB Representation v2 "
            "confirmatory studies."
        ),
    )
    parser.add_argument("--version", action="version", version=CONFIRMATORY_REPRESENTATION_BENCHMARK)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fingerprint = subparsers.add_parser(
        "policy-fingerprint",
        help="print the immutable fingerprint that an external registration must bind",
    )
    fingerprint.add_argument("policy_or_manifest")
    fingerprint.add_argument("--json", action="store_true")

    validate = subparsers.add_parser(
        "policy-validate",
        help="validate a policy and report whether supplied registration evidence matches it",
    )
    validate.add_argument("policy_or_manifest")
    validate.add_argument("--json", action="store_true")

    run = subparsers.add_parser(
        "run",
        help="run a schema-v2 confirmatory representation manifest",
    )
    run.add_argument("manifest")
    run.add_argument("--output-dir", default="bmrb-confirmatory-representation")
    run.add_argument("--json", action="store_true")
    return parser


def _policy(path: str | Path) -> ConfirmatoryRepresentationPolicy:
    payload = _load(path)
    return ConfirmatoryRepresentationPolicy.from_mapping(_policy_payload(payload))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "policy-fingerprint":
        policy = _policy(args.policy_or_manifest)
        if args.json:
            print(
                json.dumps(
                    {
                        "benchmark": CONFIRMATORY_REPRESENTATION_BENCHMARK,
                        "policy_id": policy.policy_id,
                        "decision_fingerprint": policy.decision_fingerprint,
                        "confirmatory_authority": policy.confirmatory_authority,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(policy.decision_fingerprint)
        return 0

    if args.command == "policy-validate":
        policy = _policy(args.policy_or_manifest)
        result = {
            "benchmark": CONFIRMATORY_REPRESENTATION_BENCHMARK,
            "policy_id": policy.policy_id,
            "decision_fingerprint": policy.decision_fingerprint,
            "confirmatory_authority": policy.confirmatory_authority,
            "registration_uri": (
                None
                if policy.preregistration is None
                else policy.preregistration.registration_uri
            ),
        }
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(f"policy: {policy.policy_id}")
            print(f"fingerprint: {policy.decision_fingerprint}")
            print(f"external registration binding: {str(policy.confirmatory_authority).lower()}")
        return 0

    if args.command == "run":
        json_path, report_path = write_confirmatory_representation_bundle(
            args.manifest, args.output_dir
        )
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        evidence = payload["representation_evidence"]
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print("BMRB Confirmatory Representation: completed")
            print(f"confirmatory authority: {str(evidence['confirmatory_authority']).lower()}")
            print(f"scientific criteria passed: {str(evidence['scientific_criteria_passed']).lower()}")
            print(f"promotion eligible: {str(evidence['promotion_eligible']).lower()}")
            print(f"bundle: {json_path}")
            print(f"report: {report_path}")
        return 0

    raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
