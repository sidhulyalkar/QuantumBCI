"""Materialize the v0.14 E002 trajectory-block bootstrap stability evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..stability import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_MIN_SUCCESS_FRACTION,
    run_e002_bootstrap_stability,
)
from ..trajectory_authority import load_trajectory_contract_descriptor
from .nonlinear_control_task import build_nonlinear_control_artifact
from .probabilistic_ssm_task import _read_json, _write_json


def _verify_nonlinear_artifact(
    supplied: dict[str, Any],
    *,
    descriptor_path: Path,
    trajectory_index_path: Path,
    matched_path: Path,
    classical_controls_path: Path,
    probabilistic_ssm_path: Path,
    switching_state_path: Path,
) -> dict[str, Any]:
    if supplied.get("experiment") != "E002":
        raise ValueError("nonlinear-control artifact is not E002")
    if supplied.get("artifact_role") != "flexible_nonlinear_classical_control":
        raise ValueError("nonlinear-control artifact has the wrong artifact role")
    if supplied.get("status") != "pass":
        raise ValueError("nonlinear-control artifact did not pass execution")
    if not bool(supplied.get("flexible_nonlinear_control_complete", False)):
        raise ValueError("v0.13 nonlinear classical control is incomplete")
    if not bool(supplied.get("bootstrap_stability_required", False)):
        raise ValueError("v0.13 artifact does not require bootstrap stability")
    if bool(supplied.get("intervention_stage_eligible", True)):
        raise ValueError("v0.13 artifact incorrectly permits intervention promotion")
    if bool(supplied.get("physical_quantum_promotion_eligible", True)):
        raise ValueError("v0.13 artifact incorrectly permits physical-quantum promotion")

    expected = build_nonlinear_control_artifact(
        descriptor_path=descriptor_path,
        trajectory_index_path=trajectory_index_path,
        matched_path=matched_path,
        classical_controls_path=classical_controls_path,
        probabilistic_ssm_path=probabilistic_ssm_path,
        switching_state_path=switching_state_path,
    )
    if supplied != expected:
        raise ValueError(
            "nonlinear-control artifact differs from independent v0.13 reconstruction"
        )
    return expected


def build_stability_artifact(
    *,
    descriptor_path: Path,
    trajectory_index_path: Path,
    matched_path: Path,
    classical_controls_path: Path,
    probabilistic_ssm_path: Path,
    switching_state_path: Path,
    nonlinear_control_path: Path,
    n_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    minimum_success_fraction: float = DEFAULT_MIN_SUCCESS_FRACTION,
) -> dict[str, Any]:
    data, authority = load_trajectory_contract_descriptor(descriptor_path)
    nonlinear = _read_json(nonlinear_control_path, label="nonlinear-control artifact")
    expected_nonlinear = _verify_nonlinear_artifact(
        nonlinear,
        descriptor_path=descriptor_path,
        trajectory_index_path=trajectory_index_path,
        matched_path=matched_path,
        classical_controls_path=classical_controls_path,
        probabilistic_ssm_path=probabilistic_ssm_path,
        switching_state_path=switching_state_path,
    )
    result = run_e002_bootstrap_stability(
        data,
        authority,
        n_replicates=n_replicates,
        seed=seed,
        minimum_success_fraction=minimum_success_fraction,
    )
    payload = result.to_mapping()
    for field in ("authority_fingerprint", "data_sha256"):
        if str(payload.get(field)) != str(expected_nonlinear.get(field)):
            raise RuntimeError(f"bootstrap stability produced mismatched {field}")

    return {
        **payload,
        "schema_version": 2,
        "status": "pass",
        "descriptor_name": descriptor_path.name,
        "trajectory_index_name": trajectory_index_path.name,
        "matched_baseline_name": matched_path.name,
        "classical_controls_name": classical_controls_path.name,
        "probabilistic_ssm_name": probabilistic_ssm_path.name,
        "switching_state_name": switching_state_path.name,
        "nonlinear_control_name": nonlinear_control_path.name,
        "upstream_nonlinear_artifact_verified": True,
        "upstream_nonlinear_artifact_reconstructed": True,
        "execution_complete": True,
        "stability_gate_pass": bool(payload["stability_evidence_complete"]),
        "predictive_adversary_ladder_complete": True,
        "intervention_direction_evidence_required": True,
        "intervention_stage_eligible": False,
        "dynamical_information_novel": False,
        "physical_quantum_promotion_eligible": False,
        "interpretation_ceiling": (
            "This artifact quantifies source-data perturbation stability after the complete "
            "v0.9-v0.13 predictive adversary ladder. A valid artifact may still fail the "
            "stability gate if too many bootstrap replicates fail. Single-case bootstrap "
            "intervals are not ICC. Even strong stability does not identify a biological or "
            "physical-quantum mechanism; intervention-direction evidence remains required."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", default="trajectory_contract.json")
    parser.add_argument("--trajectory-index", default="trajectory_index.json")
    parser.add_argument("--matched", default="matched_dynamics.json")
    parser.add_argument("--classical-controls", default="classical_controls.json")
    parser.add_argument("--probabilistic-ssm", default="probabilistic_ssm.json")
    parser.add_argument("--switching-state", default="switching_state.json")
    parser.add_argument("--nonlinear-control", default="nonlinear_control.json")
    parser.add_argument("--replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--minimum-success-fraction",
        type=float,
        default=DEFAULT_MIN_SUCCESS_FRACTION,
    )
    parser.add_argument("--output", default="bootstrap_stability.json")
    args = parser.parse_args(argv)

    output = Path(args.output)
    try:
        payload = build_stability_artifact(
            descriptor_path=Path(args.descriptor),
            trajectory_index_path=Path(args.trajectory_index),
            matched_path=Path(args.matched),
            classical_controls_path=Path(args.classical_controls),
            probabilistic_ssm_path=Path(args.probabilistic_ssm),
            switching_state_path=Path(args.switching_state),
            nonlinear_control_path=Path(args.nonlinear_control),
            n_replicates=args.replicates,
            seed=args.seed,
            minimum_success_fraction=args.minimum_success_fraction,
        )
        _write_json(output, payload)
        print(json.dumps({**payload, "artifact": str(output)}, sort_keys=True))
        return 0
    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        np.linalg.LinAlgError,
    ) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
