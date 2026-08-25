"""Experiment-stage task entry point.

Implemented stages are executable evidence contracts. Unfinished stages fail with
exit code 3 rather than fabricating scientific artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..dynamics_fitting import run_matched_qubit_dynamics_benchmark
from ..e002_synthetic import (
    CANONICAL_STRUCTURE_RESIDUAL_MAX,
    CLASSICAL_ADVERSARY_RESIDUAL_MIN,
    run_e002_synthetic_recovery_grid,
)
from ..equivalence import audit_density_covariance_equivalence
from ..trajectory_authority import load_trajectory_contract_descriptor


def _write_json(output: Path, payload: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _density_covariance_gate(*, output: Path, atol: float) -> dict[str, Any]:
    if atol <= 0:
        raise ValueError("atol must be positive")
    real_probe = np.asarray(
        [
            [1.0, 0.5, -0.25, 2.0],
            [0.2, -1.5, 0.8, 0.1],
            [2.1, 0.4, 1.3, -0.7],
            [-0.6, 1.2, -1.1, 0.9],
        ],
        dtype=float,
    )
    complex_probe = real_probe.astype(complex) + 1j * np.asarray(
        [
            [0.0, 0.2, -0.1, 0.3],
            [0.4, 0.0, 0.5, -0.2],
            [-0.3, 0.1, 0.0, 0.6],
            [0.2, -0.4, 0.3, 0.0],
        ],
        dtype=float,
    )
    probes: list[dict[str, Any]] = []
    for name, values in (("real", real_probe), ("complex", complex_probe)):
        for center in (True, False):
            audit = audit_density_covariance_equivalence(values, center=center, atol=atol)
            probes.append({"probe": name, **audit.to_mapping()})
    equivalent = all(bool(item["equivalent_within_tolerance"]) for item in probes)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass" if equivalent else "fail",
        "experiment": "E001",
        "gate": "density-covariance-equivalence",
        "claim_class": "quantum_inspired",
        "identity": "rho = X^H X / Tr(X^H X) after optional centering",
        "equivalence_class": "trace_normalized_hermitian_second_moment",
        "representation_information_novel": False,
        "equivalent_within_tolerance": equivalent,
        "tolerance": float(atol),
        "max_abs_error": max(float(item["max_abs_error"]) for item in probes),
        "probes": probes,
        "interpretation": (
            "The current density constructor is an operator-valued reparameterization "
            "of a classical Hermitian second moment. Predictive differences against "
            "weaker controls cannot be attributed to additional representation information."
        ),
    }
    _write_json(output, payload)
    return payload


def _e002_synthetic_recovery(*, output: Path, seed: int, noise_std: float) -> dict[str, Any]:
    payload = run_e002_synthetic_recovery_grid(seed=int(seed), noise_std=float(noise_std))
    payload = {
        **payload,
        "status": "pass" if payload["synthetic_identifiability_gate_pass"] else "fail",
        "artifact_role": "synthetic_recovery_witness",
    }
    _write_json(output, payload)
    return payload


def _e002_identifiability_gate(*, input_path: Path, output: Path) -> dict[str, Any]:
    source = _read_json_object(input_path, label="synthetic recovery artifact")
    if source.get("experiment") != "E002":
        raise ValueError("identifiability gate requires an E002 synthetic recovery artifact")
    median_error = float(source.get("median_normalized_recovery_error", float("inf")))
    sign_inversions = int(source.get("systematic_sign_inversions", -1))
    affine_equivalence = bool(source.get("affine_equivalence_pass", False))
    gauge_witness = bool(source.get("gauge_nonidentifiability_witness_pass", False))
    max_structure_residual = float(source.get("max_canonical_structure_residual", float("inf")))
    canonical_structure = bool(source.get("canonical_structure_pass", False))
    adversary = source.get("classical_adversary", {})
    if not isinstance(adversary, dict):
        adversary = {}
    adversary_residual = float(adversary.get("canonical_structure_residual", float("-inf")))
    adversary_rejected = bool(adversary.get("rejected_as_canonical_family", False))
    source_gate = bool(source.get("synthetic_identifiability_gate_pass", False))
    passed = bool(
        median_error <= 0.20
        and sign_inversions == 0
        and affine_equivalence
        and gauge_witness
        and canonical_structure
        and max_structure_residual <= CANONICAL_STRUCTURE_RESIDUAL_MAX
        and adversary_rejected
        and adversary_residual >= CLASSICAL_ADVERSARY_RESIDUAL_MIN
        and source_gate
    )
    payload = {
        "schema_version": 2,
        "status": "pass" if passed else "fail",
        "experiment": "E002",
        "gate": "synthetic-identifiability-and-specificity",
        "claim_class": "quantum_inspired",
        "source_artifact_name": input_path.name,
        "criteria": {
            "median_normalized_recovery_error_max": 0.20,
            "systematic_sign_inversions_max": 0,
            "canonical_structure_residual_max": CANONICAL_STRUCTURE_RESIDUAL_MAX,
            "classical_adversary_residual_min": CLASSICAL_ADVERSARY_RESIDUAL_MIN,
            "require_affine_equivalence_witness": True,
            "require_gauge_nonidentifiability_witness": True,
            "require_noncanonical_classical_adversary_rejection": True,
        },
        "observed": {
            "median_normalized_recovery_error": median_error,
            "systematic_sign_inversions": sign_inversions,
            "affine_equivalence_pass": affine_equivalence,
            "gauge_nonidentifiability_witness_pass": gauge_witness,
            "max_canonical_structure_residual": max_structure_residual,
            "canonical_structure_pass": canonical_structure,
            "classical_adversary_structure_residual": adversary_residual,
            "classical_adversary_rejected": adversary_rejected,
            "source_synthetic_gate_pass": source_gate,
        },
        "trajectory_contract_stage_eligible": passed,
        "dynamical_information_novel": False,
        "physical_quantum_promotion_eligible": False,
        "interpretation": (
            "Passing this gate establishes recoverability and family specificity only. "
            "The fully observed qubit dynamics remain affine-equivalent and do not support "
            "a physical neural quantum claim."
        ),
    }
    _write_json(output, payload)
    return payload


def _e002_trajectory_contract(*, input_path: Path, output: Path) -> dict[str, Any]:
    if not input_path.is_file():
        raise FileNotFoundError(f"trajectory contract descriptor not found: {input_path}")
    data, authority = load_trajectory_contract_descriptor(input_path)
    payload = {
        "schema_version": 1,
        "status": "pass",
        "experiment": "E002",
        "artifact_role": "trajectory_evidence_authority",
        "claim_class": "quantum_inspired",
        "descriptor_name": input_path.name,
        "authority": authority.to_dict(data=data),
        "shared_tensor_contract": {
            "data_sha256": data.data_sha256,
            "state_dimension": data.state_dimension,
            "fit_transition_count": int(len(authority.transition_pairs(data, "fit"))),
            "calibration_transition_count": int(len(authority.transition_pairs(data, "calibration"))),
            "evaluation_transition_count": int(len(authority.transition_pairs(data, "evaluation"))),
            "required_for_all_model_lanes": True,
        },
        "interpretation": (
            "The trajectory authority freezes exact state bytes, temporal identity, legal "
            "within-role transitions, purge gaps, and representation-fit scope."
        ),
    }
    _write_json(output, payload)
    return payload


def _verify_trajectory_index(
    *,
    trajectory_index_path: Path,
    data: Any,
    authority: Any,
) -> dict[str, Any]:
    source = _read_json_object(trajectory_index_path, label="trajectory index artifact")
    if source.get("experiment") != "E002" or source.get("artifact_role") != "trajectory_evidence_authority":
        raise ValueError("matched fitting requires an E002 trajectory evidence authority artifact")
    recorded_authority = source.get("authority")
    expected_authority = authority.to_dict(data=data)
    if recorded_authority != expected_authority:
        raise ValueError("trajectory index authority differs from the reconstructed descriptor authority")
    shared = source.get("shared_tensor_contract")
    if not isinstance(shared, dict):
        raise ValueError("trajectory index is missing shared_tensor_contract")
    if str(shared.get("data_sha256")) != data.data_sha256:
        raise ValueError("trajectory index state tensor SHA-256 differs from descriptor data")
    if not bool(shared.get("required_for_all_model_lanes", False)):
        raise ValueError("trajectory index does not require one shared tensor across model lanes")
    return source


def _e002_fit_matched_dynamics(
    *,
    descriptor_path: Path,
    trajectory_index_path: Path,
    output: Path,
    ridge: float,
) -> dict[str, Any]:
    if float(ridge) != 0.0:
        raise ValueError(
            "the authoritative v0.9 matched baseline requires ridge=0 because nonzero "
            "regularization has different geometry in the affine and canonical coordinates"
        )
    if not descriptor_path.is_file():
        raise FileNotFoundError(f"trajectory contract descriptor not found: {descriptor_path}")
    data, authority = load_trajectory_contract_descriptor(descriptor_path)
    _verify_trajectory_index(
        trajectory_index_path=trajectory_index_path,
        data=data,
        authority=authority,
    )
    result = run_matched_qubit_dynamics_benchmark(data, authority, ridge=0.0)
    payload = {
        **result.to_mapping(),
        "status": "pass",
        "artifact_role": "matched_dynamics_baseline",
        "descriptor_name": descriptor_path.name,
        "trajectory_index_name": trajectory_index_path.name,
        "claim_class": "quantum_inspired",
        "baseline_scope": [
            "unconstrained_affine_generator",
            "canonical_lindblad_family",
        ],
        "authoritative_ridge": 0.0,
        "regularization_geometry_matched": True,
        "fit_rank_diagnostics": {
            "affine_predictor_design_rank": int(result.affine.fit_design_rank),
            "affine_parameter_rank": int(result.affine.fit_design_rank * data.state_dimension),
            "affine_parameter_count": int(result.affine.parameter_count),
            "canonical_design_rank": int(result.canonical.fit_design_rank),
            "canonical_parameter_rank": int(result.canonical.fit_design_rank),
            "canonical_parameter_count": int(result.canonical.parameter_count),
        },
        "extended_classical_controls_required": True,
        "intervention_stage_eligible": False,
        "interpretation_ceiling": (
            "This artifact establishes a matched first predictive baseline only. Near parity "
            "with fewer parameters may support a useful constrained parameterization; it does "
            "not establish new dynamical information or a physical quantum mechanism."
        ),
    }
    _write_json(output, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("experiment")
    parser.add_argument("qualifier", nargs="?")
    parser.add_argument("--output")
    parser.add_argument("--input")
    parser.add_argument("--trajectory-index")
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--noise-std", type=float, default=0.003)
    parser.add_argument("--ridge", type=float, default=0.0)
    args = parser.parse_args(argv)

    try:
        if args.task == "equivalence-audit" and args.experiment == "E001" and args.qualifier == "density-covariance":
            output = Path(args.output or "equivalence_audit.json")
            payload = _density_covariance_gate(output=output, atol=float(args.atol))
            print(json.dumps({**payload, "artifact": str(output)}, sort_keys=True))
            return 0 if payload["equivalent_within_tolerance"] else 2

        if args.task == "synthetic-recovery" and args.experiment == "E002":
            output = Path(args.output or "synthetic_recovery.json")
            payload = _e002_synthetic_recovery(output=output, seed=int(args.seed), noise_std=float(args.noise_std))
            print(json.dumps({**payload, "artifact": str(output)}, sort_keys=True))
            return 0 if payload["synthetic_identifiability_gate_pass"] else 2

        if args.task == "gate" and args.experiment == "E002" and args.qualifier == "identifiability":
            input_path = Path(args.input or "synthetic_recovery.json")
            output = Path(args.output or "identifiability_gate.json")
            payload = _e002_identifiability_gate(input_path=input_path, output=output)
            print(json.dumps({**payload, "artifact": str(output)}, sort_keys=True))
            return 0 if payload["trajectory_contract_stage_eligible"] else 2

        if args.task == "trajectory-contract" and args.experiment == "E002":
            input_path = Path(args.input or "trajectory_contract.json")
            output = Path(args.output or "trajectory_index.json")
            payload = _e002_trajectory_contract(input_path=input_path, output=output)
            print(json.dumps({**payload, "artifact": str(output)}, sort_keys=True))
            return 0

        if args.task == "fit-matched-dynamics" and args.experiment == "E002":
            descriptor_path = Path(args.input or "trajectory_contract.json")
            trajectory_index_path = Path(args.trajectory_index or "trajectory_index.json")
            output = Path(args.output or "matched_dynamics.json")
            payload = _e002_fit_matched_dynamics(
                descriptor_path=descriptor_path,
                trajectory_index_path=trajectory_index_path,
                output=output,
                ridge=float(args.ridge),
            )
            print(json.dumps({**payload, "artifact": str(output)}, sort_keys=True))
            return 0
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, sort_keys=True))
        return 2

    print(
        json.dumps(
            {
                "status": "not_implemented",
                "task": args.task,
                "experiment": args.experiment,
                "qualifier": args.qualifier,
                "message": (
                    "This manifest stage is a reviewed orchestration contract, but its "
                    "dataset/model executor is not implemented in this release boundary."
                ),
            },
            sort_keys=True,
        )
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
