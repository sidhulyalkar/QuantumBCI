"""Experiment-stage task entry point.

Most manifest stages remain deliberate reviewed contracts until their dataset/model
executors land. Implemented mathematical and synthetic gates may run here when they
can be qualified without downloading data or fabricating scientific artifacts.
Unknown or unfinished stages fail explicitly with exit code 3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..e002_synthetic import run_e002_synthetic_recovery_grid
from ..equivalence import audit_density_covariance_equivalence


def _write_json(output: Path, payload: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _density_covariance_gate(*, output: Path, atol: float) -> dict[str, Any]:
    """Materialize the exact E001 density/covariance equivalence contract.

    The identity ``rho = X^H X / Tr(X^H X)`` is algebraic. The deterministic
    real/complex probes below are regression witnesses that our implementation
    continues to realize that identity within numerical tolerance. They are not
    empirical neuroscience evidence.
    """

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
            audit = audit_density_covariance_equivalence(
                values,
                center=center,
                atol=atol,
            )
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


def _e002_synthetic_recovery(
    *,
    output: Path,
    seed: int,
    noise_std: float,
) -> dict[str, Any]:
    payload = run_e002_synthetic_recovery_grid(
        seed=int(seed),
        noise_std=float(noise_std),
    )
    payload = {
        **payload,
        "status": "pass" if payload["synthetic_identifiability_gate_pass"] else "fail",
        "artifact_role": "synthetic_recovery_witness",
    }
    _write_json(output, payload)
    return payload


def _e002_identifiability_gate(
    *,
    input_path: Path,
    output: Path,
) -> dict[str, Any]:
    if not input_path.is_file():
        raise FileNotFoundError(f"synthetic recovery artifact not found: {input_path}")
    source = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or source.get("experiment") != "E002":
        raise ValueError("identifiability gate requires an E002 synthetic recovery artifact")

    median_error = float(source.get("median_normalized_recovery_error", float("inf")))
    sign_inversions = int(source.get("systematic_sign_inversions", -1))
    affine_equivalence = bool(source.get("affine_equivalence_pass", False))
    gauge_witness = bool(source.get("gauge_nonidentifiability_witness_pass", False))
    canonical_recovery = bool(source.get("synthetic_identifiability_gate_pass", False))
    passed = bool(
        median_error <= 0.20
        and sign_inversions == 0
        and affine_equivalence
        and gauge_witness
        and canonical_recovery
    )
    payload = {
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "experiment": "E002",
        "gate": "synthetic-identifiability",
        "claim_class": "quantum_inspired",
        "source_artifact": str(input_path),
        "criteria": {
            "median_normalized_recovery_error_max": 0.20,
            "systematic_sign_inversions_max": 0,
            "require_affine_equivalence_witness": True,
            "require_gauge_nonidentifiability_witness": True,
        },
        "observed": {
            "median_normalized_recovery_error": median_error,
            "systematic_sign_inversions": sign_inversions,
            "affine_equivalence_pass": affine_equivalence,
            "gauge_nonidentifiability_witness_pass": gauge_witness,
            "canonical_recovery_pass": canonical_recovery,
        },
        "trajectory_contract_stage_eligible": passed,
        "dynamical_information_novel": False,
        "physical_quantum_promotion_eligible": False,
        "interpretation": (
            "Passing this gate means a declared gauge-fixed canonical parameter family "
            "can be recovered from moderate-noise synthetic trajectories. It does not "
            "make the qubit Lindblad trajectory information-distinct from its exact "
            "classical affine Bloch representation and does not support a physical neural "
            "quantum claim."
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
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--noise-std", type=float, default=0.003)
    args = parser.parse_args(argv)

    try:
        if (
            args.task == "equivalence-audit"
            and args.experiment == "E001"
            and args.qualifier == "density-covariance"
        ):
            output = Path(args.output or "equivalence_audit.json")
            payload = _density_covariance_gate(output=output, atol=float(args.atol))
            payload = {**payload, "artifact": str(output)}
            print(json.dumps(payload, sort_keys=True))
            return 0 if payload["equivalent_within_tolerance"] else 2

        if args.task == "synthetic-recovery" and args.experiment == "E002":
            output = Path(args.output or "synthetic_recovery.json")
            payload = _e002_synthetic_recovery(
                output=output,
                seed=int(args.seed),
                noise_std=float(args.noise_std),
            )
            print(json.dumps({**payload, "artifact": str(output)}, sort_keys=True))
            return 0 if payload["synthetic_identifiability_gate_pass"] else 2

        if (
            args.task == "gate"
            and args.experiment == "E002"
            and args.qualifier == "identifiability"
        ):
            input_path = Path(args.input or "synthetic_recovery.json")
            output = Path(args.output or "identifiability_gate.json")
            payload = _e002_identifiability_gate(
                input_path=input_path,
                output=output,
            )
            print(json.dumps({**payload, "artifact": str(output)}, sort_keys=True))
            return 0 if payload["trajectory_contract_stage_eligible"] else 2
    except (FileNotFoundError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
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
