"""Experiment-stage task entry point.

Most manifest stages remain deliberate reviewed contracts until their dataset/model
executors land. Implemented mathematical gates may run here when they can be
qualified without downloading data or fabricating scientific artifacts. Unknown or
unfinished stages fail explicitly with exit code 3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..equivalence import audit_density_covariance_equivalence


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
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("experiment")
    parser.add_argument("qualifier", nargs="?")
    parser.add_argument("--output")
    parser.add_argument("--atol", type=float, default=1e-10)
    args = parser.parse_args(argv)

    if (
        args.task == "equivalence-audit"
        and args.experiment == "E001"
        and args.qualifier == "density-covariance"
    ):
        output = Path(args.output or "equivalence_audit.json")
        try:
            payload = _density_covariance_gate(output=output, atol=float(args.atol))
        except ValueError as exc:
            print(json.dumps({"status": "error", "message": str(exc)}, sort_keys=True))
            return 2
        payload = {**payload, "artifact": str(output)}
        print(json.dumps(payload, sort_keys=True))
        return 0 if payload["equivalent_within_tolerance"] else 2

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
