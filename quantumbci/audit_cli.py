"""Installed mechanism-equivalence and adversarial benchmark CLI for QuantumBCI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .benchmarking import IndexSplit, benchmark_e001_embeddings
from .dynamics_equivalence import (
    audit_lindblad_gauge_nonidentifiability,
    audit_qubit_lindblad_affine_equivalence,
)
from .e002_synthetic import (
    CanonicalQubitParameters,
    canonical_qubit_model,
    run_e002_synthetic_recovery_grid,
)
from .equivalence import audit_density_covariance_equivalence, audit_embedding_batch


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _density(args: argparse.Namespace) -> int:
    values = np.load(args.embeddings, allow_pickle=False)
    if values.ndim == 2:
        result = audit_density_covariance_equivalence(
            values,
            center=not args.no_center,
            atol=args.atol,
        ).to_mapping()
        result["scope"] = "single_samples_by_features_matrix"
    elif values.ndim == 3:
        result = audit_embedding_batch(
            values,
            center_tokens=not args.no_center,
            atol=args.atol,
        ).to_mapping()
        result["scope"] = "examples_by_tokens_by_features_batch"
    else:
        raise ValueError(
            "density audit expects samples×features or examples×tokens×features; "
            f"got shape {values.shape}"
        )
    result["input"] = str(Path(args.embeddings))
    result["claim_class"] = "quantum_inspired"
    result["promotion_eligible_as_new_information"] = bool(result["novel_information"])
    if args.json:
        _print_json(result)
    else:
        print("QuantumBCI density equivalence audit")
        print(f"scope: {result['scope']}")
        print(f"equivalence class: {result['equivalence_class']}")
        print(f"equivalent within tolerance: {result['equivalent_within_tolerance']}")
        print(f"novel representation information: {result['novel_information']}")
        print(f"max absolute error: {result['max_abs_error']:.3e}")
        print("interpretation: density notation alone does not add information beyond the matched Hermitian second moment")
    return 0


def _e001(args: argparse.Namespace) -> int:
    embeddings = np.load(args.embeddings, allow_pickle=False)
    labels = np.load(args.labels, allow_pickle=False)
    train = np.load(args.train_indices, allow_pickle=False)
    test = np.load(args.test_indices, allow_pickle=False)
    split = IndexSplit(train, test, name=args.split_name)
    result = benchmark_e001_embeddings(
        embeddings,
        labels,
        split,
        ridge=args.ridge,
        center_tokens=not args.no_center,
        covariance_regularization=args.covariance_regularization,
    )
    payload = result.to_mapping(include_predictions=args.include_predictions)
    payload["inputs"] = {
        "embeddings": str(Path(args.embeddings)),
        "labels": str(Path(args.labels)),
        "train_indices": str(Path(args.train_indices)),
        "test_indices": str(Path(args.test_indices)),
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        _print_json(payload)
    else:
        print("QuantumBCI E001 adversarial representation audit")
        print(f"split: {result.split_name}")
        print(f"density BA: {result.metrics['density'].balanced_accuracy:.3f}")
        print(
            "normalized covariance BA: "
            f"{result.metrics['normalized_covariance'].balanced_accuracy:.3f}"
        )
        print(f"strongest classical control: {result.strongest_classical_control}")
        print(f"density - strongest control: {result.density_minus_strongest_control:+.3f}")
        print(f"density - off-diagonal intervention: {result.density_minus_ablation:+.3f}")
        print(f"density information novel: {result.density_information_novel}")
        print("promotion note: the current density constructor is information-equivalent to normalized covariance")
        if args.output:
            print(f"written: {args.output}")
    return 0


def _canonical_parameters(args: argparse.Namespace) -> CanonicalQubitParameters:
    return CanonicalQubitParameters(
        omega_x=float(args.omega_x),
        omega_z=float(args.omega_z),
        gamma_dephasing=float(args.gamma_dephasing),
        gamma_relaxation=float(args.gamma_relaxation),
    )


def _dynamics(args: argparse.Namespace) -> int:
    parameters = _canonical_parameters(args)
    hamiltonian, collapses = canonical_qubit_model(parameters)
    equivalence = audit_qubit_lindblad_affine_equivalence(
        hamiltonian,
        collapses,
        atol=float(args.atol),
    )
    gauge = audit_lindblad_gauge_nonidentifiability(
        hamiltonian,
        collapses,
        atol=float(args.atol),
    )
    payload = {
        "schema_version": 1,
        "claim_class": "quantum_inspired",
        "canonical_parameters": parameters.to_mapping(),
        "affine_equivalence": equivalence.to_mapping(),
        "gauge_nonidentifiability": gauge.to_mapping(),
        "dynamical_information_novel": False,
        "promotion_interpretation": (
            "The fully observed qubit Lindblad trajectory compiles exactly to a "
            "three-dimensional classical affine ODE. Canonical Lindblad parameters may "
            "still be useful as constrained coordinates, but arbitrary Hamiltonian/collapse "
            "matrix entries are not separately identifiable."
        ),
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        _print_json(payload)
    else:
        print("QuantumBCI E002 qubit dynamics equivalence audit")
        print(f"affine equivalence: {equivalence.equivalent_within_tolerance}")
        print(f"max generator error: {equivalence.max_generator_error:.3e}")
        print(f"max trajectory error: {equivalence.max_trajectory_error:.3e}")
        print(f"gauge witnesses pass: {gauge.equivalent_within_tolerance}")
        print("dynamical information novel: False")
        if args.output:
            print(f"written: {args.output}")
    return 0


def _e002_synthetic(args: argparse.Namespace) -> int:
    payload = run_e002_synthetic_recovery_grid(
        seed=int(args.seed),
        noise_std=float(args.noise_std),
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        _print_json(payload)
    else:
        print("QuantumBCI E002 synthetic identifiability audit")
        print(f"cases: {payload['n_cases']}")
        print(
            "median normalized recovery error: "
            f"{payload['median_normalized_recovery_error']:.4f}"
        )
        print(f"sign inversions: {payload['systematic_sign_inversions']}")
        print(f"affine equivalence: {payload['affine_equivalence_pass']}")
        print(f"gauge audit: {payload['gauge_nonidentifiability_witness_pass']}")
        print(f"synthetic gate pass: {payload['synthetic_identifiability_gate_pass']}")
        print("dynamical information novel: False")
        if args.output:
            print(f"written: {args.output}")
    return 0 if payload["synthetic_identifiability_gate_pass"] else 2


def _add_canonical_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--omega-x", type=float, default=1.2)
    parser.add_argument("--omega-z", type=float, default=0.8)
    parser.add_argument("--gamma-dephasing", type=float, default=0.25)
    parser.add_argument("--gamma-relaxation", type=float, default=0.35)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantumbci-audit",
        description=(
            "Detect classical equivalences before interpreting quantum-structured neural mechanisms."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    density = subparsers.add_parser(
        "density",
        help="audit density_from_samples against its exact normalized second-moment control",
    )
    density.add_argument("embeddings", help="2D samples×features or 3D examples×tokens×features .npy")
    density.add_argument("--atol", type=float, default=1e-10)
    density.add_argument("--no-center", action="store_true")
    density.add_argument("--json", action="store_true")
    density.set_defaults(func=_density)

    e001 = subparsers.add_parser(
        "e001",
        help="run the full E001 density-vs-classical representation control gauntlet",
    )
    e001.add_argument("embeddings", help="examples×tokens×features .npy")
    e001.add_argument("labels", help="one label per example .npy")
    e001.add_argument("--train-indices", required=True)
    e001.add_argument("--test-indices", required=True)
    e001.add_argument("--split-name", default="explicit")
    e001.add_argument("--ridge", type=float, default=1e-3)
    e001.add_argument("--covariance-regularization", type=float, default=1e-6)
    e001.add_argument("--no-center", action="store_true")
    e001.add_argument("--include-predictions", action="store_true")
    e001.add_argument("--output")
    e001.add_argument("--json", action="store_true")
    e001.set_defaults(func=_e001)

    dynamics = subparsers.add_parser(
        "dynamics",
        help="compile a canonical qubit Lindblad model to its exact affine Bloch control",
    )
    _add_canonical_arguments(dynamics)
    dynamics.add_argument("--atol", type=float, default=1e-9)
    dynamics.add_argument("--output")
    dynamics.add_argument("--json", action="store_true")
    dynamics.set_defaults(func=_dynamics)

    e002 = subparsers.add_parser(
        "e002-synthetic",
        help="run the gauge-fixed moderate-SNR E002 synthetic recovery grid",
    )
    e002.add_argument("--seed", type=int, default=2027)
    e002.add_argument("--noise-std", type=float, default=0.003)
    e002.add_argument("--output")
    e002.add_argument("--json", action="store_true")
    e002.set_defaults(func=_e002_synthetic)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
