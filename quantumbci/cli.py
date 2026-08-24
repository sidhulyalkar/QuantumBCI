"""Command-line interface for the QuantumBCI research workbench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .benchmarking import IndexSplit, benchmark_density_embeddings
from .contextuality import order_effect, projector
from .experiments.manifest import ManifestError, load_manifest
from .experiments.orchestration import build_plan, materialize_plan, render_plan
from .interpretability import mechanism_delta, state_signature
from .open_system import dephasing_collapse, evolve_lindblad
from .states import project_density_matrix
from .workbench import RunStore, doctor_report, find_manifest_files, load_config, run_density_smoke, write_default_config


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _mechanism_demo() -> dict[str, Any]:
    rho0 = project_density_matrix(np.array([[0.5, 0.45], [0.45, 0.5]], dtype=complex))
    hamiltonian = np.array([[0.0, 0.8], [0.8, 0.2]], dtype=complex)
    collapse = [dephasing_collapse(2, 0, 0.7), dephasing_collapse(2, 1, 0.7)]
    trajectory = evolve_lindblad(rho0, hamiltonian, np.linspace(0.0, 2.0, 201), collapse_operators=collapse)
    before = state_signature(trajectory[0])
    after = state_signature(trajectory[-1])
    a = projector([1.0, 0.0])
    b = projector([1.0, 1.0])
    return {
        "lindblad_mechanism_delta": mechanism_delta(before, after),
        "context_order_probe": order_effect(rho0, a, b),
    }


def _config_from_args(args: argparse.Namespace):
    return load_config(getattr(args, "config", None))


def _command_init(args: argparse.Namespace) -> int:
    path = write_default_config(args.path, force=args.force)
    print(f"Created {path}")
    print("Next: quantumbci doctor && quantumbci smoke")
    return 0


def _command_doctor(args: argparse.Namespace) -> int:
    report = doctor_report(_config_from_args(args))
    if args.json:
        _print_json(report)
    else:
        print(f"QuantumBCI doctor: {report['status']}")
        print(f"Python: {report['python']['version']} (supported={report['python']['supported']})")
        print(f"NumPy: {report['numpy']}")
        print(f"Artifact root: {report['artifact_root']} (writable={report['artifact_root_writable']})")
        print("Optional integrations:")
        for name, installed in sorted(report["optional"].items()):
            print(f"  {name}: {installed or 'not installed'}")
        print(f"Source identity: {report['source_sha']}")
    return 0 if report["status"] == "ok" else 2


def _command_smoke(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    if args.output_root is not None:
        config = type(config)(artifact_root=Path(args.output_root), default_seed=config.default_seed, source_sha=config.source_sha)
    result = run_density_smoke(config, seed=args.seed)
    payload = {
        "run_id": result.run_id,
        "run_dir": str(result.run_dir),
        "report": str(result.run_dir / "report.html"),
        "scientific_fingerprint": result.scientific_fingerprint,
        "metrics": {key: value for key, value in result.metrics.items() if key != "per_subject"},
    }
    if args.json:
        _print_json(payload)
    else:
        print("QuantumBCI density smoke: completed")
        print(f"Run: {result.run_id}")
        print(f"Artifacts: {result.run_dir}")
        print(f"Report: {result.run_dir / 'report.html'}")
        print(f"Density BA: {payload['metrics']['density_balanced_accuracy']:.3f}")
        print(f"Diagonal-control BA: {payload['metrics']['diagonal_balanced_accuracy']:.3f}")
        print(f"Off-diagonal ablation BA: {payload['metrics']['offdiagonal_ablated_balanced_accuracy']:.3f}")
        print(f"Mechanism delta: {payload['metrics']['density_minus_ablated']:+.3f}")
        print("Claim ceiling: quantum_inspired / synthetic_sanity")
    return 0


def _command_benchmark(args: argparse.Namespace) -> int:
    embeddings = np.load(args.embeddings, allow_pickle=False)
    labels = np.load(args.labels, allow_pickle=False)
    train_indices = np.load(args.train_indices, allow_pickle=False)
    test_indices = np.load(args.test_indices, allow_pickle=False)
    split = IndexSplit(train_indices=train_indices, test_indices=test_indices, name=args.split_name)
    result = benchmark_density_embeddings(embeddings, labels, split, ridge=args.ridge)
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
        print(f"split: {result.split_name}")
        print(f"classes: {', '.join(result.classes)}")
        print(f"density balanced accuracy: {result.density.balanced_accuracy:.3f}")
        print(f"diagonal-control balanced accuracy: {result.diagonal_control.balanced_accuracy:.3f}")
        print(f"pooled-control balanced accuracy: {result.pooled_control.balanced_accuracy:.3f}")
        print(f"off-diagonal ablation balanced accuracy: {result.offdiagonal_ablation.balanced_accuracy:.3f}")
        print(f"density minus diagonal: {result.density_minus_diagonal:+.3f}")
        print(f"density minus ablation: {result.density_minus_ablation:+.3f}")
        if args.output:
            print(f"written: {args.output}")
        print("Claim ceiling: quantum_inspired; split supplied explicitly by caller")
    return 0


def _command_demo(args: argparse.Namespace) -> int:
    result = _mechanism_demo()
    if args.json:
        _print_json(result)
    else:
        print("Lindblad mechanism delta:", result["lindblad_mechanism_delta"])
        print("Context/order probe:", result["context_order_probe"])
    return 0


def _command_experiments_list(args: argparse.Namespace) -> int:
    rows = []
    for path in find_manifest_files(args.manifest_dir or ()):
        try:
            manifest = load_manifest(path)
            rows.append({
                "id": manifest.id,
                "title": manifest.title,
                "claim_class": manifest.claim_class.value,
                "path": str(path),
                "digest": manifest.digest,
                "valid": True,
            })
        except (ManifestError, OSError, ValueError) as exc:
            rows.append({"id": path.stem, "path": str(path), "valid": False, "error": str(exc)})
    if args.json:
        _print_json(rows)
    else:
        if not rows:
            print("No manifests found. Pass --manifest-dir or run from a source checkout.")
            return 1
        for row in rows:
            if row["valid"]:
                print(f"{row['id']:<28} {row['claim_class']:<18} {row['digest'][:10]}  {row['title']}")
            else:
                print(f"{row['id']:<28} INVALID  {row['error']}")
    return 0 if all(row["valid"] for row in rows) else 2


def _command_experiments_validate(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    payload = {
        "valid": True,
        "id": manifest.id,
        "title": manifest.title,
        "claim_class": manifest.claim_class.value,
        "digest": manifest.digest,
        "stages": len(manifest.stages),
        "decision_gates": len(manifest.decision_gates),
    }
    if args.json:
        _print_json(payload)
    else:
        print(f"valid: {manifest.id}")
        print(f"claim ceiling: {manifest.claim_class.value}")
        print(f"digest: {manifest.digest}")
        print(f"stages: {len(manifest.stages)}")
        print(f"decision gates: {len(manifest.decision_gates)}")
    return 0


def _command_experiments_plan(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    source_sha = args.source_sha or _config_from_args(args).source_sha
    plan = build_plan(manifest, source_sha)
    if args.output is not None:
        path = materialize_plan(manifest, source_sha, args.output)
        plan["materialized_path"] = str(path)
    if args.json:
        _print_json(plan)
    else:
        print(render_plan(manifest, source_sha))
        if args.output is not None:
            print(f"written: {plan['materialized_path']}")
    return 0


def _command_runs_list(args: argparse.Namespace) -> int:
    rows = RunStore(_config_from_args(args).artifact_root).records()
    if args.json:
        _print_json(rows)
    else:
        if not rows:
            print("No local runs found.")
            return 0
        for row in rows:
            metrics = row.get("metrics", {})
            primary = metrics.get("density_balanced_accuracy")
            suffix = f"  density_BA={primary:.3f}" if isinstance(primary, (int, float)) else ""
            print(f"{row.get('run_id', '?')}  {row.get('status', '?'):<10} {row.get('experiment_id', '?')}{suffix}")
    return 0


def _command_runs_show(args: argparse.Namespace) -> int:
    row = RunStore(_config_from_args(args).artifact_root).load(args.run_id)
    if args.json:
        _print_json(row)
    else:
        print(f"run: {row['run_id']}")
        print(f"experiment: {row.get('experiment_id')}")
        print(f"status: {row.get('status')}")
        print(f"claim class: {row.get('claim_class')}")
        print(f"evidence tier: {row.get('evidence_tier')}")
        print(f"fingerprint: {row.get('scientific_fingerprint')}")
        print(f"artifacts: {row.get('run_dir')}")
        for name, value in sorted(row.get("metrics", {}).items()):
            if isinstance(value, (int, float)):
                print(f"{name}: {value:.6f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantumbci", description="Local-first workbench for falsifiable quantum-inspired neural experiments.")
    parser.add_argument("--version", action="store_true", help="print the installed package version and exit")
    subparsers = parser.add_subparsers(dest="command")

    init = subparsers.add_parser("init", help="create a local quantumbci.json config")
    init.add_argument("path", nargs="?", default="quantumbci.json")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=_command_init)

    doctor = subparsers.add_parser("doctor", help="check local research-workbench readiness")
    doctor.add_argument("--config")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=_command_doctor)

    smoke = subparsers.add_parser("smoke", help="run a deterministic end-to-end density-mechanism sanity study")
    smoke.add_argument("--config")
    smoke.add_argument("--seed", type=int)
    smoke.add_argument("--output-root")
    smoke.add_argument("--json", action="store_true")
    smoke.set_defaults(func=_command_smoke)

    benchmark = subparsers.add_parser("benchmark", help="benchmark density geometry on frozen .npy embeddings")
    benchmark.add_argument("embeddings", help="examples x tokens x features .npy array")
    benchmark.add_argument("labels", help="one label per example .npy array")
    benchmark.add_argument("--train-indices", required=True, help="explicit train-index .npy array")
    benchmark.add_argument("--test-indices", required=True, help="explicit test-index .npy array")
    benchmark.add_argument("--split-name", default="explicit")
    benchmark.add_argument("--ridge", type=float, default=1e-3)
    benchmark.add_argument("--output")
    benchmark.add_argument("--include-predictions", action="store_true")
    benchmark.add_argument("--json", action="store_true")
    benchmark.set_defaults(func=_command_benchmark)

    demo = subparsers.add_parser("demo", help="run the original compact mechanism demo")
    demo.add_argument("--json", action="store_true")
    demo.set_defaults(func=_command_demo)

    experiments = subparsers.add_parser("experiments", help="inspect experiment contracts")
    experiment_sub = experiments.add_subparsers(dest="experiments_command", required=True)
    exp_list = experiment_sub.add_parser("list", help="list discovered experiment manifests")
    exp_list.add_argument("--manifest-dir", action="append")
    exp_list.add_argument("--json", action="store_true")
    exp_list.set_defaults(func=_command_experiments_list)
    exp_validate = experiment_sub.add_parser("validate", help="validate one experiment manifest")
    exp_validate.add_argument("manifest")
    exp_validate.add_argument("--json", action="store_true")
    exp_validate.set_defaults(func=_command_experiments_validate)
    exp_plan = experiment_sub.add_parser("plan", help="build a deterministic experiment plan")
    exp_plan.add_argument("manifest")
    exp_plan.add_argument("--config")
    exp_plan.add_argument("--source-sha")
    exp_plan.add_argument("--output")
    exp_plan.add_argument("--json", action="store_true")
    exp_plan.set_defaults(func=_command_experiments_plan)

    runs = subparsers.add_parser("runs", help="inspect local run artifacts")
    run_sub = runs.add_subparsers(dest="runs_command", required=True)
    run_list = run_sub.add_parser("list", help="list local workbench runs")
    run_list.add_argument("--config")
    run_list.add_argument("--json", action="store_true")
    run_list.set_defaults(func=_command_runs_list)
    run_show = run_sub.add_parser("show", help="show one local run")
    run_show.add_argument("run_id")
    run_show.add_argument("--config")
    run_show.add_argument("--json", action="store_true")
    run_show.set_defaults(func=_command_runs_show)
    return parser


def _installed_version() -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("quantum-bci")
    except PackageNotFoundError:
        return "source-checkout"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print(_installed_version())
        return 0
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    try:
        return int(args.func(args))
    except (ManifestError, FileNotFoundError, FileExistsError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
