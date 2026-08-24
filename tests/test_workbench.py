from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from quantumbci.cli import main
from quantumbci.experiments.manifest import load_manifest
from quantumbci.workbench import (
    RunStore,
    WorkbenchConfig,
    find_manifest_files,
    run_density_smoke,
)


def test_smoke_materializes_a_self_describing_run(tmp_path: Path) -> None:
    config = WorkbenchConfig(
        artifact_root=tmp_path / "runs",
        default_seed=3,
        source_sha="test-source",
    )
    result = run_density_smoke(config)
    assert result.metrics["density_balanced_accuracy"] >= 0.95
    assert result.metrics["density_minus_ablated"] >= 0.25
    for name in (
        "run.json",
        "study_manifest.json",
        "metrics.json",
        "mechanism.json",
        "predictions.jsonl",
        "artifact_hashes.json",
        "report.md",
        "report.html",
    ):
        assert (result.run_dir / name).exists()

    record = RunStore(config.artifact_root).load(result.run_id)
    assert record["status"] == "completed"
    assert record["claim_class"] == "quantum_inspired"
    assert record["evidence_tier"] == "synthetic_sanity"
    hashes = json.loads((result.run_dir / "artifact_hashes.json").read_text())
    assert "metrics.json" in hashes
    assert "report.html" in hashes


def test_cli_init_doctor_smoke_and_runs(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "quantumbci.json"
    assert main(["init", str(config_path)]) == 0
    capsys.readouterr()  # isolate the next machine-readable command
    payload = json.loads(config_path.read_text())
    payload["artifact_root"] = "runs"
    payload["source_sha"] = "cli-test"
    config_path.write_text(json.dumps(payload))

    assert main(["doctor", "--config", str(config_path), "--json"]) == 0
    doctor = json.loads(capsys.readouterr().out)
    assert doctor["experiment_manifests"] >= 4

    assert main(["smoke", "--config", str(config_path), "--seed", "7", "--json"]) == 0
    smoke_output = json.loads(capsys.readouterr().out)
    assert Path(smoke_output["run_dir"]).exists()

    assert main(["runs", "list", "--config", str(config_path), "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1
    assert rows[0]["run_id"] == smoke_output["run_id"]


def test_cli_benchmarks_npy_embeddings(tmp_path: Path, capsys) -> None:
    rng = np.random.default_rng(4)
    windows = []
    labels = []
    t = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    for index in range(80):
        label = index % 2
        a = np.sin(t + 0.11 * index)
        sign = 1.0 if label else -1.0
        window = np.stack([a, sign * a, np.cos(2 * t), np.sin(3 * t)], axis=1)
        window += rng.normal(0.0, 0.02, size=window.shape)
        windows.append(window)
        labels.append(label)

    embeddings_path = tmp_path / "embeddings.npy"
    labels_path = tmp_path / "labels.npy"
    train_path = tmp_path / "train.npy"
    test_path = tmp_path / "test.npy"
    output_path = tmp_path / "result.json"
    np.save(embeddings_path, np.stack(windows))
    np.save(labels_path, np.asarray(labels))
    np.save(train_path, np.arange(60))
    np.save(test_path, np.arange(60, 80))

    assert main([
        "benchmark",
        str(embeddings_path),
        str(labels_path),
        "--train-indices",
        str(train_path),
        "--test-indices",
        str(test_path),
        "--split-name",
        "fixed-test",
        "--output",
        str(output_path),
        "--json",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["split_name"] == "fixed-test"
    assert result["density"]["balanced_accuracy"] >= 0.95
    assert result["density_minus_ablation"] >= 0.25
    assert output_path.exists()


def test_cli_validates_and_plans_committed_manifest(tmp_path: Path, capsys) -> None:
    manifest = "experiments/manifests/E001_density_geometry.json"
    assert main(["experiments", "validate", manifest, "--json"]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["valid"] is True
    assert validated["claim_class"] == "quantum_inspired"

    output = tmp_path / "plan"
    assert main([
        "experiments",
        "plan",
        manifest,
        "--source-sha",
        "abc123",
        "--output",
        str(output),
        "--json",
    ]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["source_sha"] == "abc123"
    assert (output / "plan.json").exists()


def test_packaged_manifests_match_source_registry() -> None:
    source = Path("experiments/manifests")
    packaged = Path("quantumbci/experiments/manifests")
    source_paths = sorted(source.glob("*.json"))
    assert len(source_paths) >= 4
    for source_path in source_paths:
        packaged_path = packaged / source_path.name
        assert packaged_path.exists(), f"missing packaged manifest: {source_path.name}"
        assert load_manifest(source_path).digest == load_manifest(packaged_path).digest


def test_manifest_discovery_exposes_the_packaged_catalog() -> None:
    paths = find_manifest_files()
    names = {path.name for path in paths}
    assert {
        "E001_density_geometry.json",
        "E002_lindblad_dynamics.json",
        "E003_contextual_order.json",
        "E004_quantum_resource_sandbox.json",
    } <= names
