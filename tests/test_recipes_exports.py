from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from quantumbci.cli import main
from quantumbci.exporting import (
    export_run_bids_derivative_container,
    export_run_ro_crate,
    verify_run_artifacts,
)
from quantumbci.recipes import load_recipe, run_recipe, write_recipe_template
from quantumbci.workbench import WorkbenchConfig


def _write_study_inputs(root: Path) -> Path:
    rng = np.random.default_rng(12)
    windows = []
    labels = []
    t = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    for index in range(80):
        label = index % 2
        a = np.sin(t + 0.09 * index)
        sign = 1.0 if label else -1.0
        window = np.stack([a, sign * a, np.cos(2 * t), np.sin(3 * t)], axis=1)
        window += rng.normal(0.0, 0.02, size=window.shape)
        windows.append(window)
        labels.append(label)
    np.save(root / "embeddings.npy", np.stack(windows))
    np.save(root / "labels.npy", np.asarray(labels))
    np.save(root / "train_indices.npy", np.arange(60))
    np.save(root / "test_indices.npy", np.arange(60, 80))
    recipe_path = root / "recipe.json"
    recipe_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "public-density-study",
                "title": "Public density study",
                "claim_class": "quantum_inspired",
                "evidence_tier": "exploratory",
                "source_dataset": "doi:10.example/dataset",
                "source_model": "example-encoder@abc123",
                "data": {
                    "embeddings": "embeddings.npy",
                    "labels": "labels.npy",
                    "train_indices": "train_indices.npy",
                    "test_indices": "test_indices.npy",
                    "split_name": "fixed-heldout",
                },
                "benchmark": {"ridge": 0.001},
            }
        ),
        encoding="utf-8",
    )
    return recipe_path


def test_recipe_resolves_relative_inputs_and_runs(tmp_path: Path) -> None:
    recipe_path = _write_study_inputs(tmp_path)
    recipe = load_recipe(recipe_path)
    assert recipe.embeddings == (tmp_path / "embeddings.npy").resolve()
    assert recipe.input_fingerprints()["embeddings"]["sha256"]

    config = WorkbenchConfig(artifact_root=tmp_path / "runs", source_sha="test-frontier")
    result = run_recipe(recipe_path, config)
    assert result.metrics["density"]["balanced_accuracy"] >= 0.95
    assert result.metrics["density_minus_ablation"] >= 0.25
    assert verify_run_artifacts(result.run_dir)["valid"] is True
    assert (result.run_dir / "recipe.json").exists()
    assert (result.run_dir / "inputs.json").exists()
    assert (result.run_dir / "report.html").exists()


def test_recipe_rejects_stronger_claim_class(tmp_path: Path) -> None:
    recipe_path = _write_study_inputs(tmp_path)
    payload = json.loads(recipe_path.read_text())
    payload["claim_class"] = "physical_quantum"
    recipe_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="quantum_inspired"):
        load_recipe(recipe_path)


def test_ro_crate_and_bids_aware_exports(tmp_path: Path) -> None:
    recipe_path = _write_study_inputs(tmp_path)
    result = run_recipe(
        recipe_path,
        WorkbenchConfig(artifact_root=tmp_path / "runs", source_sha="export-test"),
    )

    crate = export_run_ro_crate(result.run_dir, tmp_path / "crate")
    metadata = json.loads((crate / "ro-crate-metadata.json").read_text())
    assert metadata["@context"] == "https://w3id.org/ro/crate/1.3/context"
    graph = {entity["@id"]: entity for entity in metadata["@graph"]}
    assert graph["ro-crate-metadata.json"]["conformsTo"]["@id"] == "https://w3id.org/ro/crate/1.3"
    assert graph["."]["@type"] == "Dataset"
    assert (crate / "ro-crate-preview.html").exists()
    assert (crate / "data" / "run.json").exists()

    bids_target = export_run_bids_derivative_container(
        result.run_dir,
        tmp_path / "bids",
        bids_version="1.10.1",
        source_dataset_url="https://example.org/source-dataset",
    )
    description = json.loads(
        (tmp_path / "bids" / "derivatives" / "quantumbci" / "dataset_description.json").read_text()
    )
    assert description["DatasetType"] == "derivative"
    assert description["GeneratedBy"][0]["Name"] == "QuantumBCI"
    assert bids_target.joinpath("run.json").exists()


def test_tampering_blocks_export(tmp_path: Path) -> None:
    recipe_path = _write_study_inputs(tmp_path)
    result = run_recipe(
        recipe_path,
        WorkbenchConfig(artifact_root=tmp_path / "runs", source_sha="tamper-test"),
    )
    (result.run_dir / "metrics.json").write_text("{}\n", encoding="utf-8")
    verification = verify_run_artifacts(result.run_dir)
    assert verification["valid"] is False
    assert "metrics.json" in verification["mismatched"]
    with pytest.raises(ValueError, match="verification failed"):
        export_run_ro_crate(result.run_dir, tmp_path / "crate")


def test_cli_recipe_verify_and_export(tmp_path: Path, capsys) -> None:
    recipe_path = _write_study_inputs(tmp_path)
    config_path = tmp_path / "quantumbci.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_root": "runs",
                "default_seed": 0,
                "source_sha": "cli-recipe-test",
            }
        )
    )

    assert main(["recipe", "validate", str(recipe_path), "--json"]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["valid"] is True
    assert validated["inputs"]["embeddings"]["sha256"]

    assert main(["recipe", "run", str(recipe_path), "--config", str(config_path), "--json"]) == 0
    run = json.loads(capsys.readouterr().out)
    run_id = run["run_id"]

    assert main(["runs", "verify", run_id, "--config", str(config_path), "--json"]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["valid"] is True

    crate_path = tmp_path / "shared-crate"
    assert main([
        "runs",
        "export",
        run_id,
        "--config",
        str(config_path),
        "--format",
        "ro-crate",
        "--output",
        str(crate_path),
        "--json",
    ]) == 0
    exported = json.loads(capsys.readouterr().out)
    assert Path(exported["export"]).joinpath("ro-crate-metadata.json").exists()


def test_recipe_template_is_safe_by_default(tmp_path: Path) -> None:
    path = write_recipe_template(tmp_path / "recipe.json")
    payload = json.loads(path.read_text())
    assert payload["claim_class"] == "quantum_inspired"
    assert payload["evidence_tier"] == "exploratory"
