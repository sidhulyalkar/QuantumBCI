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
from quantumbci.recipes import load_recipe, preflight_recipe, run_recipe, write_recipe_template
from quantumbci.workbench import WorkbenchConfig


def _write_study_inputs(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
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


def test_recipe_resolves_relative_inputs_preflights_and_runs(tmp_path: Path) -> None:
    recipe_path = _write_study_inputs(tmp_path)
    recipe = load_recipe(recipe_path)
    assert recipe.embeddings == (tmp_path / "embeddings.npy").resolve()
    assert recipe.input_fingerprints()["embeddings"]["sha256"]
    preflight = preflight_recipe(recipe)
    assert preflight["embeddings"]["shape"] == [80, 32, 4]
    assert preflight["n_train"] == 60
    assert preflight["n_test"] == 20

    config = WorkbenchConfig(artifact_root=tmp_path / "runs", source_sha="test-frontier")
    result = run_recipe(recipe_path, config)
    assert result.metrics["density"]["balanced_accuracy"] >= 0.95
    assert result.metrics["density_minus_ablation"] >= 0.25
    verification = verify_run_artifacts(result.run_dir)
    assert verification["valid"] is True
    assert verification["unexpected"] == []
    assert (result.run_dir / "recipe.json").exists()
    assert (result.run_dir / "inputs.json").exists()
    assert (result.run_dir / "report.html").exists()


def test_recipe_scientific_identity_is_content_addressed_not_filename_addressed(tmp_path: Path) -> None:
    first_path = _write_study_inputs(tmp_path / "lab-a")
    second_path = _write_study_inputs(tmp_path / "lab-b")
    second_payload = json.loads(second_path.read_text())
    rename_map = {
        "embeddings": "model_tokens.npy",
        "labels": "targets.npy",
        "train_indices": "fit_rows.npy",
        "test_indices": "heldout_rows.npy",
    }
    for role, new_name in rename_map.items():
        old_name = second_payload["data"][role]
        (second_path.parent / old_name).rename(second_path.parent / new_name)
        second_payload["data"][role] = new_name
    second_path.write_text(json.dumps(second_payload), encoding="utf-8")

    first = load_recipe(first_path)
    second = load_recipe(second_path)
    assert first.input_fingerprints()["embeddings"]["filename"] != second.input_fingerprints()["embeddings"]["filename"]
    assert first.scientific_input_fingerprints() == second.scientific_input_fingerprints()
    assert first.identity_mapping(source_sha="same", array_contract=preflight_recipe(first)) == second.identity_mapping(
        source_sha="same", array_contract=preflight_recipe(second)
    )


def test_recipe_rejects_stronger_claim_class(tmp_path: Path) -> None:
    recipe_path = _write_study_inputs(tmp_path)
    payload = json.loads(recipe_path.read_text())
    payload["claim_class"] = "physical_quantum"
    recipe_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="quantum_inspired"):
        load_recipe(recipe_path)


def test_recipe_validate_fails_before_fit_on_bad_split(tmp_path: Path, capsys) -> None:
    recipe_path = _write_study_inputs(tmp_path)
    np.save(tmp_path / "test_indices.npy", np.arange(50, 80))
    assert main(["recipe", "validate", str(recipe_path), "--json"]) == 2
    captured = capsys.readouterr()
    assert "overlap" in captured.err


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
        bids_version="1.11.1",
        source_dataset_url="https://example.org/source-dataset",
    )
    description = json.loads(
        (tmp_path / "bids" / "derivatives" / "quantumbci" / "dataset_description.json").read_text()
    )
    assert description["DatasetType"] == "derivative"
    assert description["BIDSVersion"] == "1.11.1"
    assert description["GeneratedBy"][0]["Name"] == "QuantumBCI"
    assert description["GeneratedBy"][0]["CodeURL"].endswith("/QuantumBCI")
    assert bids_target.joinpath("run.json").exists()
    export_contract = json.loads(bids_target.joinpath("quantumbci_export.json").read_text())
    assert export_contract["standardized_modality_derivative"] is False


def test_bids_container_rejects_version_drift(tmp_path: Path) -> None:
    first_recipe = _write_study_inputs(tmp_path / "first")
    second_recipe = _write_study_inputs(tmp_path / "second")
    first = run_recipe(first_recipe, WorkbenchConfig(artifact_root=tmp_path / "runs-a", source_sha="a"))
    second = run_recipe(second_recipe, WorkbenchConfig(artifact_root=tmp_path / "runs-b", source_sha="b"))
    export_run_bids_derivative_container(first.run_dir, tmp_path / "bids", bids_version="1.11.1")
    with pytest.raises(ValueError, match="different BIDSVersion"):
        export_run_bids_derivative_container(second.run_dir, tmp_path / "bids", bids_version="1.10.1")


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


def test_untracked_artifact_blocks_export(tmp_path: Path) -> None:
    recipe_path = _write_study_inputs(tmp_path)
    result = run_recipe(recipe_path, WorkbenchConfig(artifact_root=tmp_path / "runs", source_sha="extra-file"))
    (result.run_dir / "untracked-analysis.txt").write_text("post-hoc edit\n", encoding="utf-8")
    verification = verify_run_artifacts(result.run_dir)
    assert verification["valid"] is False
    assert verification["unexpected"] == ["untracked-analysis.txt"]
    with pytest.raises(ValueError, match="verification failed"):
        export_run_ro_crate(result.run_dir, tmp_path / "crate")


def test_unsafe_ledger_entry_is_rejected(tmp_path: Path) -> None:
    recipe_path = _write_study_inputs(tmp_path)
    result = run_recipe(recipe_path, WorkbenchConfig(artifact_root=tmp_path / "runs", source_sha="ledger-test"))
    ledger_path = result.run_dir / "artifact_hashes.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["../outside.txt"] = "0" * 64
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    verification = verify_run_artifacts(result.run_dir)
    assert verification["valid"] is False
    assert "../outside.txt" in verification["invalid_ledger_entries"]


def test_recipe_report_escapes_user_metadata(tmp_path: Path) -> None:
    recipe_path = _write_study_inputs(tmp_path)
    payload = json.loads(recipe_path.read_text())
    payload["title"] = "<script>alert('x')</script>"
    payload["source_model"] = "<img src=x onerror=alert(1)>"
    recipe_path.write_text(json.dumps(payload), encoding="utf-8")
    result = run_recipe(recipe_path, WorkbenchConfig(artifact_root=tmp_path / "runs", source_sha="html-test"))
    report = (result.run_dir / "report.html").read_text()
    assert "<script>" not in report
    assert "<img src=x" not in report
    assert "&lt;script&gt;" in report


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
    assert verified["unexpected"] == []

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
