from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantumbci.experiments.manifest import ExperimentManifest, ManifestError, load_manifest
from quantumbci.experiments.orchestration import build_plan, render_plan, topological_layers


def example_mapping() -> dict:
    return {
        "schema_version": 1,
        "id": "EXAMPLE",
        "title": "Example",
        "claim_class": "quantum_inspired",
        "objective": "Validate orchestration.",
        "datasets": ["synthetic"],
        "encoders": ["none"],
        "primary_metrics": ["error"],
        "stages": [
            {
                "id": "prepare",
                "kind": "data",
                "depends_on": [],
                "command": ["python", "prepare.py"],
                "resources": {"cpu": 1},
                "produces": ["prepared.json"],
            },
            {
                "id": "fit_a",
                "kind": "model",
                "depends_on": ["prepare"],
                "command": ["python", "fit.py", "a"],
                "resources": {"cpu": 1},
                "produces": ["a.json"],
            },
            {
                "id": "fit_b",
                "kind": "model",
                "depends_on": ["prepare"],
                "command": ["python", "fit.py", "b"],
                "resources": {"cpu": 1},
                "produces": ["b.json"],
            },
            {
                "id": "report",
                "kind": "report",
                "depends_on": ["fit_a", "fit_b"],
                "command": ["python", "report.py"],
                "resources": {"cpu": 1},
                "produces": ["report.md"],
            },
        ],
        "decision_gates": ["A gate must exist."],
    }


def test_topological_layers_preserve_parallelism() -> None:
    manifest = ExperimentManifest.from_mapping(example_mapping())
    layers = [[stage.id for stage in layer] for layer in topological_layers(manifest)]
    assert layers == [["prepare"], ["fit_a", "fit_b"], ["report"]]


def test_manifest_digest_is_stable() -> None:
    first = ExperimentManifest.from_mapping(example_mapping())
    second = ExperimentManifest.from_mapping(json.loads(json.dumps(example_mapping())))
    assert first.digest == second.digest


def test_build_plan_binds_source_revision() -> None:
    manifest = ExperimentManifest.from_mapping(example_mapping())
    a = build_plan(manifest, "abc123")
    b = build_plan(manifest, "def456")
    assert a["plan_id"] != b["plan_id"]
    assert a["source_sha"] == "abc123"
    assert "dataset_fingerprint" in a["final_run_id_note"]


def test_unknown_dependency_is_rejected() -> None:
    value = example_mapping()
    value["stages"][1]["depends_on"] = ["missing"]
    with pytest.raises(ManifestError, match="unknown dependencies"):
        ExperimentManifest.from_mapping(value)


def test_cycle_is_rejected() -> None:
    value = example_mapping()
    value["stages"][0]["depends_on"] = ["report"]
    with pytest.raises(ManifestError, match="cycle"):
        ExperimentManifest.from_mapping(value)


def test_duplicate_artifact_owner_is_rejected() -> None:
    value = example_mapping()
    value["stages"][2]["produces"] = ["a.json"]
    with pytest.raises(ManifestError, match="produced by both"):
        ExperimentManifest.from_mapping(value)


def test_loader_and_renderer(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(example_mapping()), encoding="utf-8")
    manifest = load_manifest(path)
    rendered = render_plan(manifest, "abc123")
    assert "layer 2: fit_a, fit_b" in rendered
    assert "quantum_inspired" in rendered


def test_bundled_manifest_can_be_loaded_by_stable_id() -> None:
    manifest = load_manifest("E001_density_geometry")
    assert manifest.id == "E001_density_geometry"
    assert manifest.claim_class.value == "quantum_inspired"


def test_unknown_bundled_manifest_gives_discovery_hint() -> None:
    with pytest.raises(FileNotFoundError, match="experiments list"):
        load_manifest("E999_does_not_exist")


def test_repository_manifests_validate() -> None:
    root = Path(__file__).resolve().parents[1]
    manifests = sorted((root / "experiments" / "manifests").glob("*.json"))
    assert len(manifests) >= 4
    loaded = [load_manifest(path) for path in manifests]
    assert {item.id for item in loaded} >= {
        "E001_density_geometry",
        "E002_lindblad_dynamics",
        "E003_contextual_order",
        "E004_quantum_resource_sandbox",
    }
