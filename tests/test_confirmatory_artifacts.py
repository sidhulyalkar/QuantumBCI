from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from quantumbci.confirmatory_representation import load_confirmatory_representation_manifest


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _lane(root: Path, lane: str, family: str, *, model: str | None = None) -> Path:
    directory = root / lane
    directory.mkdir()
    scientific = ("a" if lane == "raw" else "b") * 64
    representation_sha = ("c" if lane == "raw" else "d") * 64
    run = {
        "schema_version": 1,
        "run_id": f"run-{lane}",
        "scientific_fingerprint": scientific,
    }
    manifest = {
        "schema_version": 1,
        "artifact_role": "quantumbci.e001-representation-lane.v1",
        "study_id": f"study-{lane}",
        "representation_id": f"source-{lane}",
        "representation_family": family,
        "model_id": model,
        "model_revision": None if model is None else "pinned-rev",
        "scientific_fingerprint": scientific,
    }
    cases = []
    for participant in ("p1", "p2", "p3"):
        authority = f"authority-{participant}"
        rows = []
        for budget in (0, 10):
            rows.append(
                {
                    "schema_version": 1,
                    "case_id": f"{participant}-ses-5",
                    "authority_fingerprint": authority,
                    "representation_id": f"source-{lane}",
                    "representation_sha256": representation_sha,
                    "calibration_per_class": budget,
                    "benchmark": {
                        # This field intentionally points at a test-set winner. Confirmatory
                        # v2 must ignore it and use policy.primary_classical_control.
                        "strongest_classical_control": "pooled_mean_std",
                        "density_information_novel": False,
                        "metrics": {
                            "density": {"balanced_accuracy": 0.80},
                            "normalized_covariance": {"balanced_accuracy": 0.80},
                            "pooled_mean_std": {"balanced_accuracy": 0.95},
                            "offdiagonal_ablation": {"balanced_accuracy": 0.60},
                        },
                    },
                }
            )
        cases.append(
            {
                "schema_version": 2,
                "representation_id": f"source-{lane}",
                "representation_sha256": representation_sha,
                "study_fingerprint": participant[-1] * 64,
                "authority": {
                    "case_id": f"{participant}-ses-5",
                    "authority_fingerprint": authority,
                    "held_out_values": ["5"],
                    "case_metadata": {"subject": participant, "held_out_session": "5"},
                },
                "rows": rows,
            }
        )
    case_payload = {
        "schema_version": 1,
        "artifact_role": "e001_representation_lane_cases",
        "scientific_fingerprint": scientific,
        "cases": cases,
    }
    _write(directory / "run.json", run)
    _write(directory / "study_manifest.json", manifest)
    _write(directory / "case_results.json", case_payload)
    (directory / "report.md").write_text("fixture\n", encoding="utf-8")
    ledger = {
        name: _hash(directory / name)
        for name in ("run.json", "study_manifest.json", "case_results.json", "report.md")
    }
    _write(directory / "artifact_hashes.json", ledger)
    return directory


def _policy_mapping() -> dict[str, object]:
    return {
        "policy_id": "fixed-control-v2",
        "reference_representation_id": "raw",
        "primary_calibration_per_class": 10,
        "primary_classical_control": "normalized_covariance",
        "min_participants": 3,
        "min_representations": 2,
        "min_representation_families": 2,
        "min_candidate_advantage": 0.0,
        "min_ablation_necessity": 0.1,
        "min_reference_positive_fraction": 1.0,
        "min_all_lane_positive_fraction": 1.0,
        "min_direction_match_fraction": 1.0,
        "min_ablation_direction_match_fraction": 1.0,
        "min_information_novel_representation_fraction": 1.0,
        "sample_size_rationale": "Artifact contract fixture, not a biological power claim.",
        "inference_seed": 31,
        "bootstrap_resamples": 200,
        "preregistration": None,
    }


def test_loader_uses_predeclared_control_not_final_evaluation_winner(tmp_path: Path) -> None:
    raw = _lane(tmp_path, "raw", "raw_neural")
    learned = _lane(tmp_path, "labram", "foundation_model", model="LaBraM")
    manifest = {
        "schema_version": 2,
        "study_id": "fixed-control-study",
        "mechanism_id": "density_second_moment",
        "participant_key": "subject",
        "policy": _policy_mapping(),
        "lanes": [
            {"lane_id": "raw", "artifact_dir": raw.name},
            {"lane_id": "labram", "artifact_dir": learned.name},
        ],
    }
    path = tmp_path / "manifest.json"
    _write(path, manifest)

    _, _, policy, observations, _ = load_confirmatory_representation_manifest(path)
    assert policy.primary_classical_control == "normalized_covariance"
    assert observations
    assert all(item.candidate_advantage == pytest.approx(0.0) for item in observations)
    # If the final-test winner had been selected post hoc, these would be -0.15.
    assert all(item.candidate_advantage != pytest.approx(-0.15) for item in observations)


def test_loader_rejects_missing_predeclared_control(tmp_path: Path) -> None:
    raw = _lane(tmp_path, "raw", "raw_neural")
    learned = _lane(tmp_path, "labram", "foundation_model", model="LaBraM")
    policy = _policy_mapping()
    policy["primary_classical_control"] = "not_present"
    path = tmp_path / "manifest.json"
    _write(
        path,
        {
            "schema_version": 2,
            "study_id": "missing-control",
            "mechanism_id": "fixture",
            "participant_key": "subject",
            "policy": policy,
            "lanes": [
                {"lane_id": "raw", "artifact_dir": raw.name},
                {"lane_id": "labram", "artifact_dir": learned.name},
            ],
        },
    )
    with pytest.raises(ValueError, match="missing preregistered method"):
        load_confirmatory_representation_manifest(path)
