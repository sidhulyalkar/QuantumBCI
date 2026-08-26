from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from quantumbci.benchmarking import BenchmarkMetrics, E001RepresentationBenchmarkResult
from quantumbci.bmrb_representation import (
    build_bmrb_representation_bundle,
    verify_bmrb_representation_mapping,
    write_bmrb_representation_bundle,
)
from quantumbci.longitudinal import LongitudinalE001CaseResult, LongitudinalE001Row
from quantumbci.recapitulation import EvidenceTier
from quantumbci.representation_studies import write_e001_representation_lane_bundle


def _metrics(value: float) -> BenchmarkMetrics:
    return BenchmarkMetrics(
        accuracy=value,
        balanced_accuracy=value,
        per_class_recall={"0": value, "1": value},
    )


def _case(
    lane: str,
    participant: str,
    *,
    advantage: float,
    ablation: float,
    novel: bool,
) -> LongitudinalE001CaseResult:
    candidate = 0.80
    control = candidate - advantage
    ablated = candidate - ablation
    authority_fingerprint = f"authority-{participant}"
    representation_sha = (lane[0] * 64)
    benchmark = E001RepresentationBenchmarkResult(
        classes=("0", "1"),
        split_name=f"split-{participant}",
        metrics={
            "density": _metrics(candidate),
            "normalized_covariance": _metrics(control),
            "offdiagonal_ablation": _metrics(ablated),
            "pooled_mean_std": _metrics(0.55),
        },
        feature_dimensions={
            "density": 4,
            "normalized_covariance": 4,
            "offdiagonal_ablation": 4,
            "pooled_mean_std": 4,
        },
        predictions={
            "density": np.asarray(["0", "1"]),
            "normalized_covariance": np.asarray(["0", "1"]),
            "offdiagonal_ablation": np.asarray(["1", "0"]),
            "pooled_mean_std": np.asarray(["1", "0"]),
        },
        test_labels=np.asarray(["0", "1"]),
        equivalence_audit={"novel_information": novel},
        strongest_classical_control="normalized_covariance",
    )
    metadata = {"subject": participant, "held_out_session": "1"}
    authority = {
        "dataset_id": "fixture-dataset",
        "case_id": f"fixture/{participant}/session-1",
        "authority_fingerprint": authority_fingerprint,
        "partition_fingerprint": f"partition-{participant}",
        "calibration_split_fingerprint": f"splitfp-{participant}",
        "processed_data_sha256": participant[-1] * 64,
        "held_out_values": ["1"],
        "case_metadata": metadata,
    }
    row = LongitudinalE001Row(
        dataset_id="fixture-dataset",
        case_id=authority["case_id"],
        authority_fingerprint=authority_fingerprint,
        partition_fingerprint=authority["partition_fingerprint"],
        calibration_split_fingerprint=authority["calibration_split_fingerprint"],
        processed_data_sha256=authority["processed_data_sha256"],
        held_out_values=("1",),
        case_metadata=metadata,
        representation_id=f"fixture-{lane}",
        representation_sha256=representation_sha,
        calibration_per_class=0,
        source_train_samples=10,
        calibration_samples=0,
        evaluation_samples=2,
        result=benchmark,
    )
    return LongitudinalE001CaseResult(
        representation_id=f"fixture-{lane}",
        representation_sha256=representation_sha,
        authority=authority,
        provenance={
            "upstream_dataset_fingerprint": "d" * 64,
            "quantumbci_source_sha": "qbc-source",
            "neuros_source_sha": "neuros-source",
        },
        rows=(row,),
        study_fingerprint=(lane[0] + participant[-1]) * 32,
    )


def _write_lane(
    root: Path,
    lane: str,
    family: str,
    *,
    model_id: str | None,
    novel: bool,
) -> Path:
    cases = [
        _case(lane, participant, advantage=0.18 + 0.02 * index, ablation=0.30, novel=novel)
        for index, participant in enumerate(("p1", "p2", "p3"))
    ]
    output = root / lane
    write_e001_representation_lane_bundle(
        cases,
        output,
        study_id=f"fixture-{lane}",
        representation_family=family,
        model_id=model_id,
        model_revision=None if model_id is None else "rev-1",
    )
    return output


def _manifest(root: Path, *, novel: bool = True) -> Path:
    raw = _write_lane(root, "raw", "raw_neural", model_id=None, novel=novel)
    labram = _write_lane(root, "labram", "foundation_model", model_id="LaBraM", novel=novel)
    eegpt = _write_lane(root, "eegpt", "foundation_model", model_id="EEGPT", novel=novel)
    path = root / "bmrb-representation-manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "study_id": "fixture-cross-representation",
                "mechanism_id": "cross_feature_second_moment",
                "participant_key": "subject",
                "policy": {
                    "policy_id": "fixture-policy",
                    "preregistered": True,
                    "reference_representation_id": "raw",
                    "min_participants": 3,
                    "min_representations": 3,
                    "min_representation_families": 2,
                    "min_reference_positive_fraction": 0.8,
                    "min_all_lane_positive_fraction": 0.8,
                    "min_all_lane_ablation_positive_fraction": 0.8,
                    "min_direction_match_fraction": 0.8,
                    "min_ablation_direction_match_fraction": 0.8,
                    "min_information_novel_representation_fraction": 1.0,
                },
                "lanes": [
                    {"lane_id": "raw", "artifact_dir": raw.name},
                    {"lane_id": "labram", "artifact_dir": labram.name},
                    {"lane_id": "eegpt", "artifact_dir": eegpt.name},
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_build_and_write_representation_bundle(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    bundle = build_bmrb_representation_bundle(manifest)
    assert bundle.conservation.conservation_criteria_passed is True
    assert bundle.conservation.adversary_survival_passed is True
    assert bundle.conservation.promotion_eligible is True
    assert bundle.profile.promotion_ceiling == EvidenceTier.REPEATED_CASE
    assert bundle.artifact_fingerprint

    json_path, html_path = write_bmrb_representation_bundle(bundle, tmp_path / "output")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert verify_bmrb_representation_mapping(payload)["artifact_fingerprint"] == bundle.artifact_fingerprint
    assert "BMRB-Representation" in html_path.read_text(encoding="utf-8")


def test_equivalent_density_lanes_are_conserved_but_not_promoted(tmp_path: Path) -> None:
    bundle = build_bmrb_representation_bundle(_manifest(tmp_path, novel=False))
    assert bundle.conservation.conservation_criteria_passed is True
    assert bundle.conservation.adversary_survival_passed is False
    assert bundle.profile.first_failing_gate == "matched_representation_adversaries"
    assert bundle.profile.promotion_ceiling == EvidenceTier.PREDICTIVE


def test_tampered_lane_fails_before_cross_representation_analysis(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    case_path = tmp_path / "labram" / "case_results.json"
    payload = json.loads(case_path.read_text(encoding="utf-8"))
    payload["cases"][0]["rows"][0]["benchmark"]["metrics"]["density"]["balanced_accuracy"] = 0.999
    case_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="failed artifact verification"):
        build_bmrb_representation_bundle(manifest)


def test_output_artifact_fingerprint_rejects_report_edit(tmp_path: Path) -> None:
    bundle = build_bmrb_representation_bundle(_manifest(tmp_path))
    payload = bundle.to_mapping()
    payload["representation_conservation"]["direction_match_fraction"] = 0.123
    with pytest.raises(ValueError, match="artifact fingerprint mismatch"):
        verify_bmrb_representation_mapping(payload)


def test_foundation_lane_requires_model_revision_identity(tmp_path: Path) -> None:
    cases = [_case("badfm", p, advantage=0.2, ablation=0.3, novel=True) for p in ("p1", "p2", "p3")]
    lane = tmp_path / "badfm"
    write_e001_representation_lane_bundle(
        cases,
        lane,
        study_id="bad-foundation-lane",
        representation_family="foundation_model",
    )
    raw = _write_lane(tmp_path, "raw", "raw_neural", model_id=None, novel=True)
    manifest = tmp_path / "bad-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "study_id": "bad",
                "mechanism_id": "cross_feature_second_moment",
                "policy": {
                    "policy_id": "bad",
                    "preregistered": True,
                    "reference_representation_id": "raw",
                },
                "lanes": [
                    {"lane_id": "raw", "artifact_dir": raw.name},
                    {"lane_id": "badfm", "artifact_dir": lane.name},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must pin model_id and model_revision"):
        build_bmrb_representation_bundle(manifest)
