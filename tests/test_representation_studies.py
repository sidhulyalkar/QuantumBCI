from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quantumbci.benchmarking import BenchmarkMetrics, E001RepresentationBenchmarkResult
from quantumbci.exporting import verify_run_artifacts
from quantumbci.longitudinal import LongitudinalE001CaseResult, LongitudinalE001Row
from quantumbci.representation_studies import (
    encode_frozen_epochs,
    write_e001_representation_lane_bundle,
)


class _Encoder:
    def encode(self, epoch: np.ndarray, *, sample_rate_hz: float) -> np.ndarray:
        assert sample_rate_hz == 100.0
        x = np.asarray(epoch, dtype=float)
        return np.stack([x.mean(axis=-1), x.std(axis=-1)], axis=0)


class _ChangingEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, epoch: np.ndarray, *, sample_rate_hz: float) -> np.ndarray:
        self.calls += 1
        width = 2 if self.calls == 1 else 3
        return np.ones((2, width), dtype=float)


def _metric(value: float) -> BenchmarkMetrics:
    return BenchmarkMetrics(
        accuracy=value,
        balanced_accuracy=value,
        per_class_recall={"0": value, "1": value},
    )


def _case(participant: str) -> LongitudinalE001CaseResult:
    benchmark = E001RepresentationBenchmarkResult(
        classes=("0", "1"),
        split_name="fixture",
        metrics={
            "density": _metric(0.8),
            "normalized_covariance": _metric(0.6),
            "offdiagonal_ablation": _metric(0.5),
        },
        feature_dimensions={
            "density": 4,
            "normalized_covariance": 4,
            "offdiagonal_ablation": 4,
        },
        predictions={
            "density": np.asarray(["0", "1"]),
            "normalized_covariance": np.asarray(["0", "1"]),
            "offdiagonal_ablation": np.asarray(["1", "0"]),
        },
        test_labels=np.asarray(["0", "1"]),
        equivalence_audit={"novel_information": True},
        strongest_classical_control="normalized_covariance",
    )
    metadata = {"subject": participant, "held_out_session": "1"}
    authority = {
        "dataset_id": "fixture",
        "case_id": f"fixture/{participant}/1",
        "authority_fingerprint": f"authority-{participant}",
        "partition_fingerprint": f"partition-{participant}",
        "calibration_split_fingerprint": f"split-{participant}",
        "processed_data_sha256": participant[-1] * 64,
        "held_out_values": ["1"],
        "case_metadata": metadata,
    }
    row = LongitudinalE001Row(
        dataset_id="fixture",
        case_id=authority["case_id"],
        authority_fingerprint=authority["authority_fingerprint"],
        partition_fingerprint=authority["partition_fingerprint"],
        calibration_split_fingerprint=authority["calibration_split_fingerprint"],
        processed_data_sha256=authority["processed_data_sha256"],
        held_out_values=("1",),
        case_metadata=metadata,
        representation_id="fixture-foundation-representation",
        representation_sha256="f" * 64,
        calibration_per_class=0,
        source_train_samples=10,
        calibration_samples=0,
        evaluation_samples=2,
        result=benchmark,
    )
    return LongitudinalE001CaseResult(
        representation_id="fixture-foundation-representation",
        representation_sha256="f" * 64,
        authority=authority,
        provenance={
            "upstream_dataset_fingerprint": "d" * 64,
            "quantumbci_source_sha": "qbc-source",
            "neuros_source_sha": "neuros-source",
        },
        rows=(row,),
        study_fingerprint=(participant[-1] * 64),
    )


def test_encode_frozen_epochs_preserves_sample_alignment() -> None:
    epochs = np.arange(4 * 3 * 8, dtype=float).reshape(4, 3, 8)
    encoded = encode_frozen_epochs(epochs, _Encoder(), sample_rate_hz=100.0)
    assert encoded.shape == (4, 2, 3)
    assert np.all(np.isfinite(encoded))


def test_encode_frozen_epochs_rejects_shape_drift() -> None:
    epochs = np.ones((2, 3, 8), dtype=float)
    with pytest.raises(ValueError, match="shape changed"):
        encode_frozen_epochs(epochs, _ChangingEncoder(), sample_rate_hz=100.0)


def test_portable_lane_bundle_is_closed_world_and_verified(tmp_path: Path) -> None:
    result = write_e001_representation_lane_bundle(
        [_case("p1"), _case("p2"), _case("p3")],
        tmp_path / "lane",
        study_id="foundation-lane",
        representation_family="foundation_model",
        model_id="FixtureFM",
        model_revision="sha-123",
    )
    assert result["artifact_verification"]["valid"] is True
    verification = verify_run_artifacts(tmp_path / "lane")
    assert verification["valid"] is True
    assert set(verification["verified"]) == {
        "case_results.json",
        "report.md",
        "run.json",
        "study_manifest.json",
    }


def test_lane_bundle_requires_model_revision_when_model_is_declared(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model_revision"):
        write_e001_representation_lane_bundle(
            [_case("p1"), _case("p2")],
            tmp_path / "lane",
            study_id="bad",
            representation_family="foundation_model",
            model_id="FixtureFM",
        )
