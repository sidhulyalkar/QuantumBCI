from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from quantumbci.trajectory_authority import (
    TrajectoryEvidenceAuthority,
    TrajectoryEvidenceData,
    load_trajectory_contract_descriptor,
)


def _fixture() -> TrajectoryEvidenceData:
    # One six-window historical trajectory plus one six-window target trajectory.
    # Target calibration uses t=[0,2); evaluation uses t=[3,6), leaving a one-second purge.
    states = np.asarray(
        [[0.1 * i, np.sin(i), np.cos(i)] for i in range(12)],
        dtype=np.float64,
    )
    trajectory_ids = np.asarray(["source"] * 6 + ["target"] * 6)
    starts = np.asarray(list(range(6)) + list(range(6)), dtype=float)
    stops = starts + 1.0
    return TrajectoryEvidenceData(
        dataset_id="synthetic-continuous",
        states=states,
        trajectory_ids=trajectory_ids,
        start_times_s=starts,
        stop_times_s=stops,
        metadata={"window_seconds": 1.0, "encoder": "frozen-test-v1"},
    )


def _authority(data: TrajectoryEvidenceData) -> TrajectoryEvidenceAuthority:
    return TrajectoryEvidenceAuthority.from_data(
        data,
        case_id="subject-01/recording-02",
        fit_indices=np.arange(0, 6),
        calibration_indices=np.asarray([6, 7]),
        evaluation_indices=np.asarray([9, 10, 11]),
        representation_fit_indices=np.arange(0, 4),
        latent_dimension=3,
        time_step_policy="fixed",
        expected_step_seconds=1.0,
        step_tolerance_seconds=1e-9,
        purge_seconds=1.0,
        upstream_authority_fingerprint="neuros-case-abc123",
        source_revisions={"quantumbci": "qbc-sha", "neuros": "neuros-sha"},
    )


def test_authority_freezes_identity_and_legal_transition_graph() -> None:
    data = _fixture()
    authority = _authority(data)
    assert len(data.data_sha256) == 64
    assert len(authority.authority_fingerprint) == 16
    assert authority.representation_fit_indices == (0, 1, 2, 3)
    assert np.array_equal(
        authority.transition_pairs(data, "fit"),
        np.asarray([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]),
    )
    assert np.array_equal(authority.transition_pairs(data, "calibration"), np.asarray([[6, 7]]))
    assert np.array_equal(
        authority.transition_pairs(data, "evaluation"),
        np.asarray([[9, 10], [10, 11]]),
    )
    payload = authority.to_dict(data=data)
    assert payload["transition_counts"] == {"fit": 5, "calibration": 1, "evaluation": 2}
    restored = TrajectoryEvidenceAuthority.from_dict(payload)
    assert restored.authority_fingerprint == authority.authority_fingerprint
    restored.restore(data)


def test_tensor_mutation_breaks_authority_identity() -> None:
    data = _fixture()
    authority = _authority(data)
    changed = np.asarray(data.states).copy()
    changed[0, 0] += 1e-3
    mutated = TrajectoryEvidenceData(
        dataset_id=data.dataset_id,
        states=changed,
        trajectory_ids=data.trajectory_ids,
        start_times_s=data.start_times_s,
        stop_times_s=data.stop_times_s,
        metadata=data.metadata,
    )
    assert mutated.data_sha256 != data.data_sha256
    with pytest.raises(ValueError, match="SHA-256"):
        authority.restore(mutated)


def test_temporal_purge_blocks_target_boundary_leakage() -> None:
    data = _fixture()
    with pytest.raises(ValueError, match="temporal leakage"):
        TrajectoryEvidenceAuthority.from_data(
            data,
            case_id="leaky",
            fit_indices=np.arange(0, 6),
            calibration_indices=np.asarray([6, 7]),
            evaluation_indices=np.asarray([8, 9, 10, 11]),
            representation_fit_indices=np.arange(0, 4),
            latent_dimension=3,
            expected_step_seconds=1.0,
            purge_seconds=1.0,
        )


def test_representation_fit_cannot_use_calibration_or_evaluation() -> None:
    data = _fixture()
    with pytest.raises(ValueError, match="subset of fit_indices"):
        TrajectoryEvidenceAuthority.from_data(
            data,
            case_id="representation-leak",
            fit_indices=np.arange(0, 6),
            calibration_indices=np.asarray([6, 7]),
            evaluation_indices=np.asarray([9, 10, 11]),
            representation_fit_indices=np.asarray([0, 1, 6]),
            latent_dimension=3,
            expected_step_seconds=1.0,
            purge_seconds=1.0,
        )


def test_invalid_window_in_evidence_role_fails_closed() -> None:
    base = _fixture()
    valid = np.ones(base.n_windows, dtype=bool)
    valid[10] = False
    data = TrajectoryEvidenceData(
        dataset_id=base.dataset_id,
        states=base.states,
        trajectory_ids=base.trajectory_ids,
        start_times_s=base.start_times_s,
        stop_times_s=base.stop_times_s,
        valid_mask=valid,
        metadata=base.metadata,
    )
    with pytest.raises(ValueError, match="invalid/missing"):
        _authority(data)


def test_duplicate_temporal_coordinate_fails_closed() -> None:
    base = _fixture()
    starts = np.asarray(base.start_times_s).copy()
    starts[10] = starts[9]
    stops = starts + 1.0
    data = TrajectoryEvidenceData(
        dataset_id=base.dataset_id,
        states=base.states,
        trajectory_ids=base.trajectory_ids,
        start_times_s=starts,
        stop_times_s=stops,
        metadata=base.metadata,
    )
    with pytest.raises(ValueError, match="duplicate/non-increasing"):
        _authority(data)


def _write_descriptor(root: Path, *, states_name: str = "states.npy") -> Path:
    data = _fixture()
    np.save(root / states_name, data.states)
    np.save(root / "trajectory_ids.npy", data.trajectory_ids)
    np.save(root / "starts.npy", data.start_times_s)
    np.save(root / "stops.npy", data.stop_times_s)
    payload = {
        "schema_version": 1,
        "dataset_id": data.dataset_id,
        "case_id": "portable-case",
        "latent_dimension": 3,
        "time_step_policy": "fixed",
        "expected_step_seconds": 1.0,
        "step_tolerance_seconds": 1e-9,
        "purge_seconds": 1.0,
        "upstream_authority_fingerprint": "neuros-case-abc123",
        "source_revisions": {"quantumbci": "qbc-sha", "neuros": "neuros-sha"},
        "data_metadata": dict(data.metadata),
        "data": {
            "states": states_name,
            "trajectory_ids": "trajectory_ids.npy",
            "start_times_s": "starts.npy",
            "stop_times_s": "stops.npy"
        },
        "split": {
            "fit_indices": list(range(6)),
            "calibration_indices": [6, 7],
            "evaluation_indices": [9, 10, 11],
            "representation_fit_indices": [0, 1, 2, 3]
        }
    }
    path = root / "trajectory_contract.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_descriptor_is_content_addressed_not_filename_addressed(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_path = _write_descriptor(first_dir, states_name="states.npy")
    second_path = _write_descriptor(second_dir, states_name="renamed_states.npy")

    first_data, first = load_trajectory_contract_descriptor(first_path)
    second_data, second = load_trajectory_contract_descriptor(second_path)
    assert first_data.data_sha256 == second_data.data_sha256
    assert first.authority_fingerprint == second.authority_fingerprint


def test_descriptor_rejects_missing_required_array(tmp_path: Path) -> None:
    path = _write_descriptor(tmp_path)
    payload = json.loads(path.read_text())
    del payload["data"]["stop_times_s"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="data.stop_times_s"):
        load_trajectory_contract_descriptor(path)
