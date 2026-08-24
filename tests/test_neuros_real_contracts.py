from __future__ import annotations

import numpy as np
import pytest

from quantumbci.integrations.neuros import bind_neuros_evidence


fm = pytest.importorskip("neuros.foundation_models")


def test_real_neuros_longitudinal_authority_binds_into_run_identity() -> None:
    GroupedEvaluationData = fm.GroupedEvaluationData
    chronological_partition = fm.chronological_partition
    make_nested_calibration_split = fm.make_nested_calibration_split

    rng = np.random.default_rng(19)
    data = GroupedEvaluationData(
        dataset_id="quantumbci-neuros-contract-smoke",
        X=rng.normal(size=(12, 4, 16)),
        y=np.asarray([0, 1, 0, 1] * 3),
        groups={
            "subject": np.asarray(["p1"] * 12),
            "session": np.asarray(["s1"] * 4 + ["s2"] * 4 + ["s3"] * 4),
        },
    )
    partition = chronological_partition(
        data,
        split_unit="session",
        held_out_value="s3",
    )
    calibration = make_nested_calibration_split(
        partition,
        evaluation_fraction=0.5,
        seed=23,
    )

    binding = bind_neuros_evidence(
        {"plan_id": "quantumbci-plan-smoke"},
        dataset_fingerprint="upstream-raw-sha256-smoke",
        partition=partition,
        calibration_split=calibration,
        neuros_source_sha="2e1a3c2d57fbba6b8b318d26639abb52bff930a5",
    )

    assert binding.partition_fingerprint == partition.fingerprint
    assert binding.split_fingerprint == calibration.fingerprint
    assert binding.package_versions["neuros-core"] is not None
    assert binding.package_versions["neuros-foundation"] is not None
    assert len(binding.scientific_run_id) == 64

    # NeurOS chronology must keep the future target session out of source training.
    sessions = np.asarray(data.groups["session"])
    assert set(sessions[partition.train_indices]) == {"s1", "s2"}
    assert set(sessions[partition.test_indices]) == {"s3"}
    assert not np.intersect1d(
        calibration.evaluation_indices,
        calibration.calibration_indices(1),
    ).size
