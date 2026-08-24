from __future__ import annotations

import numpy as np
import pytest

from quantumbci.integrations.neuros import bind_neuros_evidence
from quantumbci.longitudinal import evaluate_density_information_gate, run_longitudinal_e001_case


fm = pytest.importorskip("neuros.foundation_models")


def _longitudinal_fixture():
    GroupedEvaluationData = fm.GroupedEvaluationData
    rng = np.random.default_rng(19)
    X = []
    y = []
    subject = []
    session_values = []
    t = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    for session_index, session in enumerate(("s1", "s2", "s3")):
        for label in (0, 1):
            for trial in range(6):
                phase = 0.08 * trial + 0.03 * session_index
                a = np.sin(t + phase)
                sign = 1.0 if label else -1.0
                signal = np.stack(
                    [a, sign * a, np.cos(2 * t), np.sin(3 * t + phase)],
                    axis=0,
                )
                signal += rng.normal(0.0, 0.025, size=signal.shape)
                X.append(signal.astype(np.float32))
                y.append(label)
                subject.append("p1")
                session_values.append(session)
    return GroupedEvaluationData(
        dataset_id="quantumbci-neuros-contract-smoke",
        X=np.asarray(X),
        y=np.asarray(y),
        groups={
            "subject": np.asarray(subject),
            "session": np.asarray(session_values),
        },
    )


def test_real_neuros_longitudinal_authority_binds_into_run_identity() -> None:
    chronological_partition = fm.chronological_partition
    make_nested_calibration_split = fm.make_nested_calibration_split

    data = _longitudinal_fixture()
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
        neuros_source_sha="ffa28ed552dc75158b673fdcd70729b1c9c69b47",
    )

    assert binding.partition_fingerprint == partition.fingerprint
    assert binding.split_fingerprint == calibration.fingerprint
    assert binding.package_versions["neuros-core"] is not None
    assert binding.package_versions["neuros-foundation"] is not None
    assert len(binding.scientific_run_id) == 64

    sessions = np.asarray(data.groups["session"])
    assert set(sessions[partition.train_indices]) == {"s1", "s2"}
    assert set(sessions[partition.test_indices]) == {"s3"}
    assert not np.intersect1d(
        calibration.evaluation_indices,
        calibration.calibration_indices(1),
    ).size


def test_quantumbci_consumes_real_longitudinal_case_authority() -> None:
    LongitudinalCaseAuthority = fm.LongitudinalCaseAuthority
    partition = fm.chronological_partition(
        _longitudinal_fixture(),
        split_unit="session",
        held_out_value="s3",
    )
    calibration = fm.make_nested_calibration_split(
        partition,
        evaluation_fraction=0.5,
        seed=23,
    )
    authority = LongitudinalCaseAuthority.from_split(
        calibration,
        case_id="p1-s3",
        history_policy="prior",
        case_metadata={"subject": "p1", "original_protocol": "GR"},
    )
    data = partition.data

    # QuantumBCI keeps token-level geometry instead of pretending neurOS's pooled
    # frozen decoder embeddings have a token axis. Here raw time is the token axis.
    representations = np.transpose(np.asarray(data.X), (0, 2, 1))
    result = run_longitudinal_e001_case(
        data,
        authority,
        representations,
        representation_id="raw-time-by-channel-v1",
        budgets_per_class=(0, 1, 2),
    )

    assert result.authority["authority_fingerprint"] == authority.authority_fingerprint
    assert result.authority["processed_data_sha256"] == authority.processed_data_sha256
    assert len(result.representation_sha256) == 64
    assert {row.evaluation_samples for row in result.rows} == {
        len(calibration.evaluation_indices)
    }
    assert [row.calibration_samples for row in result.rows] == [0, 2, 4]
    for row in result.rows:
        assert row.result.equivalence_audit["equivalent_within_tolerance"] is True
        assert np.array_equal(
            row.result.predictions["density"],
            row.result.predictions["normalized_covariance"],
        )

    gate = evaluate_density_information_gate(result.rows)
    assert gate["promotion_eligible"] is False
    assert gate["mathematical_equivalence_detected"] is True
