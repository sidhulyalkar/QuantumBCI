from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from quantumbci.longitudinal import (
    evaluate_density_information_gate,
    paired_participant_bootstrap,
    run_longitudinal_e001_case,
)


@dataclass
class _Data:
    y: np.ndarray


class _Split:
    def __init__(self) -> None:
        self.source_train_indices = np.arange(0, 48, dtype=np.int64)
        self.evaluation_indices = np.arange(64, 80, dtype=np.int64)
        self._pool0 = np.arange(48, 56, dtype=np.int64)
        self._pool1 = np.arange(56, 64, dtype=np.int64)
        self.max_budget_per_class = 8

    def calibration_indices(self, per_class: int) -> np.ndarray:
        budget = int(per_class)
        if budget < 0 or budget > self.max_budget_per_class:
            raise ValueError("bad budget")
        if budget == 0:
            return np.asarray([], dtype=np.int64)
        return np.sort(np.concatenate([self._pool0[:budget], self._pool1[:budget]]))

    def train_indices_for_budget(self, per_class: int) -> np.ndarray:
        calibration = self.calibration_indices(per_class)
        if len(calibration) == 0:
            return self.source_train_indices.copy()
        return np.sort(np.concatenate([self.source_train_indices, calibration]))


class _Authority:
    def __init__(self, subject: str, case: str) -> None:
        self.dataset_id = "fixture"
        self.case_id = case
        self.authority_fingerprint = f"authority-{case}"
        self.partition_fingerprint = f"partition-{case}"
        self.calibration_split_fingerprint = f"calibration-{case}"
        self.processed_data_sha256 = "a" * 64
        self.held_out_values = ("s3",)
        self.case_metadata = {"subject": subject}
        self.n_samples = 80
        self._split = _Split()

    def restore(self, data: _Data) -> _Split:
        assert len(data.y) == self.n_samples
        return self._split


def _fixture(seed: int) -> tuple[_Data, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.asarray([0, 1] * 40)
    t = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    rows = []
    for index, label in enumerate(labels):
        a = np.sin(t + 0.05 * index)
        sign = 1.0 if label else -1.0
        x = np.stack([a, sign * a, np.cos(2 * t), np.sin(3 * t)], axis=1)
        x += rng.normal(0.0, 0.02, size=x.shape)
        rows.append(x)
    return _Data(y=labels), np.stack(rows)


def _run(
    data: _Data,
    authority: _Authority,
    representations: np.ndarray,
    *,
    budgets: tuple[int, ...] = (0, 2, 4),
    qbc_sha: str = "qbc-test-sha",
):
    return run_longitudinal_e001_case(
        data,
        authority,
        representations,
        representation_id="fixture-tokens-v1",
        budgets_per_class=budgets,
        upstream_dataset_fingerprint="raw-dataset-fixture-sha256",
        quantumbci_source_sha=qbc_sha,
        neuros_source_sha="neuros-test-sha",
    )


def test_longitudinal_case_reuses_fixed_authority_across_budgets() -> None:
    data, representations = _fixture(8)
    authority = _Authority("p1", "p1-s3")
    result = _run(data, authority, representations)
    assert len(result.rows) == 3
    assert len(result.representation_sha256) == 64
    assert len(result.study_fingerprint) == 64
    assert result.provenance["upstream_dataset_fingerprint"] == "raw-dataset-fixture-sha256"
    assert result.provenance["quantumbci_source_sha"] == "qbc-test-sha"
    assert result.provenance["neuros_source_sha"] == "neuros-test-sha"
    assert {row.evaluation_samples for row in result.rows} == {16}
    assert [row.calibration_samples for row in result.rows] == [0, 4, 8]
    assert all(row.authority_fingerprint == "authority-p1-s3" for row in result.rows)
    assert all(row.result.density_information_novel is False for row in result.rows)
    for row in result.rows:
        assert np.array_equal(
            row.result.predictions["density"],
            row.result.predictions["normalized_covariance"],
        )


def test_study_fingerprint_binds_source_revisions() -> None:
    data, representations = _fixture(9)
    authority = _Authority("p1", "p1-s3")
    first = _run(data, authority, representations, qbc_sha="qbc-a")
    second = _run(data, authority, representations, qbc_sha="qbc-b")
    assert first.representation_sha256 == second.representation_sha256
    assert first.authority == second.authority
    assert first.study_fingerprint != second.study_fingerprint


def test_longitudinal_case_requires_complete_upstream_provenance() -> None:
    data, representations = _fixture(10)
    authority = _Authority("p1", "p1-s3")
    with pytest.raises(ValueError, match="upstream_dataset_fingerprint"):
        run_longitudinal_e001_case(
            data,
            authority,
            representations,
            representation_id="fixture",
            budgets_per_class=(0,),
            upstream_dataset_fingerprint="",
            quantumbci_source_sha="qbc",
            neuros_source_sha="neuros",
        )


def test_participant_bootstrap_never_bootstraps_windows() -> None:
    rows = []
    for subject, seed in (("p1", 3), ("p2", 5), ("p3", 7)):
        data, representations = _fixture(seed)
        result = _run(
            data,
            _Authority(subject, f"{subject}-s3"),
            representations,
            budgets=(0, 2),
        )
        rows.extend(result.rows)

    summaries = paired_participant_bootstrap(
        rows,
        control="normalized_covariance",
        n_resamples=500,
        seed=11,
    )
    assert len(summaries) == 2
    assert {item.n_units for item in summaries} == {3}
    assert all(abs(item.observed_mean_delta) < 1e-12 for item in summaries)
    assert all(abs(item.ci_lower) < 1e-12 for item in summaries)
    assert all(abs(item.ci_upper) < 1e-12 for item in summaries)

    gate = evaluate_density_information_gate(rows)
    assert gate["mathematical_equivalence_detected"] is True
    assert gate["normalized_covariance_prediction_identity"] is True
    assert gate["promotion_eligible"] is False


def test_participant_bootstrap_requires_same_units_at_every_budget() -> None:
    rows = []
    for subject, seed in (("p1", 31), ("p2", 33), ("p3", 35)):
        data, representations = _fixture(seed)
        rows.extend(
            _run(
                data,
                _Authority(subject, f"{subject}-s3"),
                representations,
                budgets=(0, 2),
            ).rows
        )
    incomplete = [
        row
        for row in rows
        if not (row.calibration_per_class == 2 and row.case_metadata["subject"] == "p3")
    ]
    with pytest.raises(ValueError, match="participant membership differs"):
        paired_participant_bootstrap(
            incomplete,
            control="normalized_covariance",
            n_resamples=100,
        )


def test_participant_bootstrap_fails_without_participant_metadata() -> None:
    data, representations = _fixture(4)
    authority = _Authority("p1", "case")
    authority.case_metadata = {}
    row = _run(data, authority, representations, budgets=(0,)).rows[0]
    with pytest.raises(ValueError, match="participant-level inference"):
        paired_participant_bootstrap([row, row], control="covariance", n_resamples=100)
