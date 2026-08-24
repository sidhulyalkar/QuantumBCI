from __future__ import annotations

import numpy as np
import pytest

from quantumbci.benchmarking import IndexSplit, benchmark_e001_embeddings
from quantumbci.e001_prepared import (
    benchmark_prepared_e001,
    prepare_e001_case_features,
    prepare_e001_static_features,
)


def _fixture(seed: int = 41) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    windows = []
    labels = []
    for index in range(96):
        label = index % 2
        a = np.sin(t + 0.04 * index)
        sign = 1.0 if label else -1.0
        x = np.stack(
            [a, sign * a, np.cos(2.0 * t), np.sin(3.0 * t + 0.01 * index)],
            axis=1,
        )
        x += rng.normal(0.0, 0.02, size=x.shape)
        windows.append(x)
        labels.append(label)
    return np.stack(windows), np.asarray(labels)


def test_prepared_e001_matches_direct_benchmark_when_pca_fit_set_equals_train() -> None:
    embeddings, labels = _fixture()
    split = IndexSplit(np.arange(64), np.arange(72, 96), name="source-only")

    direct = benchmark_e001_embeddings(embeddings, labels, split)
    static = prepare_e001_static_features(embeddings, labels)
    prepared = prepare_e001_case_features(static, split.train_indices)
    cached = benchmark_prepared_e001(prepared, split)

    assert prepared.pca_dimension == direct.feature_dimensions["pca_flattened"]
    assert cached.equivalence_audit == direct.equivalence_audit
    assert cached.strongest_classical_control == direct.strongest_classical_control
    assert set(cached.metrics) == set(direct.metrics)
    for name in cached.predictions:
        assert np.array_equal(cached.predictions[name], direct.predictions[name]), name
        assert cached.metrics[name].balanced_accuracy == direct.metrics[name].balanced_accuracy


def test_prepared_pca_is_frozen_when_target_calibration_changes_readout() -> None:
    embeddings, labels = _fixture(seed=43)
    static = prepare_e001_static_features(embeddings, labels)
    source = np.arange(48)
    prepared = prepare_e001_case_features(static, source)

    zero = IndexSplit(source, np.arange(72, 96), name="budget-0")
    calibrated = IndexSplit(np.arange(56), np.arange(72, 96), name="budget-4-per-class")
    zero_result = benchmark_prepared_e001(prepared, zero)
    calibrated_result = benchmark_prepared_e001(prepared, calibrated)

    assert np.array_equal(prepared.pca_fit_indices, source)
    assert prepared.pca_dimension <= len(source) - 1
    assert zero_result.feature_dimensions["pca_flattened"] == prepared.pca_dimension
    assert calibrated_result.feature_dimensions["pca_flattened"] == prepared.pca_dimension
    assert np.array_equal(
        zero_result.predictions["density"],
        zero_result.predictions["normalized_covariance"],
    )
    assert np.array_equal(
        calibrated_result.predictions["density"],
        calibrated_result.predictions["normalized_covariance"],
    )


def test_prepared_e001_rejects_malformed_source_authority() -> None:
    embeddings, labels = _fixture(seed=47)
    static = prepare_e001_static_features(embeddings, labels)

    with pytest.raises(ValueError, match="duplicate"):
        prepare_e001_case_features(static, np.asarray([0, 1, 1, 2]))
    with pytest.raises(ValueError, match="exceed"):
        prepare_e001_case_features(static, np.asarray([0, len(labels)]))
    with pytest.raises(ValueError, match="at least two"):
        prepare_e001_case_features(static, np.asarray([0]))
