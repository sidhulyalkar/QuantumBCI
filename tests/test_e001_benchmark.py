from __future__ import annotations

import numpy as np
import pytest

from quantumbci.benchmarking import IndexSplit, benchmark_e001_embeddings


def _correlation_fixture(seed: int = 14) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    windows = []
    labels = []
    t = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    for index in range(96):
        label = index % 2
        a = np.sin(t + 0.08 * index)
        sign = 1.0 if label else -1.0
        window = np.stack(
            [a, sign * a, np.cos(2 * t), np.sin(3 * t + 0.02 * index)],
            axis=1,
        )
        window += rng.normal(0.0, 0.025, size=window.shape)
        windows.append(window)
        labels.append(label)
    return np.stack(windows), np.asarray(labels)


def test_e001_includes_exact_covariance_equivalence_control() -> None:
    embeddings, labels = _correlation_fixture()
    split = IndexSplit(np.arange(72), np.arange(72, 96), name="fixed")
    result = benchmark_e001_embeddings(embeddings, labels, split)

    assert result.equivalence_audit["equivalent_within_tolerance"] is True
    assert result.density_information_novel is False
    assert np.array_equal(
        result.predictions["density"],
        result.predictions["normalized_covariance"],
    )
    assert (
        result.metrics["density"].balanced_accuracy
        == result.metrics["normalized_covariance"].balanced_accuracy
    )
    assert result.density_minus_strongest_control <= 1e-12
    assert result.metrics["density"].balanced_accuracy >= 0.95
    assert result.density_minus_ablation >= 0.25


def test_e001_control_suite_is_train_split_safe() -> None:
    embeddings, labels = _correlation_fixture(seed=19)
    split = IndexSplit(np.arange(64), np.arange(64, 96), name="heldout")
    result = benchmark_e001_embeddings(embeddings, labels, split)
    expected = {
        "density",
        "normalized_covariance",
        "covariance",
        "log_covariance",
        "bilinear_second_moment",
        "pooled_mean_std",
        "pca_flattened",
        "diagonal_density",
        "offdiagonal_ablation",
    }
    assert set(result.metrics) == expected
    assert set(result.predictions) == expected
    assert result.feature_dimensions["pca_flattened"] <= len(split.train_indices) - 1


def test_index_split_rejects_duplicate_sample_weighting() -> None:
    with pytest.raises(ValueError, match="train_indices contain duplicate"):
        IndexSplit(np.asarray([0, 1, 1, 2]), np.asarray([3, 4]))
    with pytest.raises(ValueError, match="test_indices contain duplicate"):
        IndexSplit(np.asarray([0, 1, 2]), np.asarray([3, 4, 4]))


def test_e001_rejects_evaluation_class_absent_from_training() -> None:
    embeddings, labels = _correlation_fixture(seed=23)
    labels = labels.copy()
    labels[72:] = 2
    split = IndexSplit(np.arange(72), np.arange(72, 96))
    with pytest.raises(ValueError, match="classes absent from training authority"):
        benchmark_e001_embeddings(embeddings, labels, split)


def test_e001_rejects_invalid_regularization() -> None:
    embeddings, labels = _correlation_fixture(seed=29)
    split = IndexSplit(np.arange(72), np.arange(72, 96))
    with pytest.raises(ValueError, match="ridge must be finite and non-negative"):
        benchmark_e001_embeddings(embeddings, labels, split, ridge=-1.0)
    with pytest.raises(ValueError, match="covariance regularization"):
        benchmark_e001_embeddings(
            embeddings,
            labels,
            split,
            covariance_regularization=0.0,
        )
