from __future__ import annotations

import numpy as np
import pytest

from quantumbci.benchmarking import IndexSplit, benchmark_density_embeddings


def _correlation_dataset(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    values = []
    labels = []
    t = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    for index in range(80):
        label = index % 2
        phase = 0.13 * index
        a = np.sin(t + phase)
        sign = 1.0 if label else -1.0
        x = np.stack([a, sign * a, np.cos(2 * t), np.sin(3 * t)], axis=1)
        x += rng.normal(0.0, 0.02, size=x.shape)
        values.append(x)
        labels.append(label)
    return np.stack(values), np.asarray(labels)


def test_density_benchmark_recovers_correlation_and_ablation() -> None:
    embeddings, labels = _correlation_dataset()
    split = IndexSplit(np.arange(60), np.arange(60, 80), name="fixed")
    result = benchmark_density_embeddings(embeddings, labels, split)
    assert result.density.balanced_accuracy >= 0.95
    assert result.density_minus_ablation >= 0.25
    assert result.density.balanced_accuracy > result.diagonal_control.balanced_accuracy
    assert result.to_mapping()["claim_class"] == "quantum_inspired"


def test_split_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        IndexSplit(np.array([0, 1]), np.array([1, 2]))


def test_partition_adapter_is_duck_typed() -> None:
    class Partition:
        train_indices = np.array([0, 1])
        test_indices = np.array([2, 3])
        split_unit = "session"

    split = IndexSplit.from_partition(Partition())
    assert split.name == "session"
    assert split.train_indices.tolist() == [0, 1]
