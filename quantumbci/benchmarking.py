"""Dependency-light representation benchmarks for frozen neural embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .states import density_from_samples


@dataclass(frozen=True)
class IndexSplit:
    """Immutable train/test index authority for one benchmark."""

    train_indices: np.ndarray
    test_indices: np.ndarray
    name: str = "explicit"

    def __post_init__(self) -> None:
        train = np.asarray(self.train_indices, dtype=np.int64).reshape(-1)
        test = np.asarray(self.test_indices, dtype=np.int64).reshape(-1)
        if train.size == 0 or test.size == 0:
            raise ValueError("train_indices and test_indices must both be non-empty")
        if np.any(train < 0) or np.any(test < 0):
            raise ValueError("split indices must be non-negative")
        if np.intersect1d(train, test).size:
            raise ValueError("train/test split overlap detected")
        object.__setattr__(self, "train_indices", train)
        object.__setattr__(self, "test_indices", test)

    @classmethod
    def from_partition(cls, partition: Any, *, name: str | None = None) -> "IndexSplit":
        """Adapt a neurOS-style partition or another object with train/test indices."""

        if not hasattr(partition, "train_indices") or not hasattr(partition, "test_indices"):
            raise TypeError("partition must expose train_indices and test_indices")
        return cls(
            train_indices=np.asarray(partition.train_indices),
            test_indices=np.asarray(partition.test_indices),
            name=name or getattr(partition, "split_unit", "partition"),
        )

    def validate_length(self, n_samples: int) -> None:
        if np.any(self.train_indices >= n_samples) or np.any(self.test_indices >= n_samples):
            raise ValueError("split index exceeds embedding sample count")


@dataclass(frozen=True)
class BenchmarkMetrics:
    accuracy: float
    balanced_accuracy: float
    per_class_recall: Mapping[str, float]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "accuracy": float(self.accuracy),
            "balanced_accuracy": float(self.balanced_accuracy),
            "per_class_recall": dict(self.per_class_recall),
        }


@dataclass(frozen=True)
class DensityBenchmarkResult:
    """Matched readout comparison on one frozen embedding tensor."""

    classes: tuple[str, ...]
    split_name: str
    density: BenchmarkMetrics
    diagonal_control: BenchmarkMetrics
    pooled_control: BenchmarkMetrics
    offdiagonal_ablation: BenchmarkMetrics
    predictions: Mapping[str, np.ndarray]
    test_labels: np.ndarray

    @property
    def density_minus_diagonal(self) -> float:
        return self.density.balanced_accuracy - self.diagonal_control.balanced_accuracy

    @property
    def density_minus_ablation(self) -> float:
        return self.density.balanced_accuracy - self.offdiagonal_ablation.balanced_accuracy

    def to_mapping(self, *, include_predictions: bool = False) -> dict[str, Any]:
        value = {
            "claim_class": "quantum_inspired",
            "split_name": self.split_name,
            "classes": list(self.classes),
            "density": self.density.to_mapping(),
            "diagonal_control": self.diagonal_control.to_mapping(),
            "pooled_control": self.pooled_control.to_mapping(),
            "offdiagonal_ablation": self.offdiagonal_ablation.to_mapping(),
            "density_minus_diagonal": self.density_minus_diagonal,
            "density_minus_ablation": self.density_minus_ablation,
        }
        if include_predictions:
            value["test_labels"] = np.asarray(self.test_labels).astype(str).tolist()
            value["predictions"] = {
                key: np.asarray(pred).astype(str).tolist()
                for key, pred in self.predictions.items()
            }
        return value


@dataclass(frozen=True)
class _LinearReadout:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    classes: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        z = (values - self.mean) / self.scale
        design = np.concatenate([z, np.ones((len(z), 1))], axis=1)
        scores = design @ self.weights
        return self.classes[np.argmax(scores, axis=1)]


def _fit_readout(x: np.ndarray, y: np.ndarray, *, ridge: float) -> _LinearReadout:
    values = np.asarray(x, dtype=float)
    labels = np.asarray(y).reshape(-1)
    if values.ndim != 2 or len(values) != len(labels):
        raise ValueError("readout requires 2D features aligned with labels")
    classes = np.unique(labels)
    if len(classes) < 2:
        raise ValueError("readout requires at least two classes")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale < 1e-10, 1.0, scale)
    z = (values - mean) / scale
    design = np.concatenate([z, np.ones((len(z), 1))], axis=1)
    target = np.stack([(labels == label).astype(float) for label in classes], axis=1)
    penalty = np.eye(design.shape[1]) * float(ridge)
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return _LinearReadout(mean=mean, scale=scale, weights=weights, classes=classes)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> BenchmarkMetrics:
    truth = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    if len(truth) != len(pred):
        raise ValueError("prediction count does not match labels")
    recalls: dict[str, float] = {}
    for label in np.unique(truth):
        mask = truth == label
        recalls[str(label)] = float(np.mean(pred[mask] == label))
    return BenchmarkMetrics(
        accuracy=float(np.mean(truth == pred)),
        balanced_accuracy=float(np.mean(list(recalls.values()))),
        per_class_recall=recalls,
    )


def vectorize_density(rho: np.ndarray) -> np.ndarray:
    """Encode a Hermitian d x d density operator into exactly d^2 real values."""

    state = np.asarray(rho, dtype=complex)
    if state.ndim != 2 or state.shape[0] != state.shape[1]:
        raise ValueError("rho must be square")
    upper = np.triu_indices(state.shape[0], k=1)
    return np.concatenate(
        [
            np.diag(state).real,
            state[upper].real,
            state[upper].imag,
        ]
    ).astype(float, copy=False)


def remove_density_offdiagonals(rho: np.ndarray) -> np.ndarray:
    state = np.asarray(rho, dtype=complex)
    if state.ndim != 2 or state.shape[0] != state.shape[1]:
        raise ValueError("rho must be square")
    return np.diag(np.diag(state))


def benchmark_density_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    split: IndexSplit,
    *,
    ridge: float = 1e-3,
    center_tokens: bool = True,
) -> DensityBenchmarkResult:
    """Compare density geometry to matched lightweight controls.

    ``embeddings`` must have shape ``(examples, tokens, features)``. The function
    never creates a random split on the caller's behalf: promoted studies should
    bind ``split`` to neurOS or another immutable evidence authority.
    """

    values = np.asarray(embeddings, dtype=float)
    target = np.asarray(labels).reshape(-1)
    if values.ndim != 3:
        raise ValueError("embeddings must have shape (examples, tokens, features)")
    if len(values) != len(target):
        raise ValueError("labels must align with embedding examples")
    if values.shape[1] < 2 or values.shape[2] < 2:
        raise ValueError("density benchmark requires at least two tokens and two features")
    if not np.all(np.isfinite(values)):
        raise ValueError("embeddings contain non-finite values")
    split.validate_length(len(values))

    states = np.stack(
        [density_from_samples(example, center=center_tokens) for example in values]
    )
    density_features = np.stack([vectorize_density(rho) for rho in states])
    diagonal_features = np.stack([np.diag(rho).real for rho in states])
    pooled_features = np.concatenate(
        [values.mean(axis=1), values.std(axis=1)],
        axis=1,
    )
    ablated_features = np.stack(
        [vectorize_density(remove_density_offdiagonals(rho)) for rho in states]
    )

    train = split.train_indices
    test = split.test_indices
    y_train = target[train]
    y_test = target[test]

    density_model = _fit_readout(density_features[train], y_train, ridge=ridge)
    diagonal_model = _fit_readout(diagonal_features[train], y_train, ridge=ridge)
    pooled_model = _fit_readout(pooled_features[train], y_train, ridge=ridge)

    pred_density = density_model.predict(density_features[test])
    pred_diagonal = diagonal_model.predict(diagonal_features[test])
    pred_pooled = pooled_model.predict(pooled_features[test])
    # Intervention semantics: erase the proposed density mechanism but keep the
    # original fitted density readout fixed.
    pred_ablated = density_model.predict(ablated_features[test])

    return DensityBenchmarkResult(
        classes=tuple(str(value) for value in density_model.classes.tolist()),
        split_name=split.name,
        density=_metrics(y_test, pred_density),
        diagonal_control=_metrics(y_test, pred_diagonal),
        pooled_control=_metrics(y_test, pred_pooled),
        offdiagonal_ablation=_metrics(y_test, pred_ablated),
        predictions={
            "density": pred_density,
            "diagonal_control": pred_diagonal,
            "pooled_control": pred_pooled,
            "offdiagonal_ablation": pred_ablated,
        },
        test_labels=y_test,
    )
