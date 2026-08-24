"""Dependency-light representation benchmarks for frozen neural embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .equivalence import audit_embedding_batch, trace_normalized_second_moment
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
        if len(np.unique(train)) != len(train):
            raise ValueError("train_indices contain duplicate samples")
        if len(np.unique(test)) != len(test):
            raise ValueError("test_indices contain duplicate samples")
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
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and non-negative")
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


def _validate_class_support(y_train: np.ndarray, y_test: np.ndarray) -> None:
    train_classes = set(np.asarray(y_train).astype(str).tolist())
    test_classes = set(np.asarray(y_test).astype(str).tolist())
    missing = sorted(test_classes - train_classes)
    if missing:
        raise ValueError(
            "evaluation contains classes absent from training authority: " + ", ".join(missing)
        )


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


def _validate_embedding_benchmark(
    embeddings: np.ndarray,
    labels: np.ndarray,
    split: IndexSplit,
) -> tuple[np.ndarray, np.ndarray]:
    if np.iscomplexobj(embeddings):
        raise ValueError(
            "the current matched neural benchmark expects real-valued embeddings; "
            "complex representations require explicit complex-valued classical controls"
        )
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
    return values, target


def benchmark_density_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    split: IndexSplit,
    *,
    ridge: float = 1e-3,
    center_tokens: bool = True,
) -> DensityBenchmarkResult:
    """Compare density geometry to matched lightweight controls.

    This compatibility API is retained for v0.4/v0.5 recipes. For promotion-oriented
    E001 work use :func:`benchmark_e001_embeddings`, which includes the exact
    normalized-covariance equivalence control plus a stronger classical control suite.
    """

    values, target = _validate_embedding_benchmark(embeddings, labels, split)
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
    _validate_class_support(y_train, y_test)

    density_model = _fit_readout(density_features[train], y_train, ridge=ridge)
    diagonal_model = _fit_readout(diagonal_features[train], y_train, ridge=ridge)
    pooled_model = _fit_readout(pooled_features[train], y_train, ridge=ridge)

    pred_density = density_model.predict(density_features[test])
    pred_diagonal = diagonal_model.predict(diagonal_features[test])
    pred_pooled = pooled_model.predict(pooled_features[test])
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


def _symmetric_real_vector(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("matrix must be square")
    upper = np.triu_indices(value.shape[0], k=1)
    return np.concatenate([np.diag(value), np.sqrt(2.0) * value[upper]])


def _covariance_matrix(example: np.ndarray, *, center: bool) -> np.ndarray:
    x = np.asarray(example, dtype=float)
    if center:
        x = x - x.mean(axis=0, keepdims=True)
    denom = max(1, x.shape[0] - 1 if center else x.shape[0])
    return (x.T @ x) / float(denom)


def _log_covariance_features(
    matrices: np.ndarray,
    *,
    regularization: float,
) -> np.ndarray:
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("covariance regularization must be finite and positive")
    rows = []
    for matrix in matrices:
        cov = np.asarray(matrix, dtype=float)
        dimension = cov.shape[0]
        scale = float(np.trace(cov)) / max(1, dimension)
        ridge = regularization * max(scale, 1e-12)
        values, vectors = np.linalg.eigh((cov + cov.T) / 2 + ridge * np.eye(dimension))
        values = np.clip(values, 1e-15, None)
        logm = (vectors * np.log(values)) @ vectors.T
        rows.append(_symmetric_real_vector(logm))
    return np.stack(rows)


def _train_only_pca_features(
    values: np.ndarray,
    train_indices: np.ndarray,
    *,
    max_components: int,
) -> tuple[np.ndarray, int]:
    flat = np.asarray(values, dtype=float).reshape(len(values), -1)
    train = flat[train_indices]
    mean = train.mean(axis=0)
    centered = train - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    rank_limit = min(len(train) - 1, flat.shape[1], int(max_components), len(vh))
    if rank_limit < 1:
        raise ValueError("PCA control requires at least two training examples")
    components = vh[:rank_limit]
    return (flat - mean) @ components.T, int(rank_limit)


@dataclass(frozen=True)
class E001RepresentationBenchmarkResult:
    """Adversarial representation benchmark for E001 promotion decisions."""

    classes: tuple[str, ...]
    split_name: str
    metrics: Mapping[str, BenchmarkMetrics]
    feature_dimensions: Mapping[str, int]
    predictions: Mapping[str, np.ndarray]
    test_labels: np.ndarray
    equivalence_audit: Mapping[str, Any]
    strongest_classical_control: str

    @property
    def density_minus_strongest_control(self) -> float:
        return (
            self.metrics["density"].balanced_accuracy
            - self.metrics[self.strongest_classical_control].balanced_accuracy
        )

    @property
    def density_minus_ablation(self) -> float:
        return (
            self.metrics["density"].balanced_accuracy
            - self.metrics["offdiagonal_ablation"].balanced_accuracy
        )

    @property
    def density_information_novel(self) -> bool:
        return bool(self.equivalence_audit.get("novel_information", False))

    def to_mapping(self, *, include_predictions: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "claim_class": "quantum_inspired",
            "split_name": self.split_name,
            "classes": list(self.classes),
            "metrics": {name: metric.to_mapping() for name, metric in self.metrics.items()},
            "feature_dimensions": dict(self.feature_dimensions),
            "equivalence_audit": dict(self.equivalence_audit),
            "strongest_classical_control": self.strongest_classical_control,
            "density_minus_strongest_control": self.density_minus_strongest_control,
            "density_minus_ablation": self.density_minus_ablation,
            "density_information_novel": self.density_information_novel,
            "promotion_interpretation": (
                "The current density constructor is information-equivalent to the "
                "trace-normalized covariance control. Predictive gains over weaker controls "
                "cannot establish new representation information."
            ),
        }
        if include_predictions:
            value["test_labels"] = np.asarray(self.test_labels).astype(str).tolist()
            value["predictions"] = {
                key: np.asarray(pred).astype(str).tolist()
                for key, pred in self.predictions.items()
            }
        return value


def benchmark_e001_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    split: IndexSplit,
    *,
    ridge: float = 1e-3,
    center_tokens: bool = True,
    covariance_regularization: float = 1e-6,
) -> E001RepresentationBenchmarkResult:
    """Run the promotion-oriented E001 density-vs-classical control gauntlet.

    The exact ``normalized_covariance`` control is intentionally present even though
    it is mathematically equivalent to the current density constructor. This makes
    the information ceiling executable rather than merely documented.
    """

    values, target = _validate_embedding_benchmark(embeddings, labels, split)
    train = split.train_indices
    test = split.test_indices
    y_train = target[train]
    y_test = target[test]
    _validate_class_support(y_train, y_test)

    states = np.stack([density_from_samples(x, center=center_tokens) for x in values])
    density_features = np.stack([vectorize_density(rho) for rho in states])
    normalized_covariance_features = np.stack(
        [
            vectorize_density(trace_normalized_second_moment(x, center=center_tokens))
            for x in values
        ]
    )
    diagonal_features = np.stack([np.diag(rho).real for rho in states])
    pooled_features = np.concatenate([values.mean(axis=1), values.std(axis=1)], axis=1)
    centered_covariances = np.stack(
        [_covariance_matrix(x, center=center_tokens) for x in values]
    )
    covariance_features = np.stack(
        [_symmetric_real_vector(cov) for cov in centered_covariances]
    )
    log_covariance_features = _log_covariance_features(
        centered_covariances,
        regularization=covariance_regularization,
    )
    bilinear_matrices = np.stack([_covariance_matrix(x, center=False) for x in values])
    bilinear_features = np.stack(
        [_symmetric_real_vector(matrix) for matrix in bilinear_matrices]
    )
    pca_features, pca_dimension = _train_only_pca_features(
        values,
        train,
        max_components=density_features.shape[1],
    )
    ablated_features = np.stack(
        [vectorize_density(remove_density_offdiagonals(rho)) for rho in states]
    )

    feature_sets = {
        "density": density_features,
        "normalized_covariance": normalized_covariance_features,
        "covariance": covariance_features,
        "log_covariance": log_covariance_features,
        "bilinear_second_moment": bilinear_features,
        "pooled_mean_std": pooled_features,
        "pca_flattened": pca_features,
        "diagonal_density": diagonal_features,
    }
    models = {
        name: _fit_readout(features[train], y_train, ridge=ridge)
        for name, features in feature_sets.items()
    }
    predictions = {
        name: model.predict(feature_sets[name][test])
        for name, model in models.items()
    }
    predictions["offdiagonal_ablation"] = models["density"].predict(ablated_features[test])
    metrics = {name: _metrics(y_test, pred) for name, pred in predictions.items()}

    classical_controls = (
        "normalized_covariance",
        "covariance",
        "log_covariance",
        "bilinear_second_moment",
        "pooled_mean_std",
        "pca_flattened",
        "diagonal_density",
    )
    strongest = max(
        classical_controls,
        key=lambda name: (metrics[name].balanced_accuracy, name),
    )
    audit = audit_embedding_batch(values, center_tokens=center_tokens).to_mapping()
    if not audit["equivalent_within_tolerance"]:
        raise RuntimeError("density/normalized-covariance equivalence invariant failed")
    if not np.array_equal(predictions["density"], predictions["normalized_covariance"]):
        raise RuntimeError("equivalent density/covariance features produced different predictions")

    dimensions = {name: int(features.shape[1]) for name, features in feature_sets.items()}
    dimensions["offdiagonal_ablation"] = int(ablated_features.shape[1])
    dimensions["pca_flattened"] = pca_dimension
    return E001RepresentationBenchmarkResult(
        classes=tuple(str(value) for value in models["density"].classes.tolist()),
        split_name=split.name,
        metrics=metrics,
        feature_dimensions=dimensions,
        predictions=predictions,
        test_labels=y_test,
        equivalence_audit=audit,
        strongest_classical_control=strongest,
    )
