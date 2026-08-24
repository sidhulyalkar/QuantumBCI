"""Prepared E001 feature surfaces for nested longitudinal calibration frontiers.

The promotion-oriented E001 benchmark contains two kinds of computation:

1. representation transforms that do not depend on a calibration budget;
2. readout fitting that should change as calibration examples are added.

Real longitudinal EEG studies can contain very wide raw-token epochs. Recomputing density,
covariance, log-covariance and especially flattened PCA independently for every calibration
budget is wasteful and can also blur what is allowed to adapt. This module makes the boundary
explicit: static transforms are prepared once, PCA is fit on source-history examples only,
and every target-calibration budget reuses those exact prepared features while refitting only
the low-capacity readouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .benchmarking import (
    E001RepresentationBenchmarkResult,
    IndexSplit,
    _covariance_matrix,
    _fit_readout,
    _log_covariance_features,
    _metrics,
    _symmetric_real_vector,
    _train_only_pca_features,
    _validate_class_support,
    remove_density_offdiagonals,
    vectorize_density,
)
from .equivalence import audit_embedding_batch, trace_normalized_second_moment
from .states import density_from_samples


@dataclass(frozen=True)
class E001StaticFeatures:
    """Budget-independent E001 features prepared from one frozen token tensor."""

    values: np.ndarray
    labels: np.ndarray
    feature_sets: Mapping[str, np.ndarray]
    ablated_features: np.ndarray
    equivalence_audit: Mapping[str, Any]
    center_tokens: bool
    covariance_regularization: float

    @property
    def n_samples(self) -> int:
        return int(len(self.labels))


@dataclass(frozen=True)
class E001PreparedCaseFeatures:
    """One source-history-frozen E001 feature surface reused across calibration budgets."""

    static: E001StaticFeatures
    pca_features: np.ndarray
    pca_fit_indices: np.ndarray
    pca_dimension: int

    @property
    def labels(self) -> np.ndarray:
        return self.static.labels

    @property
    def n_samples(self) -> int:
        return self.static.n_samples


def _validate_static_inputs(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if np.iscomplexobj(embeddings):
        raise ValueError(
            "prepared E001 currently expects real-valued embeddings so the matched "
            "classical control suite remains complete"
        )
    values = np.asarray(embeddings, dtype=float)
    target = np.asarray(labels).reshape(-1)
    if values.ndim != 3:
        raise ValueError("embeddings must have shape (examples, tokens, features)")
    if len(values) != len(target):
        raise ValueError("labels must align with embedding examples")
    if values.shape[1] < 2 or values.shape[2] < 2:
        raise ValueError("E001 requires at least two tokens and two features")
    if len(values) < 3:
        raise ValueError("E001 requires at least three examples")
    if not np.all(np.isfinite(values)):
        raise ValueError("embeddings contain non-finite values")
    return values, target


def prepare_e001_static_features(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    center_tokens: bool = True,
    covariance_regularization: float = 1e-6,
) -> E001StaticFeatures:
    """Prepare all E001 transforms that are independent of train/calibration budget.

    The returned arrays remain aligned one-to-one with the input example axis. No split is
    invented and no readout is fitted. This makes the object safe to reuse across multiple
    frozen neurOS authorities for the same participant representation tensor.
    """

    if not np.isfinite(covariance_regularization) or covariance_regularization <= 0:
        raise ValueError("covariance regularization must be finite and positive")
    values, target = _validate_static_inputs(embeddings, labels)

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
    ablated_features = np.stack(
        [vectorize_density(remove_density_offdiagonals(rho)) for rho in states]
    )
    audit = audit_embedding_batch(values, center_tokens=center_tokens).to_mapping()
    if not audit["equivalent_within_tolerance"]:
        raise RuntimeError("density/normalized-covariance equivalence invariant failed")
    if not np.allclose(
        density_features,
        normalized_covariance_features,
        rtol=0.0,
        atol=float(audit["tolerance"]),
    ):
        raise RuntimeError("equivalent density/covariance feature tensors drifted numerically")

    return E001StaticFeatures(
        values=values,
        labels=target,
        feature_sets={
            "density": density_features,
            "normalized_covariance": normalized_covariance_features,
            "covariance": covariance_features,
            "log_covariance": log_covariance_features,
            "bilinear_second_moment": bilinear_features,
            "pooled_mean_std": pooled_features,
            "diagonal_density": diagonal_features,
        },
        ablated_features=ablated_features,
        equivalence_audit=audit,
        center_tokens=bool(center_tokens),
        covariance_regularization=float(covariance_regularization),
    )


def prepare_e001_case_features(
    static: E001StaticFeatures,
    source_train_indices: np.ndarray,
) -> E001PreparedCaseFeatures:
    """Fit the PCA control once on historical source examples and freeze it.

    Target calibration examples are deliberately excluded from PCA fitting. This is stricter
    than refitting PCA at every calibration budget and ensures that movement along a calibration
    frontier reflects readout adaptation rather than a changing representation basis.
    """

    source = np.asarray(source_train_indices, dtype=np.int64).reshape(-1)
    if len(source) < 2:
        raise ValueError("source-history PCA requires at least two source examples")
    if np.any(source < 0) or np.any(source >= static.n_samples):
        raise ValueError("source_train_indices exceed prepared sample authority")
    if len(np.unique(source)) != len(source):
        raise ValueError("source_train_indices contain duplicate samples")

    density_dimension = int(static.feature_sets["density"].shape[1])
    pca_features, pca_dimension = _train_only_pca_features(
        static.values,
        source,
        max_components=density_dimension,
    )
    return E001PreparedCaseFeatures(
        static=static,
        pca_features=pca_features,
        pca_fit_indices=source.copy(),
        pca_dimension=int(pca_dimension),
    )


def benchmark_prepared_e001(
    prepared: E001PreparedCaseFeatures,
    split: IndexSplit,
    *,
    ridge: float = 1e-3,
) -> E001RepresentationBenchmarkResult:
    """Fit matched readouts on one prepared E001 surface under an explicit split."""

    split.validate_length(prepared.n_samples)
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and non-negative")

    train = split.train_indices
    test = split.test_indices
    target = prepared.labels
    y_train = target[train]
    y_test = target[test]
    _validate_class_support(y_train, y_test)

    feature_sets = dict(prepared.static.feature_sets)
    feature_sets["pca_flattened"] = prepared.pca_features
    models = {
        name: _fit_readout(features[train], y_train, ridge=ridge)
        for name, features in feature_sets.items()
    }
    predictions = {
        name: model.predict(feature_sets[name][test])
        for name, model in models.items()
    }
    predictions["offdiagonal_ablation"] = models["density"].predict(
        prepared.static.ablated_features[test]
    )
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
    if not np.array_equal(predictions["density"], predictions["normalized_covariance"]):
        raise RuntimeError("equivalent density/covariance features produced different predictions")

    dimensions = {name: int(features.shape[1]) for name, features in feature_sets.items()}
    dimensions["offdiagonal_ablation"] = int(prepared.static.ablated_features.shape[1])
    dimensions["pca_flattened"] = int(prepared.pca_dimension)
    return E001RepresentationBenchmarkResult(
        classes=tuple(str(value) for value in models["density"].classes.tolist()),
        split_name=split.name,
        metrics=metrics,
        feature_dimensions=dimensions,
        predictions=predictions,
        test_labels=y_test,
        equivalence_audit=dict(prepared.static.equivalence_audit),
        strongest_classical_control=strongest,
    )
