"""Longitudinal E001 execution with source-frozen prepared representation features."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Sequence

import numpy as np

from .benchmarking import IndexSplit
from .e001_prepared import E001StaticFeatures, benchmark_prepared_e001, prepare_e001_case_features
from .longitudinal import (
    LongitudinalE001CaseResult,
    LongitudinalE001Row,
    _authority_value,
    _case_identity,
    _hash_representation,
    _required_text,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def run_prepared_longitudinal_e001_case(
    data: Any,
    authority: Any,
    static: E001StaticFeatures,
    *,
    representation_id: str,
    budgets_per_class: Sequence[int],
    upstream_dataset_fingerprint: str,
    quantumbci_source_sha: str,
    neuros_source_sha: str,
    ridge: float = 1e-3,
) -> LongitudinalE001CaseResult:
    """Run a nested calibration frontier without recomputing representation transforms.

    ``static`` must be aligned to the exact sample order validated by ``authority``. PCA is
    fitted once on ``source_train_indices`` and then held fixed for every calibration budget.
    This means target calibration changes only the readout, not the representation basis.
    """

    representation_name = _required_text("representation_id", representation_id)
    provenance = {
        "upstream_dataset_fingerprint": _required_text(
            "upstream_dataset_fingerprint", upstream_dataset_fingerprint
        ),
        "quantumbci_source_sha": _required_text("quantumbci_source_sha", quantumbci_source_sha),
        "neuros_source_sha": _required_text("neuros_source_sha", neuros_source_sha),
    }
    restore = getattr(authority, "restore", None)
    if not callable(restore):
        raise TypeError("authority must expose callable restore(data)")
    split = restore(data)

    expected_samples = int(_authority_value(authority, "n_samples"))
    if static.n_samples != expected_samples:
        raise ValueError(
            f"prepared rows ({static.n_samples}) do not match authority n_samples ({expected_samples})"
        )
    data_labels = np.asarray(getattr(data, "y", None)).reshape(-1)
    if len(data_labels) != expected_samples:
        raise ValueError("data labels do not align with authority sample count")
    if not np.array_equal(np.asarray(static.labels).astype(str), data_labels.astype(str)):
        raise ValueError("prepared labels do not match authority data labels")

    budgets = tuple(sorted(set(int(value) for value in budgets_per_class)))
    if not budgets or budgets[0] < 0:
        raise ValueError("budgets_per_class must contain non-negative values")
    max_budget = int(getattr(split, "max_budget_per_class"))
    if budgets[-1] > max_budget:
        raise ValueError(
            f"requested budget {budgets[-1]} exceeds authority maximum {max_budget}/class"
        )

    evaluation = np.asarray(getattr(split, "evaluation_indices"), dtype=np.int64)
    source = np.asarray(getattr(split, "source_train_indices"), dtype=np.int64)
    train_for_budget = getattr(split, "train_indices_for_budget", None)
    calibration_for_budget = getattr(split, "calibration_indices", None)
    if not callable(train_for_budget) or not callable(calibration_for_budget):
        raise TypeError("restored split must expose train_indices_for_budget and calibration_indices")

    prepared = prepare_e001_case_features(static, source)
    representation_sha = _hash_representation(static.values)
    identity = _case_identity(authority)
    rows: list[LongitudinalE001Row] = []
    for budget in budgets:
        train = np.asarray(train_for_budget(budget), dtype=np.int64)
        calibration = np.asarray(calibration_for_budget(budget), dtype=np.int64)
        explicit = IndexSplit(
            train_indices=train,
            test_indices=evaluation,
            name=(
                f"{identity['case_id']}|authority={identity['authority_fingerprint']}"
                f"|calibration={budget}/class|pca=source-history-only"
            ),
        )
        benchmark = benchmark_prepared_e001(prepared, explicit, ridge=ridge)
        rows.append(
            LongitudinalE001Row(
                dataset_id=identity["dataset_id"],
                case_id=identity["case_id"],
                authority_fingerprint=identity["authority_fingerprint"],
                partition_fingerprint=identity["partition_fingerprint"],
                calibration_split_fingerprint=identity["calibration_split_fingerprint"],
                processed_data_sha256=identity["processed_data_sha256"],
                held_out_values=tuple(identity["held_out_values"]),
                case_metadata=identity["case_metadata"],
                representation_id=representation_name,
                representation_sha256=representation_sha,
                calibration_per_class=budget,
                source_train_samples=int(len(source)),
                calibration_samples=int(len(calibration)),
                evaluation_samples=int(len(evaluation)),
                result=benchmark,
            )
        )

    study_identity = {
        "schema_version": 3,
        "authority": identity,
        "provenance": provenance,
        "representation_id": representation_name,
        "representation_sha256": representation_sha,
        "budgets_per_class": list(budgets),
        "ridge": float(ridge),
        "center_tokens": bool(static.center_tokens),
        "covariance_regularization": float(static.covariance_regularization),
        "pca_fit_scope": "source_history_only",
        "pca_dimension": int(prepared.pca_dimension),
    }
    fingerprint = sha256(_canonical_json(study_identity).encode("utf-8")).hexdigest()
    return LongitudinalE001CaseResult(
        representation_id=representation_name,
        representation_sha256=representation_sha,
        authority=identity,
        provenance=provenance,
        rows=tuple(rows),
        study_fingerprint=fingerprint,
    )
