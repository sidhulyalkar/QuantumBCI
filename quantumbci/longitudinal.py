"""Longitudinal E001 execution on externally frozen evidence authority.

This module intentionally depends only on NumPy and duck-typed public contracts.
When neurOS is installed, callers pass a real ``GroupedEvaluationData`` and
``LongitudinalCaseAuthority``. The authority's ``restore`` method remains the
source of truth for processed-data identity, chronology, calibration pools, and
fixed final evaluation indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .benchmarking import E001RepresentationBenchmarkResult, IndexSplit, benchmark_e001_embeddings


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _required_text(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _hash_representation(values: np.ndarray) -> str:
    array = np.asarray(values)
    if array.dtype.hasobject:
        raise TypeError("object-dtype representations cannot be fingerprinted")
    digest = sha256()
    digest.update(b"quantumbci.longitudinal-representation.v1\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    for sample in array:
        contiguous = np.ascontiguousarray(sample)
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _authority_value(authority: Any, name: str) -> Any:
    if not hasattr(authority, name):
        raise TypeError(f"authority must expose {name}")
    return getattr(authority, name)


def _authority_metadata(authority: Any) -> dict[str, Any]:
    raw = getattr(authority, "case_metadata", {}) or {}
    return dict(raw)


def _case_identity(authority: Any) -> dict[str, Any]:
    return {
        "dataset_id": str(_authority_value(authority, "dataset_id")),
        "case_id": str(_authority_value(authority, "case_id")),
        "authority_fingerprint": str(_authority_value(authority, "authority_fingerprint")),
        "partition_fingerprint": str(_authority_value(authority, "partition_fingerprint")),
        "calibration_split_fingerprint": str(
            _authority_value(authority, "calibration_split_fingerprint")
        ),
        "processed_data_sha256": str(_authority_value(authority, "processed_data_sha256")),
        "held_out_values": [str(v) for v in _authority_value(authority, "held_out_values")],
        "case_metadata": _authority_metadata(authority),
    }


@dataclass(frozen=True)
class LongitudinalE001Row:
    dataset_id: str
    case_id: str
    authority_fingerprint: str
    partition_fingerprint: str
    calibration_split_fingerprint: str
    processed_data_sha256: str
    held_out_values: tuple[str, ...]
    case_metadata: Mapping[str, Any]
    representation_id: str
    representation_sha256: str
    calibration_per_class: int
    source_train_samples: int
    calibration_samples: int
    evaluation_samples: int
    result: E001RepresentationBenchmarkResult
    status: str = "ok"

    def to_mapping(self, *, include_predictions: bool = False) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": self.status,
            "dataset_id": self.dataset_id,
            "case_id": self.case_id,
            "authority_fingerprint": self.authority_fingerprint,
            "partition_fingerprint": self.partition_fingerprint,
            "calibration_split_fingerprint": self.calibration_split_fingerprint,
            "processed_data_sha256": self.processed_data_sha256,
            "held_out_values": list(self.held_out_values),
            "case_metadata": dict(self.case_metadata),
            "representation_id": self.representation_id,
            "representation_sha256": self.representation_sha256,
            "calibration_per_class": self.calibration_per_class,
            "source_train_samples": self.source_train_samples,
            "calibration_samples": self.calibration_samples,
            "evaluation_samples": self.evaluation_samples,
            "benchmark": self.result.to_mapping(include_predictions=include_predictions),
        }


@dataclass(frozen=True)
class LongitudinalE001CaseResult:
    representation_id: str
    representation_sha256: str
    authority: Mapping[str, Any]
    provenance: Mapping[str, str]
    rows: tuple[LongitudinalE001Row, ...]
    study_fingerprint: str

    def to_mapping(self, *, include_predictions: bool = False) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "representation_id": self.representation_id,
            "representation_sha256": self.representation_sha256,
            "authority": dict(self.authority),
            "provenance": dict(self.provenance),
            "study_fingerprint": self.study_fingerprint,
            "rows": [row.to_mapping(include_predictions=include_predictions) for row in self.rows],
        }


def run_longitudinal_e001_case(
    data: Any,
    authority: Any,
    representations: np.ndarray,
    *,
    representation_id: str,
    budgets_per_class: Sequence[int],
    upstream_dataset_fingerprint: str,
    quantumbci_source_sha: str,
    neuros_source_sha: str,
    ridge: float = 1e-3,
    center_tokens: bool = True,
    covariance_regularization: float = 1e-6,
) -> LongitudinalE001CaseResult:
    """Run E001 on the exact calibration/evaluation authority frozen by neurOS.

    ``representations`` must be aligned one-to-one with ``data.X`` rows and have
    shape ``(n_samples, tokens, features)``. The function calls
    ``authority.restore(data)`` before touching those representations, so a real
    neurOS authority revalidates processed neural bytes and split identity first.

    Scientific identity additionally requires the upstream/raw dataset fingerprint
    and exact QuantumBCI/neurOS source revisions. A processed-data hash alone is not
    a substitute for upstream dataset provenance.
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

    values = np.asarray(representations)
    if np.iscomplexobj(values):
        raise ValueError(
            "longitudinal E001 currently requires real token representations so the "
            "matched classical control set is complete"
        )
    if values.ndim != 3:
        raise ValueError("representations must have shape (samples, tokens, features)")
    expected_samples = int(_authority_value(authority, "n_samples"))
    if len(values) != expected_samples:
        raise ValueError(
            f"representation rows ({len(values)}) do not match authority n_samples ({expected_samples})"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("representations contain non-finite values")
    labels = np.asarray(getattr(data, "y", None)).reshape(-1)
    if len(labels) != expected_samples:
        raise ValueError("data labels do not align with authority sample count")

    budgets = tuple(sorted(set(int(value) for value in budgets_per_class)))
    if not budgets or budgets[0] < 0:
        raise ValueError("budgets_per_class must contain non-negative values")
    max_budget = int(getattr(split, "max_budget_per_class"))
    if budgets[-1] > max_budget:
        raise ValueError(
            f"requested budget {budgets[-1]} exceeds authority maximum {max_budget}/class"
        )

    representation_sha = _hash_representation(values)
    identity = _case_identity(authority)
    rows: list[LongitudinalE001Row] = []
    evaluation = np.asarray(getattr(split, "evaluation_indices"), dtype=np.int64)
    source = np.asarray(getattr(split, "source_train_indices"), dtype=np.int64)
    train_for_budget = getattr(split, "train_indices_for_budget", None)
    calibration_for_budget = getattr(split, "calibration_indices", None)
    if not callable(train_for_budget) or not callable(calibration_for_budget):
        raise TypeError("restored split must expose train_indices_for_budget and calibration_indices")

    for budget in budgets:
        train = np.asarray(train_for_budget(budget), dtype=np.int64)
        calibration = np.asarray(calibration_for_budget(budget), dtype=np.int64)
        explicit = IndexSplit(
            train_indices=train,
            test_indices=evaluation,
            name=(
                f"{identity['case_id']}|authority={identity['authority_fingerprint']}"
                f"|calibration={budget}/class"
            ),
        )
        benchmark = benchmark_e001_embeddings(
            values,
            labels,
            explicit,
            ridge=ridge,
            center_tokens=center_tokens,
            covariance_regularization=covariance_regularization,
        )
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
        "schema_version": 2,
        "authority": identity,
        "provenance": provenance,
        "representation_id": representation_name,
        "representation_sha256": representation_sha,
        "budgets_per_class": list(budgets),
        "ridge": float(ridge),
        "center_tokens": bool(center_tokens),
        "covariance_regularization": float(covariance_regularization),
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


@dataclass(frozen=True)
class PairedBootstrapSummary:
    calibration_per_class: int
    control: str
    inference_unit: str
    n_units: int
    observed_mean_delta: float
    ci_lower: float
    ci_upper: float
    bootstrap_probability_positive: float
    n_resamples: int
    seed: int

    def to_mapping(self) -> dict[str, Any]:
        return {
            "calibration_per_class": self.calibration_per_class,
            "control": self.control,
            "inference_unit": self.inference_unit,
            "n_units": self.n_units,
            "observed_mean_delta": self.observed_mean_delta,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "bootstrap_probability_positive": self.bootstrap_probability_positive,
            "n_resamples": self.n_resamples,
            "seed": self.seed,
        }


def _unit_id(row: LongitudinalE001Row, inference_key: str) -> str:
    metadata = dict(row.case_metadata)
    value = metadata.get(inference_key)
    if value is None:
        raise ValueError(
            f"case {row.case_id!r} lacks case_metadata[{inference_key!r}]; "
            "participant-level inference must not silently fall back to sessions/cases"
        )
    return str(value)


def paired_participant_bootstrap(
    rows: Iterable[LongitudinalE001Row],
    *,
    control: str,
    inference_key: str = "subject",
    n_resamples: int = 5000,
    seed: int = 0,
) -> tuple[PairedBootstrapSummary, ...]:
    """Bootstrap density-vs-control deltas at the participant inference unit.

    Multiple held-out sessions/cases from the same participant are first averaged
    within participant at each calibration budget. Bootstrap resampling then occurs
    over participants, never over correlated trials/windows. Every budget must contain
    the same participant set, preserving a genuinely paired calibration frontier.
    """

    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100")
    materialized = tuple(rows)
    if not materialized:
        raise ValueError("rows must not be empty")
    budgets = sorted({row.calibration_per_class for row in materialized})

    unit_sets: dict[int, set[str]] = {}
    for budget in budgets:
        budget_rows = [row for row in materialized if row.calibration_per_class == budget]
        case_ids = [row.case_id for row in budget_rows]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError(f"duplicate case rows detected at calibration budget {budget}")
        unit_sets[budget] = {_unit_id(row, inference_key) for row in budget_rows}
    reference_budget = budgets[0]
    reference_units = unit_sets[reference_budget]
    for budget in budgets[1:]:
        if unit_sets[budget] != reference_units:
            missing = sorted(reference_units - unit_sets[budget])
            extra = sorted(unit_sets[budget] - reference_units)
            raise ValueError(
                "participant membership differs across calibration budgets; "
                f"budget={budget} missing={missing} extra={extra}"
            )

    summaries: list[PairedBootstrapSummary] = []
    for budget in budgets:
        budget_rows = [row for row in materialized if row.calibration_per_class == budget]
        by_unit: dict[str, list[float]] = {}
        for row in budget_rows:
            if control not in row.result.metrics:
                raise ValueError(f"unknown control {control!r} for case {row.case_id!r}")
            unit = _unit_id(row, inference_key)
            delta = (
                row.result.metrics["density"].balanced_accuracy
                - row.result.metrics[control].balanced_accuracy
            )
            by_unit.setdefault(unit, []).append(float(delta))
        units = sorted(by_unit)
        if len(units) < 2:
            raise ValueError("participant-level bootstrap requires at least two inference units")
        deltas = np.asarray([np.mean(by_unit[unit]) for unit in units], dtype=float)
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(budget)]))
        samples = rng.integers(0, len(deltas), size=(int(n_resamples), len(deltas)))
        boot = deltas[samples].mean(axis=1)
        lower, upper = np.quantile(boot, [0.025, 0.975])
        summaries.append(
            PairedBootstrapSummary(
                calibration_per_class=int(budget),
                control=str(control),
                inference_unit=str(inference_key),
                n_units=len(units),
                observed_mean_delta=float(deltas.mean()),
                ci_lower=float(lower),
                ci_upper=float(upper),
                bootstrap_probability_positive=float(np.mean(boot > 0.0)),
                n_resamples=int(n_resamples),
                seed=int(seed),
            )
        )
    return tuple(summaries)


def evaluate_density_information_gate(
    rows: Iterable[LongitudinalE001Row],
) -> dict[str, Any]:
    """Evaluate the hard information-novelty gate for the current E001 constructor."""

    materialized = tuple(rows)
    if not materialized:
        raise ValueError("rows must not be empty")
    equivalence_pass = all(
        bool(row.result.equivalence_audit.get("equivalent_within_tolerance"))
        for row in materialized
    )
    prediction_identity = all(
        np.array_equal(
            row.result.predictions["density"],
            row.result.predictions["normalized_covariance"],
        )
        for row in materialized
    )
    return {
        "schema_version": 1,
        "gate": "density_representation_information_novelty",
        "mathematical_equivalence_detected": bool(equivalence_pass),
        "normalized_covariance_prediction_identity": bool(prediction_identity),
        "promotion_eligible": False,
        "claim_class": "quantum_inspired",
        "reason": (
            "The current density constructor is exactly a trace-normalized Hermitian "
            "second moment. It cannot be promoted as containing information beyond the "
            "equivalent classical covariance representation. Off-diagonal ablation may "
            "show dependence on cross-covariance structure, not physical quantum coherence."
        ),
        "next_question": (
            "Test a non-equivalent downstream operator/dynamical/contextual mechanism with "
            "an explicit classical implementation of the same information as control."
        ),
    }
