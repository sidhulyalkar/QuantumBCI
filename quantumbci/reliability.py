"""Repeated-case reliability evidence for Brain Mechanism Recapitulation Benchmarking.

This module deliberately separates two questions that are often conflated:

1. population recurrence: does a mechanism quantity recur with a consistent direction
   across independent participants/cases?
2. person-specific reliability: are between-participant differences reproducible across
   repeated occasions?

An intraclass correlation coefficient (ICC) addresses the second question, not the
first. A mechanism can recur strongly in every participant while having a low ICC when
true between-participant variance is small. QuantumBCI therefore reports both surfaces
and computes ICC only for a complete balanced participant x occasion panel.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_RELIABILITY_BOOTSTRAP_RESAMPLES = 5000
DEFAULT_RELIABILITY_SEED = 1501
RELIABILITY_BOOTSTRAP_METHOD_ID = "participant_primary_hierarchical_bootstrap_v1"
ICC_METHOD_ID = "icc_a1_two_way_random_absolute_agreement_balanced_v1"


def _required_text(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _stable_seed_component(value: str) -> int:
    digest = sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


@dataclass(frozen=True)
class RepeatedCaseEstimate:
    """One qualified mechanism estimate from one participant and one occasion."""

    participant_id: str
    occasion_id: str
    case_id: str
    estimate_name: str
    value: float
    authority_fingerprint: str
    data_sha256: str
    artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "participant_id",
            "occasion_id",
            "case_id",
            "estimate_name",
            "authority_fingerprint",
            "data_sha256",
        ):
            _required_text(name, getattr(self, name))
        if not np.isfinite(float(self.value)):
            raise ValueError("RepeatedCaseEstimate.value must be finite")
        if self.artifact_sha256 is not None:
            _required_text("artifact_sha256", self.artifact_sha256)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "occasion_id": self.occasion_id,
            "case_id": self.case_id,
            "estimate_name": self.estimate_name,
            "value": float(self.value),
            "authority_fingerprint": self.authority_fingerprint,
            "data_sha256": self.data_sha256,
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True)
class ICCResult:
    """Balanced-panel ICC(A,1) with its ANOVA components."""

    method_id: str
    value: float
    n_participants: int
    n_occasions: int
    ms_participant: float
    ms_occasion: float
    ms_error: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "value": float(self.value),
            "n_participants": int(self.n_participants),
            "n_occasions": int(self.n_occasions),
            "ms_participant": float(self.ms_participant),
            "ms_occasion": float(self.ms_occasion),
            "ms_error": float(self.ms_error),
            "interpretation": (
                "ICC(A,1) measures absolute-agreement reliability of individual differences "
                "across repeated occasions. It is not a population-recurrence statistic."
            ),
        }


@dataclass(frozen=True)
class RepeatedCaseReliabilityResult:
    estimate_name: str
    n_cases: int
    n_participants: int
    n_occasions: int
    balanced_panel: bool
    participant_ids: tuple[str, ...]
    occasion_ids: tuple[str, ...]
    grand_mean: float
    participant_mean_std: float
    within_participant_std: float | None
    case_positive_fraction: float
    participant_positive_fraction: float
    population_sign_consistency: float
    bootstrap_mean: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    bootstrap_probability_positive: float
    n_resamples: int
    seed: int
    icc: ICCResult | None
    icc_unavailable_reason: str | None
    source_fingerprint: str

    @property
    def repeated_case_evidence_available(self) -> bool:
        return self.n_participants >= 2 and self.n_cases > self.n_participants

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "estimate_name": self.estimate_name,
            "n_cases": int(self.n_cases),
            "n_participants": int(self.n_participants),
            "n_occasions": int(self.n_occasions),
            "balanced_panel": bool(self.balanced_panel),
            "participant_ids": list(self.participant_ids),
            "occasion_ids": list(self.occasion_ids),
            "grand_mean": float(self.grand_mean),
            "participant_mean_std": float(self.participant_mean_std),
            "within_participant_std": self.within_participant_std,
            "case_positive_fraction": float(self.case_positive_fraction),
            "participant_positive_fraction": float(self.participant_positive_fraction),
            "population_sign_consistency": float(self.population_sign_consistency),
            "bootstrap": {
                "method_id": RELIABILITY_BOOTSTRAP_METHOD_ID,
                "mean": float(self.bootstrap_mean),
                "ci_percentiles": [2.5, 97.5],
                "ci_low": float(self.bootstrap_ci_low),
                "ci_high": float(self.bootstrap_ci_high),
                "probability_positive": float(self.bootstrap_probability_positive),
                "n_resamples": int(self.n_resamples),
                "seed": int(self.seed),
            },
            "icc": None if self.icc is None else self.icc.to_mapping(),
            "icc_unavailable_reason": self.icc_unavailable_reason,
            "repeated_case_evidence_available": bool(self.repeated_case_evidence_available),
            "reliability_gate_defined": False,
            "reliability_gate_pass": None,
            "source_fingerprint": self.source_fingerprint,
            "interpretation": (
                "Population sign consistency asks whether the mechanism quantity recurs in the "
                "same direction across participants. ICC, when available, asks whether stable "
                "between-participant differences are preserved across occasions. Neither is a "
                "universal binary mechanism-reliability gate without a preregistered criterion."
            ),
        }


@dataclass(frozen=True)
class RepeatedCaseReliabilityBundle:
    study_id: str
    case_count: int
    participant_count: int
    estimate_names: tuple[str, ...]
    results: tuple[RepeatedCaseReliabilityResult, ...]
    source_fingerprint: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_role": "repeated_case_reliability_evidence",
            "study_id": self.study_id,
            "case_count": int(self.case_count),
            "participant_count": int(self.participant_count),
            "estimate_names": list(self.estimate_names),
            "source_fingerprint": self.source_fingerprint,
            "results": [result.to_mapping() for result in self.results],
            "single_case_bootstrap_is_icc": False,
            "repeated_case_icc_requires_balanced_panel": True,
            "reliability_gate_defined": False,
            "reliability_gate_pass": None,
        }


def estimates_from_stability_artifact(
    artifact: Mapping[str, Any],
    *,
    participant_id: str,
    occasion_id: str,
    case_id: str,
    artifact_sha256: str | None = None,
    estimate_names: Sequence[str] | None = None,
) -> tuple[RepeatedCaseEstimate, ...]:
    """Extract point estimates from one independently qualified v0.14 artifact."""

    if artifact.get("experiment") != "E002":
        raise ValueError("stability artifact is not E002")
    if artifact.get("artifact_role") != "bootstrap_stability_evidence":
        raise ValueError("artifact has the wrong role for repeated-case reliability")
    if artifact.get("status") != "pass":
        raise ValueError("stability artifact did not pass execution")
    if bool(artifact.get("evaluation_resampled", True)):
        raise ValueError("repeated-case reliability requires fixed final evaluation evidence")
    if bool(artifact.get("single_case_bootstrap_is_icc", True)):
        raise ValueError("upstream artifact incorrectly labels single-case bootstrap as ICC")
    authority_fingerprint = _required_text(
        "authority_fingerprint", artifact.get("authority_fingerprint")
    )
    data_sha256 = _required_text("data_sha256", artifact.get("data_sha256"))
    point_estimates = artifact.get("point_estimates")
    if not isinstance(point_estimates, Mapping) or not point_estimates:
        raise ValueError("stability artifact is missing point_estimates")

    names = (
        tuple(str(name) for name in estimate_names)
        if estimate_names is not None
        else tuple(sorted(str(name) for name in point_estimates))
    )
    if not names:
        raise ValueError("estimate_names must not be empty")
    rows: list[RepeatedCaseEstimate] = []
    for name in names:
        if name not in point_estimates:
            raise ValueError(f"stability artifact does not contain estimate {name!r}")
        value = float(point_estimates[name])
        rows.append(
            RepeatedCaseEstimate(
                participant_id=_required_text("participant_id", participant_id),
                occasion_id=_required_text("occasion_id", occasion_id),
                case_id=_required_text("case_id", case_id),
                estimate_name=name,
                value=value,
                authority_fingerprint=authority_fingerprint,
                data_sha256=data_sha256,
                artifact_sha256=artifact_sha256,
            )
        )
    return tuple(rows)


def _validate_rows(rows: Sequence[RepeatedCaseEstimate]) -> None:
    if not rows:
        raise ValueError("repeated-case reliability requires estimates")
    keys: set[tuple[str, str, str]] = set()
    case_identity: dict[str, tuple[str, str, str, str | None]] = {}
    for row in rows:
        key = (row.estimate_name, row.participant_id, row.occasion_id)
        if key in keys:
            raise ValueError(
                "duplicate participant/occasion estimate detected: "
                f"estimate={row.estimate_name} participant={row.participant_id} "
                f"occasion={row.occasion_id}"
            )
        keys.add(key)
        identity = (
            row.participant_id,
            row.occasion_id,
            row.authority_fingerprint,
            row.artifact_sha256,
        )
        previous = case_identity.setdefault(row.case_id, identity)
        if previous != identity:
            raise ValueError(f"case_id {row.case_id!r} maps to conflicting evidence identity")


def _source_fingerprint(rows: Sequence[RepeatedCaseEstimate]) -> str:
    payload = [row.to_mapping() for row in sorted(
        rows,
        key=lambda item: (
            item.estimate_name,
            item.participant_id,
            item.occasion_id,
            item.case_id,
        ),
    )]
    return sha256(
        b"quantumbci.repeated-case-reliability.v1\0"
        + _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _balanced_matrix(
    rows: Sequence[RepeatedCaseEstimate],
) -> tuple[np.ndarray | None, tuple[str, ...], tuple[str, ...], str | None]:
    participants = tuple(sorted({row.participant_id for row in rows}))
    occasions = tuple(sorted({row.occasion_id for row in rows}))
    if len(participants) < 2:
        return None, participants, occasions, "ICC requires at least two participants"
    if len(occasions) < 2:
        return None, participants, occasions, "ICC requires at least two repeated occasions"
    lookup = {(row.participant_id, row.occasion_id): float(row.value) for row in rows}
    expected = {(participant, occasion) for participant in participants for occasion in occasions}
    missing = sorted(expected - set(lookup))
    if missing:
        return (
            None,
            participants,
            occasions,
            "ICC(A,1) requires a complete balanced participant x occasion panel; "
            f"missing={missing[:8]}",
        )
    matrix = np.asarray(
        [[lookup[(participant, occasion)] for occasion in occasions] for participant in participants],
        dtype=float,
    )
    return matrix, participants, occasions, None


def _icc_a1(matrix: np.ndarray) -> tuple[ICCResult | None, str | None]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("ICC matrix must be two-dimensional")
    n, k = values.shape
    if n < 2 or k < 2:
        return None, "ICC(A,1) requires at least two participants and two occasions"
    grand = float(np.mean(values))
    participant_means = np.mean(values, axis=1)
    occasion_means = np.mean(values, axis=0)
    ss_participant = float(k * np.sum((participant_means - grand) ** 2))
    ss_occasion = float(n * np.sum((occasion_means - grand) ** 2))
    residual = values - participant_means[:, None] - occasion_means[None, :] + grand
    ss_error = float(np.sum(residual**2))
    ms_participant = ss_participant / (n - 1)
    ms_occasion = ss_occasion / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denominator = (
        ms_participant
        + (k - 1) * ms_error
        + (k * (ms_occasion - ms_error) / n)
    )
    tolerance = max(1e-15, abs(ms_participant) * 1e-14)
    if not np.isfinite(denominator) or abs(denominator) <= tolerance:
        return None, "ICC(A,1) denominator is numerically degenerate"
    value = (ms_participant - ms_error) / denominator
    if not np.isfinite(value):
        return None, "ICC(A,1) is non-finite"
    return (
        ICCResult(
            method_id=ICC_METHOD_ID,
            value=float(value),
            n_participants=n,
            n_occasions=k,
            ms_participant=float(ms_participant),
            ms_occasion=float(ms_occasion),
            ms_error=float(ms_error),
        ),
        None,
    )


def _hierarchical_bootstrap(
    rows: Sequence[RepeatedCaseEstimate],
    *,
    n_resamples: int,
    seed: int,
) -> tuple[float, float, float, float]:
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100")
    by_participant: dict[str, np.ndarray] = {}
    for participant in sorted({row.participant_id for row in rows}):
        values = np.asarray(
            [row.value for row in rows if row.participant_id == participant], dtype=float
        )
        by_participant[participant] = values
    participants = tuple(sorted(by_participant))
    if len(participants) < 2:
        raise ValueError("hierarchical bootstrap requires at least two participants")
    rng = np.random.default_rng(seed)
    boot = np.empty(int(n_resamples), dtype=float)
    for replicate in range(int(n_resamples)):
        selected = rng.integers(0, len(participants), size=len(participants))
        participant_means: list[float] = []
        for index in selected:
            values = by_participant[participants[int(index)]]
            within = rng.integers(0, len(values), size=len(values))
            participant_means.append(float(np.mean(values[within])))
        boot[replicate] = float(np.mean(participant_means))
    low, high = np.quantile(boot, [0.025, 0.975])
    return (
        float(np.mean(boot)),
        float(low),
        float(high),
        float(np.mean(boot > 0.0)),
    )


def audit_repeated_case_estimate(
    rows: Iterable[RepeatedCaseEstimate],
    *,
    n_resamples: int = DEFAULT_RELIABILITY_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_RELIABILITY_SEED,
) -> RepeatedCaseReliabilityResult:
    materialized = tuple(rows)
    _validate_rows(materialized)
    names = {row.estimate_name for row in materialized}
    if len(names) != 1:
        raise ValueError("audit_repeated_case_estimate requires exactly one estimate_name")
    participants = tuple(sorted({row.participant_id for row in materialized}))
    if len(participants) < 2:
        raise ValueError("repeated-case reliability requires at least two participants")

    by_participant = {
        participant: np.asarray(
            [row.value for row in materialized if row.participant_id == participant], dtype=float
        )
        for participant in participants
    }
    participant_means = np.asarray(
        [np.mean(by_participant[participant]) for participant in participants], dtype=float
    )
    grand_mean = float(np.mean(participant_means))
    participant_mean_std = (
        float(np.std(participant_means, ddof=1)) if len(participant_means) > 1 else 0.0
    )
    residual_ss = 0.0
    residual_df = 0
    for values in by_participant.values():
        if len(values) > 1:
            residual_ss += float(np.sum((values - np.mean(values)) ** 2))
            residual_df += len(values) - 1
    within_participant_std = (
        float(np.sqrt(residual_ss / residual_df)) if residual_df > 0 else None
    )
    all_values = np.asarray([row.value for row in materialized], dtype=float)
    case_positive_fraction = float(np.mean(all_values > 0.0))
    participant_positive_fraction = float(np.mean(participant_means > 0.0))
    tolerance = max(1e-12, abs(grand_mean) * 1e-12)
    if grand_mean > tolerance:
        sign_consistency = participant_positive_fraction
    elif grand_mean < -tolerance:
        sign_consistency = float(np.mean(participant_means < 0.0))
    else:
        sign_consistency = float(np.mean(np.abs(participant_means) <= tolerance))

    estimate_name = next(iter(names))
    local_seed = int(np.random.SeedSequence([
        int(seed), _stable_seed_component(estimate_name)
    ]).generate_state(1)[0])
    boot_mean, boot_low, boot_high, boot_positive = _hierarchical_bootstrap(
        materialized, n_resamples=n_resamples, seed=local_seed
    )

    matrix, _, occasions, unavailable = _balanced_matrix(materialized)
    icc: ICCResult | None = None
    icc_reason = unavailable
    if matrix is not None:
        icc, icc_reason = _icc_a1(matrix)

    return RepeatedCaseReliabilityResult(
        estimate_name=estimate_name,
        n_cases=len(materialized),
        n_participants=len(participants),
        n_occasions=len(occasions),
        balanced_panel=matrix is not None,
        participant_ids=participants,
        occasion_ids=occasions,
        grand_mean=grand_mean,
        participant_mean_std=participant_mean_std,
        within_participant_std=within_participant_std,
        case_positive_fraction=case_positive_fraction,
        participant_positive_fraction=participant_positive_fraction,
        population_sign_consistency=sign_consistency,
        bootstrap_mean=boot_mean,
        bootstrap_ci_low=boot_low,
        bootstrap_ci_high=boot_high,
        bootstrap_probability_positive=boot_positive,
        n_resamples=int(n_resamples),
        seed=local_seed,
        icc=icc,
        icc_unavailable_reason=icc_reason,
        source_fingerprint=_source_fingerprint(materialized),
    )


def audit_repeated_case_reliability(
    rows: Iterable[RepeatedCaseEstimate],
    *,
    study_id: str,
    n_resamples: int = DEFAULT_RELIABILITY_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_RELIABILITY_SEED,
) -> RepeatedCaseReliabilityBundle:
    """Audit all estimate names in a participant/occasion evidence collection."""

    materialized = tuple(rows)
    _validate_rows(materialized)
    study = _required_text("study_id", study_id)
    estimate_names = tuple(sorted({row.estimate_name for row in materialized}))
    results = tuple(
        audit_repeated_case_estimate(
            [row for row in materialized if row.estimate_name == name],
            n_resamples=n_resamples,
            seed=seed,
        )
        for name in estimate_names
    )
    case_ids = {row.case_id for row in materialized}
    participant_ids = {row.participant_id for row in materialized}
    return RepeatedCaseReliabilityBundle(
        study_id=study,
        case_count=len(case_ids),
        participant_count=len(participant_ids),
        estimate_names=estimate_names,
        results=results,
        source_fingerprint=_source_fingerprint(materialized),
    )
