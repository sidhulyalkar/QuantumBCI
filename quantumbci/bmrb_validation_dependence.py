"""Participant-dependence stress tests for BMRB known-truth validation.

The core and stress validation suites already prove that repeated observations are accepted
and that ordinary participant heterogeneity can be tolerated. This module attacks a harder
failure mode: unequal numbers of sessions can make row-pooled summaries disagree with the
participant-level estimand.

These scenarios are synthetic software-validation contracts, not biological models. The
production confirmatory evaluator remains the BMRB decision authority.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .bmrb_validation import validation_policy
from .confirmatory_representation import (
    ConfirmatoryRepresentationObservation,
    evaluate_confirmatory_representation,
)

BMRB_DEPENDENCE_STRESS_BENCHMARK = "BMRB_KNOWN_TRUTH_DEPENDENCE_STRESS_V1"


@dataclass(frozen=True)
class BMRBDependenceScenario:
    """Known-truth responder/session-count profile."""

    scenario_id: str
    expected_bmrb_pass: bool
    participants: int
    responder_count: int
    responder_sessions: int
    nonresponder_sessions: int
    responder_reference_effect: float = 0.12
    responder_alternate_effect: float = 0.115
    responder_reference_ablation: float = 0.10
    responder_alternate_ablation: float = 0.095
    nonresponder_reference_effect: float = 0.0
    nonresponder_alternate_effect: float = 0.0
    nonresponder_reference_ablation: float = 0.0
    nonresponder_alternate_ablation: float = 0.0
    measurement_sd: float = 0.0
    expected_failure_component: str | None = None

    def __post_init__(self) -> None:
        if not self.scenario_id.strip():
            raise ValueError("scenario_id must not be empty")
        if self.participants < 4:
            raise ValueError("dependence stress requires at least four participants")
        if not 0 < self.responder_count <= self.participants:
            raise ValueError("responder_count must lie in [1, participants]")
        if self.responder_sessions < 1 or self.nonresponder_sessions < 1:
            raise ValueError("session counts must be positive")
        if self.expected_failure_component not in {None, "effect"}:
            raise ValueError("dependence stress currently localizes only effect failures")
        for name in (
            "responder_reference_effect",
            "responder_alternate_effect",
            "responder_reference_ablation",
            "responder_alternate_ablation",
            "nonresponder_reference_effect",
            "nonresponder_alternate_effect",
            "nonresponder_reference_ablation",
            "nonresponder_alternate_ablation",
            "measurement_sd",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.measurement_sd < 0.0:
            raise ValueError("measurement_sd must be non-negative")


@dataclass(frozen=True)
class BMRBDependenceReplicate:
    scenario_id: str
    replicate: int
    expected_bmrb_pass: bool
    bmrb_scientific_passed: bool
    effect_criteria_passed: bool
    expected_failure_localized: bool
    row_weighted_effect_passed: bool
    participant_balanced_effect_passed: bool
    participant_count: int
    responder_count: int
    total_case_count: int
    min_sessions_per_participant: int
    max_sessions_per_participant: int

    def to_mapping(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "replicate": self.replicate,
            "expected_bmrb_pass": self.expected_bmrb_pass,
            "bmrb_scientific_passed": self.bmrb_scientific_passed,
            "effect_criteria_passed": self.effect_criteria_passed,
            "expected_failure_localized": self.expected_failure_localized,
            "row_weighted_effect_passed": self.row_weighted_effect_passed,
            "participant_balanced_effect_passed": self.participant_balanced_effect_passed,
            "participant_count": self.participant_count,
            "responder_count": self.responder_count,
            "total_case_count": self.total_case_count,
            "min_sessions_per_participant": self.min_sessions_per_participant,
            "max_sessions_per_participant": self.max_sessions_per_participant,
        }


@dataclass(frozen=True)
class BMRBDependenceSummary:
    scenario_id: str
    expected_bmrb_pass: bool
    replicates: int
    bmrb_pass_rate: float
    row_weighted_effect_pass_rate: float
    participant_balanced_effect_pass_rate: float
    expected_failure_localization_rate: float

    @property
    def bmrb_decision_error_rate(self) -> float:
        expected = float(self.expected_bmrb_pass)
        return float(abs(self.bmrb_pass_rate - expected))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "expected_bmrb_pass": self.expected_bmrb_pass,
            "replicates": self.replicates,
            "bmrb_pass_rate": self.bmrb_pass_rate,
            "bmrb_decision_error_rate": self.bmrb_decision_error_rate,
            "row_weighted_effect_pass_rate": self.row_weighted_effect_pass_rate,
            "participant_balanced_effect_pass_rate": self.participant_balanced_effect_pass_rate,
            "expected_failure_localization_rate": self.expected_failure_localization_rate,
        }


def default_dependence_scenarios() -> tuple[BMRBDependenceScenario, ...]:
    """Return deterministic responder/session-imbalance attacks."""

    return (
        BMRBDependenceScenario(
            scenario_id="majority-responder-imbalanced-positive",
            expected_bmrb_pass=True,
            participants=8,
            responder_count=7,
            responder_sessions=1,
            nonresponder_sessions=20,
        ),
        BMRBDependenceScenario(
            scenario_id="minority-responder-overweight-trap",
            expected_bmrb_pass=False,
            expected_failure_component="effect",
            participants=8,
            responder_count=2,
            responder_sessions=20,
            nonresponder_sessions=1,
        ),
    )


def _hex_identity(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _session_count(scenario: BMRBDependenceScenario, participant_index: int) -> int:
    if participant_index < scenario.responder_count:
        return scenario.responder_sessions
    return scenario.nonresponder_sessions


def _participant_centers(
    scenario: BMRBDependenceScenario,
    participant_index: int,
) -> tuple[float, float, float, float]:
    if participant_index < scenario.responder_count:
        return (
            scenario.responder_reference_effect,
            scenario.responder_alternate_effect,
            scenario.responder_reference_ablation,
            scenario.responder_alternate_ablation,
        )
    return (
        scenario.nonresponder_reference_effect,
        scenario.nonresponder_alternate_effect,
        scenario.nonresponder_reference_ablation,
        scenario.nonresponder_alternate_ablation,
    )


def _observations(
    scenario: BMRBDependenceScenario,
    *,
    rng: np.random.Generator,
    primary_budget: int,
) -> tuple[ConfirmatoryRepresentationObservation, ...]:
    rows: list[ConfirmatoryRepresentationObservation] = []
    for participant_index in range(scenario.participants):
        participant = f"p{participant_index + 1:03d}"
        (
            reference_effect,
            alternate_effect,
            reference_ablation,
            alternate_ablation,
        ) = _participant_centers(scenario, participant_index)
        lane_specs = (
            ("raw", "raw_neural", reference_effect, reference_ablation),
            ("latent", "synthetic_representation", alternate_effect, alternate_ablation),
        )
        sessions = _session_count(scenario, participant_index)
        for session_index in range(sessions):
            occasion = f"synthetic-session-{session_index + 1:02d}"
            authority = _hex_identity(
                f"authority:{scenario.scenario_id}:{participant}:{occasion}:{primary_budget}"
            )
            for lane_id, family, effect_center, ablation_center in lane_specs:
                effect = effect_center + rng.normal(0.0, scenario.measurement_sd)
                ablation = ablation_center + rng.normal(0.0, scenario.measurement_sd)
                control = 0.64 if lane_id == "raw" else 0.67
                candidate = control + effect
                rows.append(
                    ConfirmatoryRepresentationObservation(
                        participant_id=participant,
                        occasion_id=occasion,
                        case_id=f"{participant}-{occasion}",
                        calibration_per_class=primary_budget,
                        representation_id=lane_id,
                        representation_family=family,
                        authority_fingerprint=authority,
                        representation_sha256=_hex_identity(
                            f"representation:{scenario.scenario_id}:{lane_id}"
                        ),
                        source_fingerprint=_hex_identity(
                            f"source:{scenario.scenario_id}:{participant}:{occasion}"
                        ),
                        candidate_metric=candidate,
                        primary_control_metric=control,
                        ablated_metric=candidate - ablation,
                        information_novel=True,
                        model_id=None if lane_id == "raw" else "synthetic-latent",
                        model_revision=None if lane_id == "raw" else "dependence-stress-v1",
                    )
                )
    return tuple(rows)


def _row_weighted_effect_pass(
    observations: Iterable[ConfirmatoryRepresentationObservation],
    *,
    threshold: float,
) -> bool:
    rows = tuple(observations)
    if not rows:
        raise ValueError("row-weighted comparator requires observations")
    return bool(np.mean([row.candidate_advantage for row in rows]) >= threshold)


def _participant_balanced_effect_pass(
    observations: Iterable[ConfirmatoryRepresentationObservation],
    *,
    threshold: float,
) -> bool:
    grouped: dict[str, list[float]] = {}
    for row in observations:
        grouped.setdefault(row.participant_id, []).append(row.candidate_advantage)
    if not grouped:
        raise ValueError("participant-balanced comparator requires observations")
    participant_means = [float(np.mean(values)) for values in grouped.values()]
    return bool(np.mean(participant_means) >= threshold)


def run_dependence_replicate(
    scenario: BMRBDependenceScenario,
    *,
    replicate: int,
    seed: int,
    bootstrap_resamples: int = 100,
) -> BMRBDependenceReplicate:
    """Run one dependence stress replicate through production BMRB semantics."""

    policy = validation_policy(
        participants=scenario.participants,
        inference_seed=int(seed) + int(replicate) * 17,
        bootstrap_resamples=bootstrap_resamples,
    )
    rng = np.random.default_rng(int(seed) + int(replicate) * 1009)
    observations = _observations(
        scenario,
        rng=rng,
        primary_budget=policy.primary_calibration_per_class,
    )
    result = evaluate_confirmatory_representation(
        observations,
        study_id=f"dependence-{scenario.scenario_id}-{replicate:04d}",
        mechanism_id=scenario.scenario_id,
        policy=policy,
    )
    if scenario.expected_failure_component == "effect":
        localized = not result.effect_criteria_passed
    else:
        localized = result.scientific_criteria_passed

    session_counts = [
        _session_count(scenario, index) for index in range(scenario.participants)
    ]
    unique_cases = {
        (row.participant_id, row.occasion_id, row.case_id) for row in observations
    }
    return BMRBDependenceReplicate(
        scenario_id=scenario.scenario_id,
        replicate=int(replicate),
        expected_bmrb_pass=scenario.expected_bmrb_pass,
        bmrb_scientific_passed=result.scientific_criteria_passed,
        effect_criteria_passed=result.effect_criteria_passed,
        expected_failure_localized=bool(localized),
        row_weighted_effect_passed=_row_weighted_effect_pass(
            observations,
            threshold=policy.min_candidate_advantage,
        ),
        participant_balanced_effect_passed=_participant_balanced_effect_pass(
            observations,
            threshold=policy.min_candidate_advantage,
        ),
        participant_count=result.participant_count,
        responder_count=scenario.responder_count,
        total_case_count=len(unique_cases),
        min_sessions_per_participant=min(session_counts),
        max_sessions_per_participant=max(session_counts),
    )


def validate_structured_missing_pair_rejection(*, seed: int = 3901) -> dict[str, Any]:
    """Prove structured missing representation rows are invalid, not negative evidence."""

    scenario = BMRBDependenceScenario(
        scenario_id="structured-missing-pair",
        expected_bmrb_pass=True,
        participants=8,
        responder_count=8,
        responder_sessions=2,
        nonresponder_sessions=2,
    )
    policy = validation_policy(
        participants=scenario.participants,
        inference_seed=seed,
        bootstrap_resamples=100,
    )
    observations = list(
        _observations(
            scenario,
            rng=np.random.default_rng(seed),
            primary_budget=policy.primary_calibration_per_class,
        )
    )
    before = len(observations)
    observations = [
        row
        for row in observations
        if not (
            row.representation_id == "latent"
            and row.participant_id in {"p002", "p006"}
            and row.occasion_id == "synthetic-session-02"
        )
    ]
    removed = before - len(observations)
    try:
        evaluate_confirmatory_representation(
            observations,
            study_id="dependence-structured-missing-pair",
            mechanism_id=scenario.scenario_id,
            policy=policy,
        )
    except ValueError as exc:
        return {
            "rejected": "exactly paired" in str(exc),
            "removed_representation_rows": removed,
            "classification": "software_invalid",
            "scientific_negative": False,
            "reason": str(exc),
        }
    return {
        "rejected": False,
        "removed_representation_rows": removed,
        "classification": "unexpected_valid",
        "scientific_negative": False,
        "reason": None,
    }


def _summary(
    scenario: BMRBDependenceScenario,
    rows: Iterable[BMRBDependenceReplicate],
) -> BMRBDependenceSummary:
    materialized = tuple(rows)
    if not materialized:
        raise ValueError("dependence stress summary requires replicates")
    return BMRBDependenceSummary(
        scenario_id=scenario.scenario_id,
        expected_bmrb_pass=scenario.expected_bmrb_pass,
        replicates=len(materialized),
        bmrb_pass_rate=float(
            np.mean([row.bmrb_scientific_passed for row in materialized])
        ),
        row_weighted_effect_pass_rate=float(
            np.mean([row.row_weighted_effect_passed for row in materialized])
        ),
        participant_balanced_effect_pass_rate=float(
            np.mean(
                [row.participant_balanced_effect_passed for row in materialized]
            )
        ),
        expected_failure_localization_rate=float(
            np.mean([row.expected_failure_localized for row in materialized])
        ),
    )


def run_bmrb_dependence_stress_suite(
    *,
    replicates: int = 4,
    seed: int = 3901,
    bootstrap_resamples: int = 100,
) -> dict[str, Any]:
    """Run responder/session-imbalance attacks and structured missingness validation."""

    if replicates < 1:
        raise ValueError("replicates must be positive")
    scenarios = default_dependence_scenarios()
    summaries: list[BMRBDependenceSummary] = []
    all_rows: list[BMRBDependenceReplicate] = []
    for scenario_index, scenario in enumerate(scenarios):
        rows = [
            run_dependence_replicate(
                scenario,
                replicate=replicate,
                seed=seed + scenario_index * 100_003,
                bootstrap_resamples=bootstrap_resamples,
            )
            for replicate in range(replicates)
        ]
        all_rows.extend(rows)
        summaries.append(_summary(scenario, rows))

    by_id = {summary.scenario_id: summary for summary in summaries}
    majority = by_id["majority-responder-imbalanced-positive"]
    minority = by_id["minority-responder-overweight-trap"]
    missing = validate_structured_missing_pair_rejection(seed=seed + 700_001)

    qualification = {
        "bmrb_decision_error_rate_max": 0.0,
        "row_weighted_majority_false_reject_rate_min": 1.0,
        "row_weighted_minority_false_accept_rate_min": 1.0,
        "participant_balanced_alignment_rate_min": 1.0,
        "expected_failure_localization_rate_min": 1.0,
        "structured_missing_pair_must_reject": True,
    }
    participant_alignment = float(
        np.mean(
            [
                row.participant_balanced_effect_passed == row.expected_bmrb_pass
                for row in all_rows
            ]
        )
    )
    qualified = bool(
        all(
            summary.bmrb_decision_error_rate
            <= qualification["bmrb_decision_error_rate_max"]
            for summary in summaries
        )
        and (1.0 - majority.row_weighted_effect_pass_rate)
        >= qualification["row_weighted_majority_false_reject_rate_min"]
        and minority.row_weighted_effect_pass_rate
        >= qualification["row_weighted_minority_false_accept_rate_min"]
        and participant_alignment
        >= qualification["participant_balanced_alignment_rate_min"]
        and minority.expected_failure_localization_rate
        >= qualification["expected_failure_localization_rate_min"]
        and missing["rejected"] is qualification["structured_missing_pair_must_reject"]
    )
    return {
        "schema_version": 1,
        "benchmark": BMRB_DEPENDENCE_STRESS_BENCHMARK,
        "seed": int(seed),
        "replicates_per_scenario": int(replicates),
        "bootstrap_resamples": int(bootstrap_resamples),
        "qualification_policy": qualification,
        "qualified": qualified,
        "participant_balanced_alignment_rate": participant_alignment,
        "scenario_summaries": [summary.to_mapping() for summary in summaries],
        "replicates": [row.to_mapping() for row in all_rows],
        "structured_missing_pair": missing,
        "qualification_scope": (
            "Deterministic software validation of participant-level weighting, responder "
            "mixtures, session imbalance, and exact-pairing failure behavior."
        ),
        "interpretation": (
            "These synthetic attacks validate BMRB decision behavior under declared evidence "
            "patterns. They do not validate biological truth or authorize neural or physical-"
            "quantum mechanism claims."
        ),
    }
