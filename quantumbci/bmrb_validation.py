"""Known-ground-truth validation program for BMRB decision behavior.

This module does not simulate biology. It simulates participant-level evidence patterns whose
truth is declared in advance, then asks whether the confirmatory BMRB machinery reaches the
intended decision and localizes the intended failure mode.

The validation suite is deliberately downstream of the production confirmatory evaluator. It
does not reimplement its gates, so a regression in BMRB decision semantics is visible here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .confirmatory_representation import (
    ConfirmatoryRepresentationObservation,
    ConfirmatoryRepresentationPolicy,
    ConfirmatoryRepresentationResult,
    evaluate_confirmatory_representation,
)

BMRB_VALIDATION_BENCHMARK = "BMRB_KNOWN_TRUTH_VALIDATION_V1"


@dataclass(frozen=True)
class BMRBValidationScenario:
    """One declared data-generating mechanism for benchmark validation."""

    scenario_id: str
    truth_class: str
    expected_scientific_pass: bool
    expected_failure_component: str | None
    reference_effect: float
    alternate_effect: float
    reference_ablation: float
    alternate_ablation: float
    information_novel: bool
    participant_effect_sd: float = 0.015
    measurement_sd: float = 0.005
    secondary_budget_effect: float | None = None
    secondary_budget_ablation: float | None = None

    def __post_init__(self) -> None:
        if self.truth_class not in {"known_null", "known_positive", "adversarial"}:
            raise ValueError("truth_class must be known_null, known_positive, or adversarial")
        if self.expected_failure_component not in {
            None,
            "effect",
            "adversary",
            "conservation",
            "coverage",
        }:
            raise ValueError("unknown expected_failure_component")
        for name in (
            "reference_effect",
            "alternate_effect",
            "reference_ablation",
            "alternate_ablation",
            "participant_effect_sd",
            "measurement_sd",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.participant_effect_sd < 0 or self.measurement_sd < 0:
            raise ValueError("noise scales must be non-negative")


@dataclass(frozen=True)
class BMRBValidationReplicate:
    scenario_id: str
    replicate: int
    scientific_criteria_passed: bool
    effect_criteria_passed: bool
    adversary_survival_passed: bool
    conservation_criteria_passed: bool
    coverage_criteria_passed: bool
    reference_observed_effect: float
    reference_effect_ci_lower: float
    reference_effect_ci_upper: float
    reference_effect_bias: float
    reference_ci_covers_truth: bool
    expected_failure_localized: bool

    def to_mapping(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "replicate": self.replicate,
            "scientific_criteria_passed": self.scientific_criteria_passed,
            "effect_criteria_passed": self.effect_criteria_passed,
            "adversary_survival_passed": self.adversary_survival_passed,
            "conservation_criteria_passed": self.conservation_criteria_passed,
            "coverage_criteria_passed": self.coverage_criteria_passed,
            "reference_observed_effect": self.reference_observed_effect,
            "reference_effect_ci_lower": self.reference_effect_ci_lower,
            "reference_effect_ci_upper": self.reference_effect_ci_upper,
            "reference_effect_bias": self.reference_effect_bias,
            "reference_ci_covers_truth": self.reference_ci_covers_truth,
            "expected_failure_localized": self.expected_failure_localized,
        }


@dataclass(frozen=True)
class BMRBValidationScenarioSummary:
    scenario_id: str
    truth_class: str
    expected_scientific_pass: bool
    expected_failure_component: str | None
    replicates: int
    observed_pass_rate: float
    decision_error_rate: float
    expected_failure_localization_rate: float
    mean_reference_effect_bias: float
    reference_ci_coverage: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "truth_class": self.truth_class,
            "expected_scientific_pass": self.expected_scientific_pass,
            "expected_failure_component": self.expected_failure_component,
            "replicates": self.replicates,
            "observed_pass_rate": self.observed_pass_rate,
            "decision_error_rate": self.decision_error_rate,
            "expected_failure_localization_rate": self.expected_failure_localization_rate,
            "mean_reference_effect_bias": self.mean_reference_effect_bias,
            "reference_ci_coverage": self.reference_ci_coverage,
        }


def default_validation_scenarios() -> tuple[BMRBValidationScenario, ...]:
    """Return the first ADEMP-style known-truth adversary grid.

    The scenarios isolate distinct questions rather than collapsing them into one score:
    effect-null rejection, mathematical-equivalence rejection, true shared-mechanism
    recovery, shortcut rejection, representation-specific failure, and calibration-budget
    reversal.
    """

    return (
        BMRBValidationScenario(
            scenario_id="effect-null",
            truth_class="known_null",
            expected_scientific_pass=False,
            expected_failure_component="effect",
            reference_effect=0.0,
            alternate_effect=0.0,
            reference_ablation=0.10,
            alternate_ablation=0.10,
            information_novel=True,
        ),
        BMRBValidationScenario(
            scenario_id="equivalence-null",
            truth_class="known_null",
            expected_scientific_pass=False,
            expected_failure_component="adversary",
            reference_effect=0.12,
            alternate_effect=0.12,
            reference_ablation=0.10,
            alternate_ablation=0.10,
            information_novel=False,
        ),
        BMRBValidationScenario(
            scenario_id="shared-mechanism-positive",
            truth_class="known_positive",
            expected_scientific_pass=True,
            expected_failure_component=None,
            reference_effect=0.12,
            alternate_effect=0.115,
            reference_ablation=0.10,
            alternate_ablation=0.095,
            information_novel=True,
        ),
        BMRBValidationScenario(
            scenario_id="predictive-shortcut",
            truth_class="adversarial",
            expected_scientific_pass=False,
            expected_failure_component="conservation",
            reference_effect=0.12,
            alternate_effect=0.115,
            reference_ablation=0.0,
            alternate_ablation=0.0,
            information_novel=True,
        ),
        BMRBValidationScenario(
            scenario_id="representation-specific",
            truth_class="adversarial",
            expected_scientific_pass=False,
            expected_failure_component="conservation",
            reference_effect=0.12,
            alternate_effect=-0.02,
            reference_ablation=0.10,
            alternate_ablation=-0.01,
            information_novel=True,
        ),
        BMRBValidationScenario(
            scenario_id="calibration-reversal",
            truth_class="known_positive",
            expected_scientific_pass=True,
            expected_failure_component=None,
            reference_effect=0.12,
            alternate_effect=0.115,
            reference_ablation=0.10,
            alternate_ablation=0.095,
            information_novel=True,
            secondary_budget_effect=-0.08,
            secondary_budget_ablation=-0.05,
        ),
    )


def validation_policy(
    *,
    participants: int = 8,
    primary_calibration_per_class: int = 10,
    inference_seed: int = 1901,
    bootstrap_resamples: int = 300,
) -> ConfirmatoryRepresentationPolicy:
    """Return the fixed software-validation policy.

    These thresholds are not universal biological promotion defaults. They exist only to
    test whether BMRB's declared gates respond correctly to known synthetic evidence.
    """

    if participants < 4:
        raise ValueError("validation requires at least four participants")
    return ConfirmatoryRepresentationPolicy(
        policy_id="bmrb-known-truth-validation-v1",
        reference_representation_id="raw",
        primary_calibration_per_class=primary_calibration_per_class,
        primary_classical_control="matched_control",
        min_participants=participants,
        min_representations=2,
        min_representation_families=2,
        min_candidate_advantage=0.05,
        min_ablation_necessity=0.05,
        min_reference_positive_fraction=0.75,
        min_all_lane_positive_fraction=0.75,
        min_direction_match_fraction=0.75,
        min_ablation_direction_match_fraction=0.75,
        min_information_novel_representation_fraction=1.0,
        sample_size_rationale=(
            "Synthetic software-validation grid; participant count is chosen for deterministic "
            "gate qualification, not biological power."
        ),
        inference_seed=inference_seed,
        bootstrap_resamples=bootstrap_resamples,
        preregistration=None,
    )


def _component_passed(result: ConfirmatoryRepresentationResult, component: str) -> bool:
    return {
        "effect": result.effect_criteria_passed,
        "adversary": result.adversary_survival_passed,
        "conservation": result.conservation_criteria_passed,
        "coverage": result.coverage_criteria_passed,
    }[component]


def _hex_identity(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _observations_for_scenario(
    scenario: BMRBValidationScenario,
    *,
    rng: np.random.Generator,
    participants: int,
    primary_budget: int,
) -> tuple[ConfirmatoryRepresentationObservation, ...]:
    rows: list[ConfirmatoryRepresentationObservation] = []
    lane_specs = (
        ("raw", "raw_neural", scenario.reference_effect, scenario.reference_ablation),
        (
            "latent",
            "synthetic_representation",
            scenario.alternate_effect,
            scenario.alternate_ablation,
        ),
    )
    budgets = [primary_budget]
    if scenario.secondary_budget_effect is not None:
        budgets.insert(0, 0 if primary_budget != 0 else 1)

    participant_effect_noise = rng.normal(0.0, scenario.participant_effect_sd, size=participants)
    participant_ablation_noise = rng.normal(0.0, scenario.participant_effect_sd, size=participants)

    for index in range(participants):
        participant = f"p{index + 1:03d}"
        for budget in budgets:
            is_primary = budget == primary_budget
            for lane_id, family, mean_effect, mean_ablation in lane_specs:
                if is_primary:
                    effect_center = mean_effect
                    ablation_center = mean_ablation
                else:
                    effect_center = float(scenario.secondary_budget_effect)
                    ablation_center = float(scenario.secondary_budget_ablation)
                lane_jitter = rng.normal(0.0, scenario.measurement_sd)
                ablation_jitter = rng.normal(0.0, scenario.measurement_sd)
                effect = effect_center + participant_effect_noise[index] + lane_jitter
                ablation = (
                    ablation_center + participant_ablation_noise[index] + ablation_jitter
                )
                baseline_control = 0.64 if lane_id == "raw" else 0.67
                candidate = baseline_control + effect
                rows.append(
                    ConfirmatoryRepresentationObservation(
                        participant_id=participant,
                        occasion_id="synthetic-session",
                        case_id=f"{participant}-synthetic-session",
                        calibration_per_class=budget,
                        representation_id=lane_id,
                        representation_family=family,
                        authority_fingerprint=_hex_identity(
                            f"authority:{participant}:synthetic-session:{budget}"
                        ),
                        representation_sha256=_hex_identity(f"representation:{lane_id}"),
                        source_fingerprint=_hex_identity(f"source:{participant}"),
                        candidate_metric=candidate,
                        primary_control_metric=baseline_control,
                        ablated_metric=candidate - ablation,
                        information_novel=scenario.information_novel,
                        model_id=None if lane_id == "raw" else "synthetic-latent",
                        model_revision=None if lane_id == "raw" else "known-truth-v1",
                    )
                )
    return tuple(rows)


def run_validation_replicate(
    scenario: BMRBValidationScenario,
    *,
    replicate: int,
    seed: int,
    participants: int = 8,
    primary_calibration_per_class: int = 10,
    bootstrap_resamples: int = 300,
) -> BMRBValidationReplicate:
    """Run one deterministic known-truth replicate through production BMRB semantics."""

    rng = np.random.default_rng(int(seed) + int(replicate) * 1009)
    policy = validation_policy(
        participants=participants,
        primary_calibration_per_class=primary_calibration_per_class,
        inference_seed=int(seed) + int(replicate) * 17,
        bootstrap_resamples=bootstrap_resamples,
    )
    observations = _observations_for_scenario(
        scenario,
        rng=rng,
        participants=participants,
        primary_budget=primary_calibration_per_class,
    )
    result = evaluate_confirmatory_representation(
        observations,
        study_id=f"validation-{scenario.scenario_id}-{replicate:04d}",
        mechanism_id=scenario.scenario_id,
        policy=policy,
    )
    reference = next(
        lane for lane in result.lanes if lane.representation_id == policy.reference_representation_id
    )
    if scenario.expected_failure_component is None:
        localized = result.scientific_criteria_passed
    else:
        localized = not _component_passed(result, scenario.expected_failure_component)
    return BMRBValidationReplicate(
        scenario_id=scenario.scenario_id,
        replicate=int(replicate),
        scientific_criteria_passed=result.scientific_criteria_passed,
        effect_criteria_passed=result.effect_criteria_passed,
        adversary_survival_passed=result.adversary_survival_passed,
        conservation_criteria_passed=result.conservation_criteria_passed,
        coverage_criteria_passed=result.coverage_criteria_passed,
        reference_observed_effect=reference.candidate.observed_mean,
        reference_effect_ci_lower=reference.candidate.bootstrap_ci_lower,
        reference_effect_ci_upper=reference.candidate.bootstrap_ci_upper,
        reference_effect_bias=reference.candidate.observed_mean - scenario.reference_effect,
        reference_ci_covers_truth=bool(
            reference.candidate.bootstrap_ci_lower
            <= scenario.reference_effect
            <= reference.candidate.bootstrap_ci_upper
        ),
        expected_failure_localized=bool(localized),
    )


def _summarize(
    scenario: BMRBValidationScenario,
    replicates: Iterable[BMRBValidationReplicate],
) -> BMRBValidationScenarioSummary:
    rows = tuple(replicates)
    if not rows:
        raise ValueError("validation scenario summary requires replicates")
    passes = np.asarray([row.scientific_criteria_passed for row in rows], dtype=float)
    expected = float(scenario.expected_scientific_pass)
    return BMRBValidationScenarioSummary(
        scenario_id=scenario.scenario_id,
        truth_class=scenario.truth_class,
        expected_scientific_pass=scenario.expected_scientific_pass,
        expected_failure_component=scenario.expected_failure_component,
        replicates=len(rows),
        observed_pass_rate=float(np.mean(passes)),
        decision_error_rate=float(np.mean(np.abs(passes - expected))),
        expected_failure_localization_rate=float(
            np.mean([row.expected_failure_localized for row in rows])
        ),
        mean_reference_effect_bias=float(np.mean([row.reference_effect_bias for row in rows])),
        reference_ci_coverage=float(np.mean([row.reference_ci_covers_truth for row in rows])),
    )


def validate_missing_pair_rejection(*, seed: int = 1901) -> bool:
    """Verify that a missing representation pair is rejected rather than silently pooled."""

    scenario = default_validation_scenarios()[2]
    policy = validation_policy(participants=4, inference_seed=seed)
    observations = list(
        _observations_for_scenario(
            scenario,
            rng=np.random.default_rng(seed),
            participants=4,
            primary_budget=policy.primary_calibration_per_class,
        )
    )
    observations.pop()
    try:
        evaluate_confirmatory_representation(
            observations,
            study_id="validation-missing-pair",
            mechanism_id="shared-mechanism-positive",
            policy=policy,
        )
    except ValueError as exc:
        return "exactly paired" in str(exc)
    return False


def run_bmrb_validation_suite(
    *,
    replicates: int = 20,
    seed: int = 1901,
    participants: int = 8,
    bootstrap_resamples: int = 300,
) -> dict[str, Any]:
    """Run the first known-ground-truth BMRB validation program."""

    if replicates < 1:
        raise ValueError("replicates must be positive")
    scenarios = default_validation_scenarios()
    all_replicates: list[BMRBValidationReplicate] = []
    summaries: list[BMRBValidationScenarioSummary] = []
    for scenario_index, scenario in enumerate(scenarios):
        rows = [
            run_validation_replicate(
                scenario,
                replicate=replicate,
                seed=seed + scenario_index * 100_003,
                participants=participants,
                bootstrap_resamples=bootstrap_resamples,
            )
            for replicate in range(replicates)
        ]
        all_replicates.extend(rows)
        summaries.append(_summarize(scenario, rows))

    missing_pair_rejected = validate_missing_pair_rejection(seed=seed + 700_001)
    effect_null = next(item for item in summaries if item.scenario_id == "effect-null")
    positive = next(item for item in summaries if item.scenario_id == "shared-mechanism-positive")
    calibration = next(item for item in summaries if item.scenario_id == "calibration-reversal")
    adversarial = [item for item in summaries if item.truth_class == "adversarial"]
    equivalence = next(item for item in summaries if item.scenario_id == "equivalence-null")

    qualification = {
        "effect_null_false_positive_rate_max": 0.10,
        "known_positive_power_min": 0.90,
        "adversarial_decision_error_rate_max": 0.10,
        "expected_failure_localization_min": 0.90,
        "calibration_reversal_power_min": 0.90,
        "missing_pair_must_reject": True,
    }
    qualified = bool(
        effect_null.observed_pass_rate <= qualification["effect_null_false_positive_rate_max"]
        and positive.observed_pass_rate >= qualification["known_positive_power_min"]
        and calibration.observed_pass_rate >= qualification["calibration_reversal_power_min"]
        and equivalence.decision_error_rate
        <= qualification["adversarial_decision_error_rate_max"]
        and all(
            item.decision_error_rate <= qualification["adversarial_decision_error_rate_max"]
            for item in adversarial
        )
        and all(
            item.expected_failure_localization_rate
            >= qualification["expected_failure_localization_min"]
            for item in summaries
        )
        and missing_pair_rejected
    )
    return {
        "schema_version": 1,
        "benchmark": BMRB_VALIDATION_BENCHMARK,
        "seed": int(seed),
        "replicates_per_scenario": int(replicates),
        "participants_per_replicate": int(participants),
        "bootstrap_resamples": int(bootstrap_resamples),
        "qualification_policy": qualification,
        "qualified": qualified,
        "missing_pair_rejected": missing_pair_rejected,
        "scenario_summaries": [summary.to_mapping() for summary in summaries],
        "replicates": [row.to_mapping() for row in all_replicates],
        "interpretation": (
            "Known-truth simulation validates BMRB decision behavior and failure localization. "
            "It does not validate biological truth or authorize physical-quantum claims."
        ),
    }
