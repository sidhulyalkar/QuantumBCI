"""Extended known-truth stress tests for BMRB scientific decision semantics.

The core validation module establishes a compact qualification grid. This module adds
harder operating-characteristic attacks that are useful for a methods study: seductive
weak decision rules, invertible coordinate changes, participant heterogeneity, and
repeated noisy sessions.

The production confirmatory evaluator remains the decision authority. The deliberately
naive rules are negative controls that help demonstrate what the extra BMRB gates protect
against; they are not proposed scientific baselines.
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

BMRB_VALIDATION_STRESS_BENCHMARK = "BMRB_KNOWN_TRUTH_STRESS_V1"


@dataclass(frozen=True)
class BMRBStressScenario:
    scenario_id: str
    expected_bmrb_pass: bool
    reference_effect: float
    alternate_effect: float
    reference_ablation: float
    alternate_ablation: float
    information_novel: bool
    participant_effect_sd: float = 0.015
    measurement_sd: float = 0.005
    occasions_per_participant: int = 1
    alternate_family: str = "synthetic_representation"
    secondary_budget_effect: float | None = None
    secondary_budget_ablation: float | None = None

    def __post_init__(self) -> None:
        if not self.scenario_id.strip():
            raise ValueError("scenario_id must not be empty")
        if not self.alternate_family.strip():
            raise ValueError("alternate_family must not be empty")
        if int(self.occasions_per_participant) < 1:
            raise ValueError("occasions_per_participant must be positive")
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
        secondary = (self.secondary_budget_effect, self.secondary_budget_ablation)
        if any(value is None for value in secondary) and not all(
            value is None for value in secondary
        ):
            raise ValueError("secondary budget effect and ablation must be supplied together")
        if any(
            value is not None and not np.isfinite(float(value))
            for value in secondary
        ):
            raise ValueError("secondary budget values must be finite")


@dataclass(frozen=True)
class BMRBStressReplicate:
    scenario_id: str
    replicate: int
    bmrb_scientific_passed: bool
    naive_primary_effect_passed: bool
    naive_budget_averaged_effect_passed: bool
    participant_count: int
    occasions_per_participant: int

    def to_mapping(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "replicate": self.replicate,
            "bmrb_scientific_passed": self.bmrb_scientific_passed,
            "naive_primary_effect_passed": self.naive_primary_effect_passed,
            "naive_budget_averaged_effect_passed": self.naive_budget_averaged_effect_passed,
            "participant_count": self.participant_count,
            "occasions_per_participant": self.occasions_per_participant,
        }


@dataclass(frozen=True)
class BMRBStressSummary:
    scenario_id: str
    expected_bmrb_pass: bool
    replicates: int
    bmrb_pass_rate: float
    naive_primary_effect_pass_rate: float
    naive_budget_averaged_effect_pass_rate: float

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
            "naive_primary_effect_pass_rate": self.naive_primary_effect_pass_rate,
            "naive_budget_averaged_effect_pass_rate": self.naive_budget_averaged_effect_pass_rate,
        }


def default_stress_scenarios() -> tuple[BMRBStressScenario, ...]:
    """Return operating-characteristic attacks beyond the compact core grid."""

    return (
        BMRBStressScenario(
            scenario_id="equivalence-null-naive-trap",
            expected_bmrb_pass=False,
            reference_effect=0.12,
            alternate_effect=0.12,
            reference_ablation=0.10,
            alternate_ablation=0.10,
            information_novel=False,
        ),
        BMRBStressScenario(
            scenario_id="predictive-shortcut-naive-trap",
            expected_bmrb_pass=False,
            reference_effect=0.12,
            alternate_effect=0.115,
            reference_ablation=0.0,
            alternate_ablation=0.0,
            information_novel=True,
        ),
        BMRBStressScenario(
            scenario_id="calibration-reversal-naive-trap",
            expected_bmrb_pass=True,
            reference_effect=0.12,
            alternate_effect=0.115,
            reference_ablation=0.10,
            alternate_ablation=0.095,
            information_novel=True,
            secondary_budget_effect=-0.08,
            secondary_budget_ablation=-0.05,
        ),
        BMRBStressScenario(
            scenario_id="invertible-coordinate-positive",
            expected_bmrb_pass=True,
            reference_effect=0.12,
            alternate_effect=0.12,
            reference_ablation=0.10,
            alternate_ablation=0.10,
            information_novel=True,
            alternate_family="invertible_coordinate",
        ),
        BMRBStressScenario(
            scenario_id="heterogeneous-shared-positive",
            expected_bmrb_pass=True,
            reference_effect=0.15,
            alternate_effect=0.145,
            reference_ablation=0.12,
            alternate_ablation=0.115,
            information_novel=True,
            participant_effect_sd=0.035,
            measurement_sd=0.01,
        ),
        BMRBStressScenario(
            scenario_id="noisy-repeated-sessions-positive",
            expected_bmrb_pass=True,
            reference_effect=0.17,
            alternate_effect=0.165,
            reference_ablation=0.14,
            alternate_ablation=0.135,
            information_novel=True,
            participant_effect_sd=0.02,
            measurement_sd=0.04,
            occasions_per_participant=3,
        ),
    )


def _hex_identity(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _observations(
    scenario: BMRBStressScenario,
    *,
    rng: np.random.Generator,
    participants: int,
    primary_budget: int,
) -> tuple[ConfirmatoryRepresentationObservation, ...]:
    rows: list[ConfirmatoryRepresentationObservation] = []
    budgets = [primary_budget]
    if scenario.secondary_budget_effect is not None:
        budgets.insert(0, 0 if primary_budget != 0 else 1)

    participant_effect_noise = rng.normal(
        0.0, scenario.participant_effect_sd, size=participants
    )
    participant_ablation_noise = rng.normal(
        0.0, scenario.participant_effect_sd, size=participants
    )
    lane_specs = (
        ("raw", "raw_neural", scenario.reference_effect, scenario.reference_ablation),
        (
            "latent",
            scenario.alternate_family,
            scenario.alternate_effect,
            scenario.alternate_ablation,
        ),
    )

    for participant_index in range(participants):
        participant = f"p{participant_index + 1:03d}"
        for occasion_index in range(scenario.occasions_per_participant):
            occasion = f"synthetic-session-{occasion_index + 1:02d}"
            for budget in budgets:
                is_primary = budget == primary_budget
                for lane_id, family, mean_effect, mean_ablation in lane_specs:
                    if is_primary:
                        effect_center = mean_effect
                        ablation_center = mean_ablation
                    else:
                        effect_center = float(scenario.secondary_budget_effect)
                        ablation_center = float(scenario.secondary_budget_ablation)
                    effect = (
                        effect_center
                        + participant_effect_noise[participant_index]
                        + rng.normal(0.0, scenario.measurement_sd)
                    )
                    ablation = (
                        ablation_center
                        + participant_ablation_noise[participant_index]
                        + rng.normal(0.0, scenario.measurement_sd)
                    )
                    control = 0.64 if lane_id == "raw" else 0.67
                    candidate = control + effect
                    rows.append(
                        ConfirmatoryRepresentationObservation(
                            participant_id=participant,
                            occasion_id=occasion,
                            case_id=f"{participant}-{occasion}",
                            calibration_per_class=budget,
                            representation_id=lane_id,
                            representation_family=family,
                            authority_fingerprint=_hex_identity(
                                f"authority:{participant}:{occasion}:{budget}"
                            ),
                            representation_sha256=_hex_identity(
                                f"representation:{scenario.scenario_id}:{lane_id}"
                            ),
                            source_fingerprint=_hex_identity(
                                f"source:{scenario.scenario_id}:{participant}:{occasion}"
                            ),
                            candidate_metric=candidate,
                            primary_control_metric=control,
                            ablated_metric=candidate - ablation,
                            information_novel=scenario.information_novel,
                            model_id=None if lane_id == "raw" else "synthetic-latent",
                            model_revision=None if lane_id == "raw" else "known-truth-stress-v1",
                        )
                    )
    return tuple(rows)


def _naive_effect_pass(
    observations: Iterable[ConfirmatoryRepresentationObservation],
    *,
    threshold: float,
    primary_budget: int | None,
) -> bool:
    materialized = tuple(observations)
    if primary_budget is not None:
        materialized = tuple(
            item
            for item in materialized
            if item.calibration_per_class == primary_budget
        )
    if not materialized:
        raise ValueError("naive effect rule requires observations")
    return bool(
        np.mean([item.candidate_advantage for item in materialized]) >= threshold
    )


def run_stress_replicate(
    scenario: BMRBStressScenario,
    *,
    replicate: int,
    seed: int,
    participants: int = 8,
    primary_calibration_per_class: int = 10,
    bootstrap_resamples: int = 100,
) -> BMRBStressReplicate:
    """Run one stress replicate through BMRB and deliberately weaker rules."""

    policy = validation_policy(
        participants=participants,
        primary_calibration_per_class=primary_calibration_per_class,
        inference_seed=int(seed) + int(replicate) * 17,
        bootstrap_resamples=bootstrap_resamples,
    )
    rng = np.random.default_rng(int(seed) + int(replicate) * 1009)
    observations = _observations(
        scenario,
        rng=rng,
        participants=participants,
        primary_budget=primary_calibration_per_class,
    )
    result = evaluate_confirmatory_representation(
        observations,
        study_id=f"stress-{scenario.scenario_id}-{replicate:04d}",
        mechanism_id=scenario.scenario_id,
        policy=policy,
    )
    return BMRBStressReplicate(
        scenario_id=scenario.scenario_id,
        replicate=int(replicate),
        bmrb_scientific_passed=result.scientific_criteria_passed,
        naive_primary_effect_passed=_naive_effect_pass(
            observations,
            threshold=policy.min_candidate_advantage,
            primary_budget=policy.primary_calibration_per_class,
        ),
        naive_budget_averaged_effect_passed=_naive_effect_pass(
            observations,
            threshold=policy.min_candidate_advantage,
            primary_budget=None,
        ),
        participant_count=result.participant_count,
        occasions_per_participant=scenario.occasions_per_participant,
    )


def _summary(
    scenario: BMRBStressScenario,
    rows: Iterable[BMRBStressReplicate],
) -> BMRBStressSummary:
    materialized = tuple(rows)
    if not materialized:
        raise ValueError("stress summary requires replicates")
    return BMRBStressSummary(
        scenario_id=scenario.scenario_id,
        expected_bmrb_pass=scenario.expected_bmrb_pass,
        replicates=len(materialized),
        bmrb_pass_rate=float(
            np.mean([row.bmrb_scientific_passed for row in materialized])
        ),
        naive_primary_effect_pass_rate=float(
            np.mean([row.naive_primary_effect_passed for row in materialized])
        ),
        naive_budget_averaged_effect_pass_rate=float(
            np.mean(
                [
                    row.naive_budget_averaged_effect_passed
                    for row in materialized
                ]
            )
        ),
    )


def run_bmrb_validation_stress_suite(
    *,
    replicates: int = 4,
    seed: int = 2901,
    participants: int = 8,
    bootstrap_resamples: int = 100,
) -> dict[str, Any]:
    """Run extended known-truth attacks and compare against weak negative controls."""

    if replicates < 1:
        raise ValueError("replicates must be positive")
    scenarios = default_stress_scenarios()
    summaries: list[BMRBStressSummary] = []
    all_rows: list[BMRBStressReplicate] = []
    for scenario_index, scenario in enumerate(scenarios):
        rows = [
            run_stress_replicate(
                scenario,
                replicate=replicate,
                seed=seed + scenario_index * 100_003,
                participants=participants,
                bootstrap_resamples=bootstrap_resamples,
            )
            for replicate in range(replicates)
        ]
        all_rows.extend(rows)
        summaries.append(_summary(scenario, rows))

    by_id = {summary.scenario_id: summary for summary in summaries}
    equivalence = by_id["equivalence-null-naive-trap"]
    shortcut = by_id["predictive-shortcut-naive-trap"]
    reversal = by_id["calibration-reversal-naive-trap"]
    positive_ids = (
        "invertible-coordinate-positive",
        "heterogeneous-shared-positive",
        "noisy-repeated-sessions-positive",
    )

    qualification = {
        "bmrb_decision_error_rate_max": 0.10,
        "naive_adversary_false_accept_rate_min": 0.90,
        "naive_budget_averaging_false_reject_rate_min": 0.90,
        "positive_stress_recovery_min": 0.90,
    }
    qualified = bool(
        all(
            summary.bmrb_decision_error_rate
            <= qualification["bmrb_decision_error_rate_max"]
            for summary in summaries
        )
        and equivalence.naive_primary_effect_pass_rate
        >= qualification["naive_adversary_false_accept_rate_min"]
        and shortcut.naive_primary_effect_pass_rate
        >= qualification["naive_adversary_false_accept_rate_min"]
        and (1.0 - reversal.naive_budget_averaged_effect_pass_rate)
        >= qualification["naive_budget_averaging_false_reject_rate_min"]
        and all(
            by_id[scenario_id].bmrb_pass_rate
            >= qualification["positive_stress_recovery_min"]
            for scenario_id in positive_ids
        )
    )
    return {
        "schema_version": 1,
        "benchmark": BMRB_VALIDATION_STRESS_BENCHMARK,
        "seed": int(seed),
        "replicates_per_scenario": int(replicates),
        "participants_per_replicate": int(participants),
        "bootstrap_resamples": int(bootstrap_resamples),
        "qualification_policy": qualification,
        "qualified": qualified,
        "scenario_summaries": [summary.to_mapping() for summary in summaries],
        "replicates": [row.to_mapping() for row in all_rows],
        "interpretation": (
            "The weak rules are diagnostic negative controls. Passing this stress suite "
            "supports the implemented benchmark semantics on declared synthetic evidence, "
            "not biological truth or a physical-quantum claim."
        ),
    }
