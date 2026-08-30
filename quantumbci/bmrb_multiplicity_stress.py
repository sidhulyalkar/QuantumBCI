"""Known-truth candidate-search stress for BMRB multiplicity authority.

The stress deliberately uses the production ``run_validation_replicate`` path. Each candidate is
a near-boundary known null: its declared mean candidate effect is below the frozen BMRB effect
threshold, but participant heterogeneity means an individual simulated candidate can occasionally
survive all scientific gates by chance.

A naive search rule asks whether *any* searched candidate survived. The v1 multiplicity authority
asks only whether the one predeclared primary candidate survived. The contrast estimates the
winner-picking amplification created by increasing the searched family without silently turning
participant p-values into a new promotion rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .bmrb_multiplicity import apply_multiplicity_plan, winner_picking_demo_plan
from .bmrb_validation import BMRBValidationScenario, run_validation_replicate

BMRB_MULTIPLICITY_STRESS_BENCHMARK = "BMRB_CANDIDATE_SEARCH_MULTIPLICITY_STRESS_V1"


@dataclass(frozen=True)
class BMRBMultiplicityStressReplicate:
    replicate: int
    candidate_count: int
    candidate_pass_count: int
    primary_passed: bool
    naive_any_survivor: bool
    authorized_any_promotion: bool
    suppressed_nonprimary_survivors: int

    def to_mapping(self) -> dict[str, Any]:
        return {
            "replicate": self.replicate,
            "candidate_count": self.candidate_count,
            "candidate_pass_count": self.candidate_pass_count,
            "primary_passed": self.primary_passed,
            "naive_any_survivor": self.naive_any_survivor,
            "authorized_any_promotion": self.authorized_any_promotion,
            "suppressed_nonprimary_survivors": self.suppressed_nonprimary_survivors,
        }


def near_boundary_null_scenario() -> BMRBValidationScenario:
    """Return a known-null DGM just below the frozen software-validation effect threshold.

    The validation policy requires a candidate advantage of at least 0.05. This DGM is centered at
    0.049 in both exactly paired lanes, so its scientific truth remains on the null side of that
    declared boundary. Moderate participant heterogeneity creates realistic decision uncertainty
    for multiplicity stress without changing the threshold itself.
    """

    return BMRBValidationScenario(
        scenario_id="multiplicity-near-boundary-null",
        truth_class="known_null",
        expected_scientific_pass=False,
        expected_failure_component="effect",
        reference_effect=0.049,
        alternate_effect=0.049,
        reference_ablation=0.10,
        alternate_ablation=0.10,
        information_novel=True,
        participant_effect_sd=0.05,
        measurement_sd=0.005,
    )


def run_bmrb_multiplicity_stress(
    *,
    family_replicates: int = 40,
    candidate_count: int = 20,
    participants: int = 4,
    bootstrap_resamples: int = 100,
    seed: int = 5901,
) -> dict[str, Any]:
    """Estimate winner-picking amplification under one frozen candidate-family authority."""

    if int(family_replicates) < 1:
        raise ValueError("family_replicates must be positive")
    if int(candidate_count) < 2:
        raise ValueError("candidate_count must be at least two")
    if int(participants) < 4:
        raise ValueError("participants must be at least four")
    if int(bootstrap_resamples) < 100:
        raise ValueError("bootstrap_resamples must be at least 100")

    family_replicates = int(family_replicates)
    candidate_count = int(candidate_count)
    participants = int(participants)
    bootstrap_resamples = int(bootstrap_resamples)
    seed = int(seed)

    plan = winner_picking_demo_plan(exploratory_candidates=candidate_count - 1)
    scenario = near_boundary_null_scenario()
    rows: list[BMRBMultiplicityStressReplicate] = []

    for family_replicate in range(family_replicates):
        scientific_results: dict[str, bool] = {}
        for candidate_index, candidate_id in enumerate(plan.candidate_ids):
            # Candidate-specific offsets are deterministic and deliberately far apart. This is
            # development simulation authority, not final BMRB evaluation seed authority.
            candidate_seed = seed + family_replicate * 1_000_003 + candidate_index * 100_003
            result = run_validation_replicate(
                scenario,
                replicate=0,
                seed=candidate_seed,
                participants=participants,
                bootstrap_resamples=bootstrap_resamples,
            )
            scientific_results[candidate_id] = result.scientific_criteria_passed

        decision = apply_multiplicity_plan(plan, scientific_results)
        primary = next(
            candidate for candidate in decision.candidates if candidate.role == "primary"
        )
        rows.append(
            BMRBMultiplicityStressReplicate(
                replicate=family_replicate,
                candidate_count=candidate_count,
                candidate_pass_count=int(sum(scientific_results.values())),
                primary_passed=primary.scientific_criteria_passed,
                naive_any_survivor=decision.naive_any_survivor,
                authorized_any_promotion=decision.authorized_any_promotion,
                suppressed_nonprimary_survivors=len(
                    decision.suppressed_nonprimary_survivors
                ),
            )
        )

    naive = np.asarray([row.naive_any_survivor for row in rows], dtype=float)
    authorized = np.asarray([row.authorized_any_promotion for row in rows], dtype=float)
    candidate_passes = np.asarray([row.candidate_pass_count for row in rows], dtype=float)
    suppressed = np.asarray(
        [row.suppressed_nonprimary_survivors for row in rows], dtype=float
    )
    naive_rate = float(np.mean(naive))
    authorized_rate = float(np.mean(authorized))

    return {
        "schema_version": 1,
        "benchmark": BMRB_MULTIPLICITY_STRESS_BENCHMARK,
        "scenario": {
            "scenario_id": scenario.scenario_id,
            "truth_class": scenario.truth_class,
            "reference_effect": scenario.reference_effect,
            "alternate_effect": scenario.alternate_effect,
            "validation_effect_threshold": 0.05,
            "participant_effect_sd": scenario.participant_effect_sd,
            "measurement_sd": scenario.measurement_sd,
        },
        "seed": seed,
        "family_replicates": family_replicates,
        "candidate_count": candidate_count,
        "participants": participants,
        "bootstrap_resamples": bootstrap_resamples,
        "plan_fingerprint": plan.plan_fingerprint,
        "naive_any_survivor_rate": naive_rate,
        "authorized_primary_promotion_rate": authorized_rate,
        "winner_picking_amplification": float(naive_rate - authorized_rate),
        "mean_candidate_survivors_per_family": float(np.mean(candidate_passes)),
        "mean_suppressed_nonprimary_survivors": float(np.mean(suppressed)),
        "replicates": [row.to_mapping() for row in rows],
        "interpretation": (
            "The searched candidates are known-null relative to the declared 0.05 effect boundary. "
            "A naive any-survivor rule can amplify false promotion across the search family, while "
            "the primary-only authority remains tied to the candidate frozen before inspection."
        ),
        "claim_boundary": (
            "This synthetic multiplicity stress estimates decision behavior under a declared DGM. "
            "It does not validate biological truth or authorize a physical-quantum claim."
        ),
    }
