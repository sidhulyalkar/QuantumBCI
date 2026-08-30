"""Known-null stress for outcome-dependent BMRB candidate search.

The full candidate universe is generated before the adaptive transcript is evaluated. Every
candidate therefore remains available to the closed-world multiplicity authority even when a naive
adaptive analyst would stop inspecting after the first survivor.

This isolates optional stopping and outcome-routed inspection from missing-result reporting. It
does not authorize incomplete confirmatory evidence and does not alter BMRB's scientific gates.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .bmrb_adaptive_search import BMRBAdaptiveSearchPlan, run_adaptive_search
from .bmrb_multiplicity import winner_picking_demo_plan
from .bmrb_multiplicity_stress import near_boundary_null_scenario
from .bmrb_validation import BMRBValidationReplicate, run_validation_replicate

BMRB_ADAPTIVE_SEARCH_STRESS_BENCHMARK = "BMRB_ADAPTIVE_CANDIDATE_SEARCH_STRESS_V1"


def default_adaptive_search_plan(*, candidate_count: int = 20) -> BMRBAdaptiveSearchPlan:
    candidate_count = int(candidate_count)
    if candidate_count < 2:
        raise ValueError("candidate_count must be at least two")
    multiplicity_plan = winner_picking_demo_plan(
        exploratory_candidates=candidate_count - 1
    )
    return BMRBAdaptiveSearchPlan(
        plan_id="known-null-outcome-routed-search-v1",
        multiplicity_plan=multiplicity_plan,
        max_evaluations=candidate_count,
        routing_effect_cutoff=0.05,
        above_cutoff_stride=1,
        below_cutoff_stride=3,
        scientific_rationale=(
            "Freeze an outcome-dependent inspection trajectory and optional stopping rule while "
            "retaining complete closed-world evidence for confirmatory multiplicity authority."
        ),
    )


def _candidate_evidence(
    *,
    family_replicate: int,
    candidate_count: int,
    participants: int,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, BMRBValidationReplicate]:
    plan = winner_picking_demo_plan(exploratory_candidates=candidate_count - 1)
    scenario = near_boundary_null_scenario()
    return {
        candidate_id: run_validation_replicate(
            scenario,
            replicate=0,
            seed=seed + family_replicate * 1_000_003 + candidate_index * 100_003,
            participants=participants,
            bootstrap_resamples=bootstrap_resamples,
        )
        for candidate_index, candidate_id in enumerate(plan.candidate_ids)
    }


def run_bmrb_adaptive_search_stress(
    *,
    family_replicates: int = 8,
    candidate_count: int = 20,
    participants: int = 4,
    bootstrap_resamples: int = 100,
    seed: int = 5901,
) -> dict[str, Any]:
    """Estimate optional-stopping amplification under one frozen adaptive search plan."""

    family_replicates = int(family_replicates)
    candidate_count = int(candidate_count)
    participants = int(participants)
    bootstrap_resamples = int(bootstrap_resamples)
    seed = int(seed)
    if family_replicates < 1:
        raise ValueError("family_replicates must be positive")
    if candidate_count < 2:
        raise ValueError("candidate_count must be at least two")
    if participants < 4:
        raise ValueError("participants must be at least four")
    if bootstrap_resamples < 100:
        raise ValueError("bootstrap_resamples must be at least 100")

    plan = default_adaptive_search_plan(candidate_count=candidate_count)
    scenario = near_boundary_null_scenario()
    transcripts = []
    for family_replicate in range(family_replicates):
        evidence = _candidate_evidence(
            family_replicate=family_replicate,
            candidate_count=candidate_count,
            participants=participants,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        )
        transcripts.append(run_adaptive_search(plan, evidence))

    adaptive = np.asarray(
        [transcript.naive_adaptive_survivor for transcript in transcripts],
        dtype=float,
    )
    authorized = np.asarray(
        [transcript.authorized_primary_promotion for transcript in transcripts],
        dtype=float,
    )
    exhaustive = np.asarray(
        [transcript.exhaustive_any_survivor for transcript in transcripts],
        dtype=float,
    )
    evaluations = np.asarray(
        [len(transcript.steps) for transcript in transcripts],
        dtype=float,
    )
    early_stop = np.asarray(
        [
            transcript.naive_adaptive_survivor
            and len(transcript.steps) < plan.max_evaluations
            for transcript in transcripts
        ],
        dtype=float,
    )
    nonprimary_stop = np.asarray(
        [
            transcript.naive_adaptive_survivor
            and not transcript.authorized_primary_promotion
            for transcript in transcripts
        ],
        dtype=float,
    )
    adaptive_rate = float(np.mean(adaptive))
    authorized_rate = float(np.mean(authorized))
    exhaustive_rate = float(np.mean(exhaustive))

    return {
        "schema_version": 1,
        "benchmark": BMRB_ADAPTIVE_SEARCH_STRESS_BENCHMARK,
        "scenario": {
            "scenario_id": scenario.scenario_id,
            "truth_class": scenario.truth_class,
            "reference_effect": scenario.reference_effect,
            "alternate_effect": scenario.alternate_effect,
            "validation_effect_threshold": 0.05,
        },
        "seed": seed,
        "family_replicates": family_replicates,
        "candidate_count": candidate_count,
        "participants": participants,
        "bootstrap_resamples": bootstrap_resamples,
        "adaptive_plan": plan.to_mapping(),
        "adaptive_any_survivor_rate": adaptive_rate,
        "exhaustive_any_survivor_rate": exhaustive_rate,
        "authorized_primary_promotion_rate": authorized_rate,
        "adaptive_winner_picking_amplification": float(
            adaptive_rate - authorized_rate
        ),
        "nonprimary_adaptive_survivor_rate": float(np.mean(nonprimary_stop)),
        "early_stop_rate": float(np.mean(early_stop)),
        "mean_evaluations_used": float(np.mean(evaluations)),
        "adaptive_matches_exhaustive_with_full_budget": bool(
            np.array_equal(adaptive, exhaustive)
        ),
        "primary_authority_never_transferred": all(
            not transcript.naive_adaptive_survivor
            or transcript.authorized_primary_promotion
            or transcript.to_mapping()["first_adaptive_survivor"]
            != plan.multiplicity_plan.primary_candidate_id
            for transcript in transcripts
        ),
        "transcripts": [transcript.to_mapping() for transcript in transcripts],
        "interpretation": (
            "With a full frozen candidate universe, outcome-dependent routing and stop-on-survivor "
            "inspection recover the same naive any-survivor headline as exhaustive search. The "
            "closed multiplicity authority still restricts promotion to the predeclared primary."
        ),
        "claim_boundary": (
            "This adaptive-search stress validates decision and reporting behavior under synthetic "
            "known-null evidence. It does not validate biological truth, establish neural causal "
            "necessity, or authorize a physical-quantum claim."
        ),
    }
