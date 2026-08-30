"""Block-correlated candidate-search stress for BMRB multiplicity authority.

Candidate labels are not necessarily independent scientific opportunities. Adjacent layers,
nearby hyperparameters, preprocessing variants, or closely related representations can share most
of their evidence. This module therefore varies the number of genuinely independent latent draws
under a fixed number of searched candidate labels.

Candidates assigned to the same latent draw reuse the exact production-validation seed and are
perfectly correlated. The independent-draw sets are nested, so increasing effective search
opportunities cannot change the frozen primary candidate and cannot decrease a naive any-survivor
headline within a replicate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .bmrb_multiplicity import apply_multiplicity_plan, winner_picking_demo_plan
from .bmrb_multiplicity_stress import near_boundary_null_scenario
from .bmrb_validation import run_validation_replicate

BMRB_CORRELATED_MULTIPLICITY_BENCHMARK = (
    "BMRB_BLOCK_CORRELATED_CANDIDATE_SEARCH_STRESS_V1"
)


@dataclass(frozen=True)
class CorrelatedSearchCondition:
    independent_draws: int
    candidate_count: int
    naive_any_survivor_rate: float
    authorized_primary_promotion_rate: float
    winner_picking_amplification: float
    mean_candidate_survivors: float
    mean_unique_surviving_draws: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "independent_draws": self.independent_draws,
            "candidate_count": self.candidate_count,
            "naive_any_survivor_rate": self.naive_any_survivor_rate,
            "authorized_primary_promotion_rate": self.authorized_primary_promotion_rate,
            "winner_picking_amplification": self.winner_picking_amplification,
            "mean_candidate_survivors": self.mean_candidate_survivors,
            "mean_unique_surviving_draws": self.mean_unique_surviving_draws,
        }


def _validated_draw_counts(
    values: Iterable[int],
    *,
    candidate_count: int,
) -> tuple[int, ...]:
    counts = tuple(int(value) for value in values)
    if not counts:
        raise ValueError("independent_draw_counts must not be empty")
    if tuple(sorted(set(counts))) != counts:
        raise ValueError("independent_draw_counts must be unique and strictly increasing")
    if counts[0] != 1:
        raise ValueError("independent_draw_counts must start at one")
    if counts[-1] != candidate_count:
        raise ValueError("independent_draw_counts must end at candidate_count")
    if any(value < 1 or value > candidate_count for value in counts):
        raise ValueError("independent draw counts must lie in [1, candidate_count]")
    return counts


def candidate_draw_assignment(
    *,
    candidate_count: int,
    independent_draws: int,
) -> tuple[int, ...]:
    """Assign candidate labels to nested latent draws.

    The first ``independent_draws`` candidates introduce draws 0..k-1. Remaining labels repeat
    those draws cyclically. This makes every lower-k draw set a strict subset of every higher-k
    set while preserving candidate zero as the frozen primary draw.
    """

    candidate_count = int(candidate_count)
    independent_draws = int(independent_draws)
    if candidate_count < 2:
        raise ValueError("candidate_count must be at least two")
    if not 1 <= independent_draws <= candidate_count:
        raise ValueError("independent_draws must lie in [1, candidate_count]")
    return tuple(index % independent_draws for index in range(candidate_count))


def run_bmrb_correlated_multiplicity_stress(
    *,
    family_replicates: int = 8,
    candidate_count: int = 20,
    independent_draw_counts: tuple[int, ...] = (1, 2, 5, 20),
    participants: int = 4,
    bootstrap_resamples: int = 100,
    seed: int = 5901,
) -> dict[str, Any]:
    """Measure winner-picking as effective independent search opportunities increase."""

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
    draw_counts = _validated_draw_counts(
        independent_draw_counts,
        candidate_count=candidate_count,
    )

    plan = winner_picking_demo_plan(exploratory_candidates=candidate_count - 1)
    scenario = near_boundary_null_scenario()
    condition_rows: list[CorrelatedSearchCondition] = []
    primary_vectors: list[tuple[bool, ...]] = []
    naive_vectors: list[tuple[bool, ...]] = []

    for independent_draws in draw_counts:
        assignment = candidate_draw_assignment(
            candidate_count=candidate_count,
            independent_draws=independent_draws,
        )
        replicate_primary: list[bool] = []
        replicate_naive: list[bool] = []
        candidate_survivor_counts: list[int] = []
        unique_surviving_draw_counts: list[int] = []

        for family_replicate in range(family_replicates):
            draw_results: dict[int, bool] = {}
            for draw_index in range(independent_draws):
                draw_seed = seed + family_replicate * 1_000_003 + draw_index * 100_003
                result = run_validation_replicate(
                    scenario,
                    replicate=0,
                    seed=draw_seed,
                    participants=participants,
                    bootstrap_resamples=bootstrap_resamples,
                )
                draw_results[draw_index] = result.scientific_criteria_passed

            scientific_results = {
                candidate_id: draw_results[assignment[index]]
                for index, candidate_id in enumerate(plan.candidate_ids)
            }
            decision = apply_multiplicity_plan(plan, scientific_results)
            primary = decision.candidates[0]
            replicate_primary.append(primary.scientific_criteria_passed)
            replicate_naive.append(decision.naive_any_survivor)
            candidate_survivor_counts.append(int(sum(scientific_results.values())))
            unique_surviving_draw_counts.append(int(sum(draw_results.values())))

        primary_vector = tuple(replicate_primary)
        naive_vector = tuple(replicate_naive)
        primary_vectors.append(primary_vector)
        naive_vectors.append(naive_vector)
        naive_rate = float(np.mean(np.asarray(replicate_naive, dtype=float)))
        primary_rate = float(np.mean(np.asarray(replicate_primary, dtype=float)))
        condition_rows.append(
            CorrelatedSearchCondition(
                independent_draws=independent_draws,
                candidate_count=candidate_count,
                naive_any_survivor_rate=naive_rate,
                authorized_primary_promotion_rate=primary_rate,
                winner_picking_amplification=float(naive_rate - primary_rate),
                mean_candidate_survivors=float(np.mean(candidate_survivor_counts)),
                mean_unique_surviving_draws=float(np.mean(unique_surviving_draw_counts)),
            )
        )

    primary_invariant = all(vector == primary_vectors[0] for vector in primary_vectors[1:])
    naive_nested = all(
        all((not earlier) or later for earlier, later in zip(left, right, strict=True))
        for left, right in zip(naive_vectors, naive_vectors[1:], strict=True)
    )
    one_draw_equals_primary = naive_vectors[0] == primary_vectors[0]

    return {
        "schema_version": 1,
        "benchmark": BMRB_CORRELATED_MULTIPLICITY_BENCHMARK,
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
        "independent_draw_counts": list(draw_counts),
        "participants": participants,
        "bootstrap_resamples": bootstrap_resamples,
        "plan_fingerprint": plan.plan_fingerprint,
        "primary_result_invariant": primary_invariant,
        "naive_survival_nested": naive_nested,
        "one_draw_equals_primary": one_draw_equals_primary,
        "conditions": [condition.to_mapping() for condition in condition_rows],
        "interpretation": (
            "Candidate labels can be redundant. Under one effective latent draw, all searched "
            "labels reproduce the primary result exactly. As nested independent draws are added, "
            "a naive any-survivor headline can only stay flat or increase, while the predeclared "
            "primary candidate remains unchanged."
        ),
        "claim_boundary": (
            "This block-correlation stress validates multiplicity decision behavior under a "
            "synthetic known-null search. It does not validate biological truth, establish neural "
            "causal necessity, or authorize a physical-quantum claim."
        ),
    }
