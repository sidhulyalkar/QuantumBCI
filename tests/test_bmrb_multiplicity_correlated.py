from __future__ import annotations

import pytest

from quantumbci.bmrb_multiplicity_correlated import (
    BMRB_CORRELATED_MULTIPLICITY_BENCHMARK,
    candidate_draw_assignment,
    run_bmrb_correlated_multiplicity_stress,
)


def test_candidate_draw_assignment_preserves_primary_and_effective_draw_count() -> None:
    assert candidate_draw_assignment(candidate_count=6, independent_draws=1) == (0, 0, 0, 0, 0, 0)
    assert candidate_draw_assignment(candidate_count=6, independent_draws=2) == (0, 1, 0, 1, 0, 1)
    assert candidate_draw_assignment(candidate_count=6, independent_draws=6) == (0, 1, 2, 3, 4, 5)


def test_correlated_search_surface_preserves_primary_and_exposes_search_inflation() -> None:
    result = run_bmrb_correlated_multiplicity_stress(
        family_replicates=8,
        candidate_count=20,
        independent_draw_counts=(1, 2, 5, 20),
        participants=4,
        bootstrap_resamples=100,
        seed=5901,
    )

    assert result["benchmark"] == BMRB_CORRELATED_MULTIPLICITY_BENCHMARK
    assert result["scenario"]["reference_effect"] < result["scenario"][
        "validation_effect_threshold"
    ]
    assert result["primary_result_invariant"] is True
    assert result["naive_survival_nested"] is True
    assert result["one_draw_equals_primary"] is True

    conditions = {item["independent_draws"]: item for item in result["conditions"]}
    assert conditions[1]["naive_any_survivor_rate"] == conditions[1][
        "authorized_primary_promotion_rate"
    ]
    assert conditions[1]["winner_picking_amplification"] == 0.0
    assert conditions[20]["naive_any_survivor_rate"] > conditions[20][
        "authorized_primary_promotion_rate"
    ]
    assert conditions[20]["winner_picking_amplification"] > 0.0

    primary_rates = {
        condition["authorized_primary_promotion_rate"] for condition in result["conditions"]
    }
    assert len(primary_rates) == 1

    naive_rates = [condition["naive_any_survivor_rate"] for condition in result["conditions"]]
    assert naive_rates == sorted(naive_rates)


def test_more_candidate_labels_do_not_create_more_independent_draws() -> None:
    result = run_bmrb_correlated_multiplicity_stress(
        family_replicates=4,
        candidate_count=8,
        independent_draw_counts=(1, 2, 4, 8),
        participants=4,
        bootstrap_resamples=100,
        seed=5901,
    )
    one_draw = result["conditions"][0]
    assert one_draw["independent_draws"] == 1
    assert one_draw["candidate_count"] == 8
    assert one_draw["mean_unique_surviving_draws"] <= 1.0


def test_invalid_correlation_grids_fail_closed() -> None:
    with pytest.raises(ValueError, match="start at one"):
        run_bmrb_correlated_multiplicity_stress(
            family_replicates=1,
            candidate_count=4,
            independent_draw_counts=(2, 4),
        )
    with pytest.raises(ValueError, match="end at candidate_count"):
        run_bmrb_correlated_multiplicity_stress(
            family_replicates=1,
            candidate_count=4,
            independent_draw_counts=(1, 2),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        run_bmrb_correlated_multiplicity_stress(
            family_replicates=1,
            candidate_count=4,
            independent_draw_counts=(1, 2, 2, 4),
        )


def test_claim_boundary_remains_nonbiological() -> None:
    result = run_bmrb_correlated_multiplicity_stress(
        family_replicates=1,
        candidate_count=2,
        independent_draw_counts=(1, 2),
        participants=4,
        bootstrap_resamples=100,
        seed=5901,
    )
    assert "does not validate biological truth" in result["claim_boundary"]
    assert "physical-quantum" in result["claim_boundary"]
