from __future__ import annotations

import pytest

from quantumbci.bmrb_multiplicity_stress import (
    BMRB_MULTIPLICITY_STRESS_BENCHMARK,
    near_boundary_null_scenario,
    run_bmrb_multiplicity_stress,
)


def test_near_boundary_stress_stays_on_null_side_of_declared_effect_threshold() -> None:
    scenario = near_boundary_null_scenario()
    assert scenario.truth_class == "known_null"
    assert scenario.expected_scientific_pass is False
    assert scenario.expected_failure_component == "effect"
    assert scenario.reference_effect < 0.05
    assert scenario.alternate_effect < 0.05


def test_expanding_search_family_cannot_change_predeclared_primary_result() -> None:
    small = run_bmrb_multiplicity_stress(
        family_replicates=6,
        candidate_count=2,
        participants=4,
        bootstrap_resamples=100,
        seed=5901,
    )
    large = run_bmrb_multiplicity_stress(
        family_replicates=6,
        candidate_count=20,
        participants=4,
        bootstrap_resamples=100,
        seed=5901,
    )

    assert small["benchmark"] == BMRB_MULTIPLICITY_STRESS_BENCHMARK
    assert large["benchmark"] == BMRB_MULTIPLICITY_STRESS_BENCHMARK
    assert small["authorized_primary_promotion_rate"] == pytest.approx(
        large["authorized_primary_promotion_rate"]
    )

    small_rows = small["replicates"]
    large_rows = large["replicates"]
    assert [row["primary_passed"] for row in small_rows] == [
        row["primary_passed"] for row in large_rows
    ]
    assert all(
        large_row["candidate_pass_count"] >= small_row["candidate_pass_count"]
        for small_row, large_row in zip(small_rows, large_rows, strict=True)
    )
    assert all(
        int(large_row["naive_any_survivor"]) >= int(small_row["naive_any_survivor"])
        for small_row, large_row in zip(small_rows, large_rows, strict=True)
    )
    assert large["winner_picking_amplification"] >= small["winner_picking_amplification"]


def test_winner_picking_trap_materializes_under_fixed_development_seeds() -> None:
    result = run_bmrb_multiplicity_stress(
        family_replicates=8,
        candidate_count=20,
        participants=4,
        bootstrap_resamples=100,
        seed=5901,
    )

    assert result["scenario"]["reference_effect"] < result["scenario"][
        "validation_effect_threshold"
    ]
    assert result["naive_any_survivor_rate"] > result[
        "authorized_primary_promotion_rate"
    ]
    assert result["winner_picking_amplification"] > 0.0
    assert result["mean_suppressed_nonprimary_survivors"] > 0.0
    assert any(
        row["suppressed_nonprimary_survivors"] > 0 for row in result["replicates"]
    )
    assert "does not validate biological truth" in result["claim_boundary"]


def test_multiplicity_stress_validates_execution_budget_inputs() -> None:
    with pytest.raises(ValueError, match="family_replicates"):
        run_bmrb_multiplicity_stress(family_replicates=0)
    with pytest.raises(ValueError, match="at least two"):
        run_bmrb_multiplicity_stress(candidate_count=1)
    with pytest.raises(ValueError, match="at least four"):
        run_bmrb_multiplicity_stress(participants=3)
    with pytest.raises(ValueError, match="at least 100"):
        run_bmrb_multiplicity_stress(bootstrap_resamples=99)
