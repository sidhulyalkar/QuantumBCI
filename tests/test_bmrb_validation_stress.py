from __future__ import annotations

import pytest

from quantumbci.bmrb_validation_stress import (
    default_stress_scenarios,
    run_bmrb_validation_stress_suite,
    run_stress_replicate,
)


def test_extended_stress_suite_separates_bmrb_from_naive_rules() -> None:
    result = run_bmrb_validation_stress_suite(
        replicates=4,
        seed=2901,
        participants=8,
        bootstrap_resamples=100,
    )
    assert result["qualified"] is True

    summaries = {item["scenario_id"]: item for item in result["scenario_summaries"]}

    equivalence = summaries["equivalence-null-naive-trap"]
    assert equivalence["bmrb_pass_rate"] == pytest.approx(0.0)
    assert equivalence["naive_primary_effect_pass_rate"] == pytest.approx(1.0)

    shortcut = summaries["predictive-shortcut-naive-trap"]
    assert shortcut["bmrb_pass_rate"] == pytest.approx(0.0)
    assert shortcut["naive_primary_effect_pass_rate"] == pytest.approx(1.0)

    reversal = summaries["calibration-reversal-naive-trap"]
    assert reversal["bmrb_pass_rate"] == pytest.approx(1.0)
    assert reversal["naive_budget_averaged_effect_pass_rate"] == pytest.approx(0.0)

    for scenario_id in (
        "invertible-coordinate-positive",
        "heterogeneous-shared-positive",
        "noisy-repeated-sessions-positive",
    ):
        assert summaries[scenario_id]["bmrb_pass_rate"] == pytest.approx(1.0)


def test_repeated_sessions_remain_participant_level_evidence() -> None:
    scenario = next(
        item
        for item in default_stress_scenarios()
        if item.scenario_id == "noisy-repeated-sessions-positive"
    )
    row = run_stress_replicate(
        scenario,
        replicate=0,
        seed=9901,
        participants=8,
        bootstrap_resamples=100,
    )
    assert row.bmrb_scientific_passed is True
    assert row.participant_count == 8
    assert row.occasions_per_participant == 3


def test_invertible_coordinate_change_can_conserve_declared_mechanism() -> None:
    scenario = next(
        item
        for item in default_stress_scenarios()
        if item.scenario_id == "invertible-coordinate-positive"
    )
    row = run_stress_replicate(
        scenario,
        replicate=0,
        seed=12_901,
        participants=8,
        bootstrap_resamples=100,
    )
    assert row.bmrb_scientific_passed is True


def test_stress_scenarios_are_declared_before_generation() -> None:
    scenarios = default_stress_scenarios()
    ids = [item.scenario_id for item in scenarios]
    assert len(ids) == len(set(ids))
    assert "equivalence-null-naive-trap" in ids
    assert "calibration-reversal-naive-trap" in ids
