from __future__ import annotations

import pytest

from quantumbci.bmrb_validation_dependence import (
    BMRBDependenceScenario,
    default_dependence_scenarios,
    run_bmrb_dependence_stress_suite,
    run_dependence_replicate,
    validate_structured_missing_pair_rejection,
)


def test_dependence_stress_separates_participant_estimand_from_row_pooling() -> None:
    result = run_bmrb_dependence_stress_suite(
        replicates=4,
        seed=3901,
        bootstrap_resamples=100,
    )
    assert result["qualified"] is True
    summaries = {item["scenario_id"]: item for item in result["scenario_summaries"]}

    majority = summaries["majority-responder-imbalanced-positive"]
    assert majority["bmrb_pass_rate"] == pytest.approx(1.0)
    assert majority["row_weighted_effect_pass_rate"] == pytest.approx(0.0)
    assert majority["participant_balanced_effect_pass_rate"] == pytest.approx(1.0)

    minority = summaries["minority-responder-overweight-trap"]
    assert minority["bmrb_pass_rate"] == pytest.approx(0.0)
    assert minority["row_weighted_effect_pass_rate"] == pytest.approx(1.0)
    assert minority["participant_balanced_effect_pass_rate"] == pytest.approx(0.0)
    assert minority["expected_failure_localization_rate"] == pytest.approx(1.0)


def test_unequal_sessions_preserve_participant_count() -> None:
    for index, scenario in enumerate(default_dependence_scenarios()):
        row = run_dependence_replicate(
            scenario,
            replicate=0,
            seed=4901 + index,
            bootstrap_resamples=100,
        )
        assert row.participant_count == scenario.participants
        assert row.responder_count == scenario.responder_count
        assert row.min_sessions_per_participant == 1
        assert row.max_sessions_per_participant == 20


def test_structured_missingness_is_invalid_not_negative_evidence() -> None:
    result = validate_structured_missing_pair_rejection(seed=5901)
    assert result["rejected"] is True
    assert result["removed_representation_rows"] == 2
    assert result["classification"] == "software_invalid"
    assert result["scientific_negative"] is False
    assert "exactly paired" in result["reason"]


def test_dependence_scenarios_encode_both_weighting_failure_directions() -> None:
    scenarios = {item.scenario_id: item for item in default_dependence_scenarios()}
    majority = scenarios["majority-responder-imbalanced-positive"]
    minority = scenarios["minority-responder-overweight-trap"]

    assert majority.expected_bmrb_pass is True
    assert majority.responder_count == 7
    assert majority.nonresponder_sessions > majority.responder_sessions

    assert minority.expected_bmrb_pass is False
    assert minority.expected_failure_component == "effect"
    assert minority.responder_count == 2
    assert minority.responder_sessions > minority.nonresponder_sessions


def test_dependence_scenario_rejects_invalid_responder_and_session_contracts() -> None:
    with pytest.raises(ValueError, match="responder_count"):
        BMRBDependenceScenario(
            scenario_id="invalid-responders",
            expected_bmrb_pass=False,
            participants=8,
            responder_count=0,
            responder_sessions=1,
            nonresponder_sessions=1,
        )
    with pytest.raises(ValueError, match="session counts"):
        BMRBDependenceScenario(
            scenario_id="invalid-sessions",
            expected_bmrb_pass=False,
            participants=8,
            responder_count=2,
            responder_sessions=0,
            nonresponder_sessions=1,
        )
