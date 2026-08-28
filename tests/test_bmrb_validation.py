from __future__ import annotations

import pytest

from quantumbci.bmrb_validation import (
    BMRB_VALIDATION_BENCHMARK,
    default_validation_scenarios,
    run_bmrb_validation_suite,
    run_validation_replicate,
    validate_missing_pair_rejection,
    validation_policy,
    validation_policy_for_scenario,
)


def test_known_truth_suite_qualifies_and_localizes_failures() -> None:
    result = run_bmrb_validation_suite(
        replicates=4,
        seed=1901,
        participants=8,
        bootstrap_resamples=100,
    )
    assert result["benchmark"] == BMRB_VALIDATION_BENCHMARK
    assert result["qualified"] is True
    assert result["missing_pair_rejected"] is True

    summaries = {item["scenario_id"]: item for item in result["scenario_summaries"]}
    assert summaries["effect-null"]["observed_pass_rate"] == pytest.approx(0.0)
    assert summaries["equivalence-null"]["observed_pass_rate"] == pytest.approx(0.0)
    assert summaries["shared-mechanism-positive"]["observed_pass_rate"] == pytest.approx(1.0)
    assert summaries["predictive-shortcut"]["observed_pass_rate"] == pytest.approx(0.0)
    assert summaries["representation-specific"]["observed_pass_rate"] == pytest.approx(0.0)
    assert summaries["coverage-insufficient-family-support"]["observed_pass_rate"] == pytest.approx(
        0.0
    )
    assert summaries["calibration-reversal"]["observed_pass_rate"] == pytest.approx(1.0)
    assert all(
        item["expected_failure_localization_rate"] == pytest.approx(1.0)
        for item in result["scenario_summaries"]
    )


def test_each_adversary_hits_its_declared_component() -> None:
    for index, scenario in enumerate(default_validation_scenarios()):
        row = run_validation_replicate(
            scenario,
            replicate=0,
            seed=5000 + index * 100,
            participants=8,
            bootstrap_resamples=100,
        )
        assert row.scientific_criteria_passed is scenario.expected_scientific_pass
        assert row.expected_failure_localized is True
        if scenario.expected_failure_component == "effect":
            assert row.effect_criteria_passed is False
        elif scenario.expected_failure_component == "adversary":
            assert row.adversary_survival_passed is False
        elif scenario.expected_failure_component == "conservation":
            assert row.conservation_criteria_passed is False
        elif scenario.expected_failure_component == "coverage":
            assert row.effect_criteria_passed is True
            assert row.adversary_survival_passed is True
            assert row.conservation_criteria_passed is True
            assert row.coverage_criteria_passed is False


def test_coverage_negative_changes_only_the_preregistered_coverage_requirement() -> None:
    scenarios = {scenario.scenario_id: scenario for scenario in default_validation_scenarios()}
    scenario = scenarios["coverage-insufficient-family-support"]
    baseline = validation_policy(participants=8, bootstrap_resamples=100)
    coverage = validation_policy_for_scenario(
        scenario,
        participants=8,
        bootstrap_resamples=100,
    )

    assert baseline.min_representation_families == 2
    assert coverage.min_representation_families == 3
    assert coverage.min_participants == baseline.min_participants
    assert coverage.min_representations == baseline.min_representations
    assert coverage.min_candidate_advantage == baseline.min_candidate_advantage
    assert coverage.min_ablation_necessity == baseline.min_ablation_necessity
    assert coverage.min_direction_match_fraction == baseline.min_direction_match_fraction
    assert coverage.min_information_novel_representation_fraction == (
        baseline.min_information_novel_representation_fraction
    )
    assert "coverage-negative" in coverage.policy_id


def test_missing_representation_pair_is_rejected() -> None:
    assert validate_missing_pair_rejection(seed=71) is True


def test_validation_policy_is_not_confirmatory_biological_authority() -> None:
    policy = validation_policy(participants=8, bootstrap_resamples=100)
    assert policy.confirmatory_authority is False
    assert policy.preregistration is None
    assert "Synthetic software-validation" in policy.sample_size_rationale


def test_validation_requires_multiple_independent_participants() -> None:
    with pytest.raises(ValueError, match="at least four"):
        validation_policy(participants=3, bootstrap_resamples=100)
