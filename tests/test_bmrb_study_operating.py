from __future__ import annotations

import pytest

from quantumbci.bmrb_study_operating import (
    BMRBStudyOperatingPolicy,
    StudySimulationSeedPartition,
    default_study_operating_scenarios,
    qualification_smoke_grid,
    recommended_development_grid,
    run_bmrb_study_operating_characteristics,
    run_study_operating_replicate,
)


def _policy(*, partition: str = "development", replicates: int = 1) -> BMRBStudyOperatingPolicy:
    return BMRBStudyOperatingPolicy(
        study_id="study-operating-test-v1",
        source_sha="1daa43098e6714185998d04426968fa729f2cb4c",
        partition=partition,  # type: ignore[arg-type]
        grid=qualification_smoke_grid(),
        replicates_per_cell=replicates,
        bootstrap_resamples=100,
    )


def test_nested_seed_authority_is_deterministic_unique_and_partition_disjoint() -> None:
    seeds = StudySimulationSeedPartition()
    development = {
        seeds.effective_seed("development", cell_index=cell, replicate=replicate, study_index=study)
        for cell in range(3)
        for replicate in range(4)
        for study in range(5)
    }
    evaluation = {
        seeds.effective_seed("evaluation", cell_index=cell, replicate=replicate, study_index=study)
        for cell in range(3)
        for replicate in range(4)
        for study in range(5)
    }
    assert len(development) == 60
    assert len(evaluation) == 60
    assert development.isdisjoint(evaluation)
    assert seeds.effective_seed(
        "development", cell_index=2, replicate=3, study_index=1
    ) == seeds.effective_seed("development", cell_index=2, replicate=3, study_index=1)


def test_seed_authority_rejects_overlapping_stride_or_partition_designs() -> None:
    with pytest.raises(ValueError, match="replicate_stride"):
        StudySimulationSeedPartition(
            replicate_stride=100,
            study_stride=101,
            max_studies_per_replicate=4,
        )
    with pytest.raises(ValueError, match="cell_stride"):
        StudySimulationSeedPartition(
            cell_stride=10_000,
            replicate_stride=1_000,
            study_stride=10,
            max_replicates_per_cell=16,
            max_studies_per_replicate=4,
        )
    with pytest.raises(ValueError, match="seed spaces overlap"):
        StudySimulationSeedPartition(
            development_offset=1_000_000,
            evaluation_offset=2_000_000,
        )


def test_declared_study_truths_cover_distinct_hierarchy_failure_modes() -> None:
    scenarios = {item.scenario_id: item for item in default_study_operating_scenarios()}
    assert len(scenarios) == 8
    assert scenarios["homogeneous-positive-3"].study_count == 3
    assert scenarios["homogeneous-positive-4"].study_count == 4
    assert scenarios["primary-only-positive-4"].expected_replication_pass is False
    assert scenarios["primary-only-positive-4"].expected_context_specific_only is True
    assert scenarios["primary-fail-replications-positive-4"].expected_replication_pass is False
    assert scenarios["fragile-one-conflict-4"].expected_replication_pass is True
    assert scenarios["fragile-one-conflict-4"].expected_sensitivity_warning is True
    assert scenarios["redundant-one-conflict-5"].study_count == 5
    assert scenarios["redundant-one-conflict-5"].expected_replication_pass is True
    assert scenarios["redundant-one-conflict-5"].expected_sensitivity_warning is True


def test_three_study_positive_is_successful_but_zero_margin_fragile() -> None:
    policy = _policy()
    scenario = next(
        item
        for item in default_study_operating_scenarios()
        if item.scenario_id == "homogeneous-positive-3"
    )
    row = run_study_operating_replicate(
        policy,
        scenario,
        cell_index=0,
        replicate=0,
        participants=8,
        within_scale=0.0,
        measurement_scale=0.0,
        cross_study_scale=0.0,
    )
    assert row.study_passes == (True, True, True)
    assert row.replication_criteria_passed is True
    assert row.successful_replication_margin == 0
    assert row.single_successful_replication_removal_flips_claim is True
    assert row.sensitivity_warning is True


def test_primary_cannot_be_reassigned_after_replications_succeed() -> None:
    policy = _policy()
    scenario = next(
        item
        for item in default_study_operating_scenarios()
        if item.scenario_id == "primary-fail-replications-positive-4"
    )
    row = run_study_operating_replicate(
        policy,
        scenario,
        cell_index=1,
        replicate=0,
        participants=8,
        within_scale=0.0,
        measurement_scale=0.0,
        cross_study_scale=0.0,
    )
    assert row.study_passes == (False, True, True, True)
    assert row.replication_criteria_passed is False
    assert row.context_specific_only is True


def test_margin_and_heterogeneity_are_independent_sensitivity_axes() -> None:
    policy = _policy()
    scenarios = {item.scenario_id: item for item in default_study_operating_scenarios()}
    fragile = run_study_operating_replicate(
        policy,
        scenarios["fragile-one-conflict-4"],
        cell_index=2,
        replicate=0,
        participants=8,
        within_scale=0.0,
        measurement_scale=0.0,
        cross_study_scale=0.0,
    )
    redundant_conflict = run_study_operating_replicate(
        policy,
        scenarios["redundant-one-conflict-5"],
        cell_index=3,
        replicate=0,
        participants=8,
        within_scale=0.0,
        measurement_scale=0.0,
        cross_study_scale=0.0,
    )
    robust = run_study_operating_replicate(
        policy,
        scenarios["homogeneous-positive-4"],
        cell_index=4,
        replicate=0,
        participants=8,
        within_scale=0.0,
        measurement_scale=0.0,
        cross_study_scale=0.0,
    )

    assert fragile.study_passes == (True, True, True, False)
    assert fragile.replication_criteria_passed is True
    assert fragile.successful_replication_margin == 0
    assert fragile.single_successful_replication_removal_flips_claim is True
    assert fragile.sensitivity_warning is True

    assert redundant_conflict.study_passes == (True, True, True, True, False)
    assert redundant_conflict.replication_criteria_passed is True
    assert redundant_conflict.successful_replication_margin == 1
    assert redundant_conflict.single_successful_replication_removal_flips_claim is False
    assert redundant_conflict.sensitivity_warning is True

    assert robust.study_passes == (True, True, True, True)
    assert robust.replication_criteria_passed is True
    assert robust.successful_replication_margin == 1
    assert robust.single_successful_replication_removal_flips_claim is False
    assert robust.sensitivity_warning is False


def test_smoke_grid_runs_end_to_end_and_matches_declared_truth() -> None:
    result = run_bmrb_study_operating_characteristics(_policy())
    cells = {cell.scenario_id: cell for cell in result.cells}
    assert len(result.cells) == qualification_smoke_grid().cell_count == 8
    assert all(cell.decision_error_rate == 0.0 for cell in result.cells)
    assert all(cell.context_specific_match_rate == 1.0 for cell in result.cells)
    assert all(cell.sensitivity_warning_match_rate == 1.0 for cell in result.cells)
    assert cells["primary-fail-replications-positive-4"].primary_role_protection_rate == 1.0
    assert cells["fragile-one-conflict-4"].fragile_claim_detection_rate == 1.0
    assert cells["redundant-one-conflict-5"].fragile_claim_detection_rate == 1.0
    assert cells["redundant-one-conflict-5"].mean_successful_replication_margin == 1.0
    assert result.aggregate_mapping()["qualification_defined"] is False
    mapping = result.to_mapping()
    assert mapping["evaluation_partition_executed"] is False
    assert mapping["physical_quantum_promotion_eligible"] is False


def test_operating_result_is_deterministic_and_policy_fingerprinted() -> None:
    first_policy = _policy()
    second_policy = _policy()
    assert first_policy.policy_fingerprint == second_policy.policy_fingerprint
    first = run_bmrb_study_operating_characteristics(first_policy).to_mapping()
    second = run_bmrb_study_operating_characteristics(second_policy).to_mapping()
    assert first == second

    changed = BMRBStudyOperatingPolicy(
        **{**first_policy.__dict__, "sensitivity_max_effect_range": 0.2}
    )
    assert changed.policy_fingerprint != first_policy.policy_fingerprint


def test_evaluation_partition_is_fingerprinted_but_not_executable() -> None:
    evaluation = _policy(partition="evaluation")
    assert evaluation.to_mapping()["partition"] == "evaluation"
    assert evaluation.to_mapping()["evaluation_partition_executable"] is False
    with pytest.raises(RuntimeError, match="evaluation partition remains sealed"):
        run_bmrb_study_operating_characteristics(evaluation)


def test_recommended_grid_is_development_scale_not_ci_smoke() -> None:
    grid = recommended_development_grid()
    assert grid.cell_count == 8 * 3 * 3 * 3 * 3
    assert grid.cell_count == 648
    assert grid.cell_count > qualification_smoke_grid().cell_count
