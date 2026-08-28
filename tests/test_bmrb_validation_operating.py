from __future__ import annotations

import json

import pytest

from quantumbci.bmrb_validation_operating import (
    BMRB_OPERATING_CHARACTERISTICS_BENCHMARK,
    BMRBOperatingStudyPolicy,
    OperatingCurveGrid,
    SimulationSeedPartition,
    qualification_smoke_grid,
    recommended_development_grid,
    run_bmrb_operating_characteristics,
    write_bmrb_operating_characteristics,
)


def _smoke_policy(*, partition: str = "development") -> BMRBOperatingStudyPolicy:
    return BMRBOperatingStudyPolicy(
        study_id="ci-operating-characteristics",
        source_sha="0123456789abcdef",
        partition=partition,  # type is intentionally checked by the policy at runtime
        grid=qualification_smoke_grid(),
        replicates_per_cell=2,
        bootstrap_resamples=100,
    )


def test_seed_partitions_are_deterministic_and_disjoint() -> None:
    authority = SimulationSeedPartition()
    development = {
        authority.effective_rng_seed("development", cell_index=cell, replicate=replicate)
        for cell in range(4)
        for replicate in range(5)
    }
    evaluation = {
        authority.effective_rng_seed("evaluation", cell_index=cell, replicate=replicate)
        for cell in range(4)
        for replicate in range(5)
    }

    assert development.isdisjoint(evaluation)
    assert authority.partitions_are_disjoint(cell_count=4, replicates_per_cell=5)
    assert authority.base_seed("development", cell_index=3) == authority.base_seed(
        "development", cell_index=3
    )
    assert authority.fingerprint == SimulationSeedPartition().fingerprint


def test_policy_rejects_overlapping_development_and_evaluation_authority() -> None:
    overlapping = SimulationSeedPartition(
        development_offset=10,
        evaluation_offset=20,
        cell_stride=100,
        replicate_stride=1,
        max_replicates_per_cell=10,
    )
    grid = OperatingCurveGrid(
        scenario_ids=("effect-null",),
        participant_counts=(4, 8),
        effect_scales=(1.0,),
        heterogeneity_scales=(1.0,),
        measurement_noise_scales=(1.0,),
    )

    with pytest.raises(ValueError, match="seed authorities overlap"):
        BMRBOperatingStudyPolicy(
            study_id="overlapping-authority",
            source_sha="deadbeef",
            partition="development",
            grid=grid,
            replicates_per_cell=2,
            bootstrap_resamples=100,
            seed_partition=overlapping,
        )


def test_grid_and_policy_fingerprints_bind_scientific_decisions() -> None:
    grid = qualification_smoke_grid()
    expanded = OperatingCurveGrid(
        scenario_ids=grid.scenario_ids,
        participant_counts=(4, 8),
        effect_scales=grid.effect_scales,
        heterogeneity_scales=grid.heterogeneity_scales,
        measurement_noise_scales=grid.measurement_noise_scales,
    )
    development = _smoke_policy(partition="development")
    evaluation = _smoke_policy(partition="evaluation")

    assert grid.fingerprint != expanded.fingerprint
    assert development.policy_fingerprint != evaluation.policy_fingerprint
    assert development.to_mapping()["grid_fingerprint"] == grid.fingerprint
    assert (
        development.to_mapping()["seed_partition_fingerprint"]
        == development.seed_partition.fingerprint
    )


def test_recommended_development_grid_is_substantive_but_predeclared() -> None:
    grid = recommended_development_grid()

    assert grid.participant_counts == (4, 8, 16, 32)
    assert grid.effect_scales == (0.5, 0.75, 1.0, 1.25)
    assert grid.heterogeneity_scales == (0.5, 1.0, 2.0)
    assert grid.measurement_noise_scales == (0.5, 1.0, 2.0)
    assert grid.cell_count == 864


def test_smoke_operating_study_recovers_declared_nulls_and_positives() -> None:
    result = run_bmrb_operating_characteristics(_smoke_policy())
    by_scenario = {cell.scenario_id: cell for cell in result.cells}

    assert len(result.cells) == qualification_smoke_grid().cell_count == 4
    assert by_scenario["effect-null"].observed_pass_rate == 0.0
    assert by_scenario["equivalence-null"].observed_pass_rate == 0.0
    assert by_scenario["shared-mechanism-positive"].observed_pass_rate == 1.0
    assert by_scenario["calibration-reversal"].observed_pass_rate == 1.0
    assert all(cell.expected_failure_localization_rate == 1.0 for cell in result.cells)
    assert all(0.0 <= cell.pass_rate_ci_lower <= cell.pass_rate_ci_upper <= 1.0 for cell in result.cells)

    aggregate = result.aggregate_mapping()
    assert aggregate["false_promotion_rate"] == 0.0
    assert aggregate["known_positive_recovery_rate"] == 1.0
    assert result.scientific_payload()["qualification_defined"] is False


def test_operating_study_is_bitwise_deterministic_for_one_frozen_policy() -> None:
    policy = _smoke_policy()
    first = run_bmrb_operating_characteristics(policy)
    second = run_bmrb_operating_characteristics(policy)

    assert first.to_mapping() == second.to_mapping()
    assert first.artifact_fingerprint == second.artifact_fingerprint


def test_operating_artifact_round_trip_preserves_fingerprint(tmp_path) -> None:
    result = run_bmrb_operating_characteristics(_smoke_policy())
    output = write_bmrb_operating_characteristics(result, tmp_path / "operating.json")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["benchmark"] == BMRB_OPERATING_CHARACTERISTICS_BENCHMARK
    assert payload["policy"]["policy_fingerprint"] == result.policy.policy_fingerprint
    assert payload["artifact_fingerprint"] == result.artifact_fingerprint
    assert payload["qualification_defined"] is False
    assert "does not validate biological truth" in payload["interpretation"]


def test_unknown_scenario_fails_closed_before_simulation() -> None:
    grid = OperatingCurveGrid(
        scenario_ids=("not-a-real-scenario",),
        participant_counts=(4,),
        effect_scales=(1.0,),
        heterogeneity_scales=(1.0,),
        measurement_noise_scales=(1.0,),
    )
    policy = BMRBOperatingStudyPolicy(
        study_id="unknown-scenario",
        source_sha="deadbeef",
        partition="development",
        grid=grid,
        replicates_per_cell=1,
        bootstrap_resamples=100,
    )

    with pytest.raises(ValueError, match="unknown operating-grid scenario ids"):
        run_bmrb_operating_characteristics(policy)
