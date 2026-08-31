from __future__ import annotations

import json

import pytest

from quantumbci.bmrb_study_operating import (
    BMRBStudyOperatingGrid,
    BMRBStudyOperatingPolicy,
    qualification_smoke_grid,
    recommended_development_grid,
    run_bmrb_study_operating_characteristics,
)
from quantumbci.bmrb_study_operating_shards import (
    BMRBStudyOperatingShard,
    BMRBStudyOperatingShardRange,
    merge_bmrb_study_operating_shards,
    plan_bmrb_study_operating_shards,
    run_bmrb_study_operating_shard,
    study_operating_cell_specs,
    verify_bmrb_study_operating_shard_mapping,
)


def _policy(
    *,
    partition: str = "development",
    grid: BMRBStudyOperatingGrid | None = None,
    source_sha: str = "22c99b4cac857fd4d2518386a56c5155873e1d38",
) -> BMRBStudyOperatingPolicy:
    return BMRBStudyOperatingPolicy(
        study_id="study-operating-shard-test-v1",
        source_sha=source_sha,
        partition=partition,  # type: ignore[arg-type]
        grid=qualification_smoke_grid() if grid is None else grid,
        replicates_per_cell=1,
        bootstrap_resamples=100,
    )


def test_cell_specs_match_frozen_monolithic_order() -> None:
    policy = _policy()
    specs = study_operating_cell_specs(policy)
    assert len(specs) == qualification_smoke_grid().cell_count == 8
    assert tuple(spec.cell_index for spec in specs) == tuple(range(8))
    assert tuple(spec.scenario_id for spec in specs) == policy.grid.scenario_ids
    assert all(spec.participant_count == 8 for spec in specs)
    assert all(spec.within_study_heterogeneity_scale == 0.0 for spec in specs)
    assert all(spec.measurement_noise_scale == 0.0 for spec in specs)
    assert all(spec.cross_study_effect_scale == 0.0 for spec in specs)


def test_recommended_development_plan_is_complete_deterministic_and_operational_only() -> None:
    policy = _policy(grid=recommended_development_grid())
    first = plan_bmrb_study_operating_shards(policy, cells_per_shard=32)
    second = plan_bmrb_study_operating_shards(policy, cells_per_shard=32)

    assert first.total_cells == 648
    assert len(first.ranges) == 21
    assert first.ranges[0].to_mapping() == {"start_cell": 0, "stop_cell": 32, "cell_count": 32}
    assert first.ranges[-1].to_mapping() == {"start_cell": 640, "stop_cell": 648, "cell_count": 8}
    assert first.plan_fingerprint == second.plan_fingerprint
    assert first.to_mapping() == second.to_mapping()
    assert first.to_mapping()["partial_shards_are_scientific_results"] is False
    assert first.to_mapping()["evaluation_partition_executable"] is False


def test_split_smoke_recomposition_is_exactly_identical_to_monolithic_result() -> None:
    policy = _policy()
    monolithic = run_bmrb_study_operating_characteristics(policy).to_mapping()
    left = run_bmrb_study_operating_shard(policy, start_cell=0, stop_cell=3)
    right = run_bmrb_study_operating_shard(policy, start_cell=3, stop_cell=8)
    recomposed = merge_bmrb_study_operating_shards(policy, (left, right)).to_mapping()

    assert recomposed == monolithic
    assert recomposed["evaluation_partition_executed"] is False
    assert recomposed["qualification_defined"] is False
    assert recomposed["physical_quantum_promotion_eligible"] is False


def test_shard_serialization_is_deterministic_and_partial() -> None:
    policy = _policy()
    shard = run_bmrb_study_operating_shard(policy, start_cell=2, stop_cell=5)
    mapping = shard.to_mapping()

    assert mapping == shard.to_mapping()
    assert mapping["range"] == {"start_cell": 2, "stop_cell": 5, "cell_count": 3}
    assert mapping["complete_operating_result"] is False
    assert mapping["qualification_defined"] is False
    assert mapping["evaluation_partition_executed"] is False
    assert mapping["physical_quantum_promotion_eligible"] is False
    assert [entry["cell_index"] for entry in mapping["entries"]] == [2, 3, 4]
    json.dumps(mapping)
    verify_bmrb_study_operating_shard_mapping(mapping, policy=policy)


def test_shard_verifier_rejects_tampering_even_when_outer_structure_looks_valid() -> None:
    policy = _policy()
    shard = run_bmrb_study_operating_shard(policy, start_cell=0, stop_cell=2)
    tampered = shard.to_mapping()
    tampered["entries"][0]["cell"]["participant_count"] = 16

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_bmrb_study_operating_shard_mapping(tampered, policy=policy)


def test_recomposition_rejects_gap_overlap_and_policy_drift() -> None:
    policy = _policy()
    first = run_bmrb_study_operating_shard(policy, start_cell=0, stop_cell=3)
    tail = run_bmrb_study_operating_shard(policy, start_cell=4, stop_cell=8)
    with pytest.raises(ValueError, match="coverage incomplete"):
        merge_bmrb_study_operating_shards(policy, (first, tail))

    overlap = run_bmrb_study_operating_shard(policy, start_cell=2, stop_cell=8)
    with pytest.raises(ValueError, match="overlap"):
        merge_bmrb_study_operating_shards(policy, (first, overlap))

    changed = _policy(source_sha="a" * 40)
    changed_shard = run_bmrb_study_operating_shard(changed, start_cell=3, stop_cell=8)
    with pytest.raises(ValueError, match="policy drift"):
        merge_bmrb_study_operating_shards(policy, (first, changed_shard))


def test_evaluation_partition_cannot_be_planned_run_or_recomposed() -> None:
    evaluation = _policy(partition="evaluation")
    with pytest.raises(RuntimeError, match="evaluation partition remains sealed"):
        plan_bmrb_study_operating_shards(evaluation, cells_per_shard=2)
    with pytest.raises(RuntimeError, match="evaluation partition remains sealed"):
        run_bmrb_study_operating_shard(evaluation, start_cell=0, stop_cell=1)

    development = _policy()
    shard = run_bmrb_study_operating_shard(development, start_cell=0, stop_cell=8)
    with pytest.raises(RuntimeError, match="evaluation partition remains sealed"):
        merge_bmrb_study_operating_shards(evaluation, (shard,))


def test_publication_grade_shards_reject_duplicate_numeric_axes() -> None:
    duplicate_grid = BMRBStudyOperatingGrid(
        scenario_ids=("homogeneous-positive-4", "homogeneous-null-4"),
        participant_counts=(8, 8),
        within_study_heterogeneity_scales=(0.0,),
        measurement_noise_scales=(0.0,),
        cross_study_effect_scales=(0.0,),
    )
    policy = _policy(grid=duplicate_grid)
    with pytest.raises(ValueError, match="unique participant_counts"):
        study_operating_cell_specs(policy)
    with pytest.raises(ValueError, match="unique participant_counts"):
        plan_bmrb_study_operating_shards(policy, cells_per_shard=2)


def test_shard_object_requires_exact_contiguous_declared_entries() -> None:
    policy = _policy()
    full = run_bmrb_study_operating_shard(policy, start_cell=0, stop_cell=2)
    with pytest.raises(ValueError, match="exactly cover"):
        BMRBStudyOperatingShard(
            policy=policy,
            shard_range=BMRBStudyOperatingShardRange(0, 2),
            entries=(full.entries[1], full.entries[0]),
        )
