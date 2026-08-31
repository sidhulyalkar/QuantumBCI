"""Deterministic sharding for the BMRB study-level development operating program.

The scientific operating policy and result schema remain owned by ``bmrb_study_operating``.
This module only makes the development computation resumable. A shard is explicitly a
partial execution artifact and can never be interpreted as a scientific operating result.
Only a complete, exact, non-overlapping recomposition of the frozen grid may produce a
``BMRBStudyOperatingResult``.

The evaluation partition remains sealed and cannot be executed through this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .bmrb_study_operating import (
    BMRBStudyOperatingCell,
    BMRBStudyOperatingPolicy,
    BMRBStudyOperatingResult,
    _summarize_cell,
    default_study_operating_scenarios,
    run_study_operating_replicate,
)
from .bmrb_study_operating_artifacts import verify_bmrb_study_operating_mapping
from .preregistration import canonical_scientific_fingerprint

BMRB_STUDY_OPERATING_SHARD_METHOD = "deterministic_development_cell_shard_v1"
BMRB_STUDY_OPERATING_SHARD_PLAN_METHOD = "deterministic_development_shard_plan_v1"


def _positive_int(name: str, value: Any) -> int:
    if type(value) is bool:
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if number < 1 or number != value:
        raise ValueError(f"{name} must be a positive integer")
    return number


def _sha256(name: str, value: Any) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a 64-character SHA-256 hexadecimal digest")
    return text


def _require_development(policy: BMRBStudyOperatingPolicy) -> None:
    if policy.partition != "development":
        raise RuntimeError("study-level evaluation partition remains sealed in v1")


def _require_unique_axes(policy: BMRBStudyOperatingPolicy) -> None:
    """Publication-grade shard execution refuses duplicate grid weighting."""

    axes = {
        "scenario_ids": policy.grid.scenario_ids,
        "participant_counts": policy.grid.participant_counts,
        "within_study_heterogeneity_scales": policy.grid.within_study_heterogeneity_scales,
        "measurement_noise_scales": policy.grid.measurement_noise_scales,
        "cross_study_effect_scales": policy.grid.cross_study_effect_scales,
    }
    for name, values in axes.items():
        if len(values) != len(set(values)):
            raise ValueError(f"sharded study operating execution requires unique {name}")


@dataclass(frozen=True)
class BMRBStudyOperatingCellSpec:
    cell_index: int
    scenario_id: str
    participant_count: int
    within_study_heterogeneity_scale: float
    measurement_noise_scale: float
    cross_study_effect_scale: float

    def to_mapping(self) -> dict[str, Any]:
        return self.__dict__.copy()


def study_operating_cell_specs(
    policy: BMRBStudyOperatingPolicy,
) -> tuple[BMRBStudyOperatingCellSpec, ...]:
    """Return the canonical frozen cell ordering used by the monolithic v1 runner."""

    _require_unique_axes(policy)
    known = {item.scenario_id for item in default_study_operating_scenarios()}
    unknown = sorted(set(policy.grid.scenario_ids) - known)
    if unknown:
        raise ValueError(f"unknown study operating scenario ids: {unknown}")
    specs: list[BMRBStudyOperatingCellSpec] = []
    cell_index = 0
    for scenario_id in policy.grid.scenario_ids:
        for participants in policy.grid.participant_counts:
            for within_scale in policy.grid.within_study_heterogeneity_scales:
                for measurement_scale in policy.grid.measurement_noise_scales:
                    for cross_study_scale in policy.grid.cross_study_effect_scales:
                        specs.append(
                            BMRBStudyOperatingCellSpec(
                                cell_index=cell_index,
                                scenario_id=scenario_id,
                                participant_count=participants,
                                within_study_heterogeneity_scale=within_scale,
                                measurement_noise_scale=measurement_scale,
                                cross_study_effect_scale=cross_study_scale,
                            )
                        )
                        cell_index += 1
    if cell_index != policy.grid.cell_count:
        raise RuntimeError("study operating grid coverage mismatch")
    return tuple(specs)


def run_bmrb_study_operating_cell(
    policy: BMRBStudyOperatingPolicy,
    *,
    cell_index: int,
) -> BMRBStudyOperatingCell:
    """Run one canonical development cell without changing its global seed identity."""

    _require_development(policy)
    specs = study_operating_cell_specs(policy)
    if not 0 <= cell_index < len(specs):
        raise ValueError("cell_index lies outside the frozen study operating grid")
    spec = specs[cell_index]
    scenarios = {item.scenario_id: item for item in default_study_operating_scenarios()}
    scenario = scenarios[spec.scenario_id]
    if scenario.study_count > policy.seed_partition.max_studies_per_replicate:
        raise ValueError("scenario study count exceeds seed authority capacity")
    rows = tuple(
        run_study_operating_replicate(
            policy,
            scenario,
            cell_index=spec.cell_index,
            replicate=replicate,
            participants=spec.participant_count,
            within_scale=spec.within_study_heterogeneity_scale,
            measurement_scale=spec.measurement_noise_scale,
            cross_study_scale=spec.cross_study_effect_scale,
        )
        for replicate in range(policy.replicates_per_cell)
    )
    return _summarize_cell(
        scenario,
        rows,
        participants=spec.participant_count,
        within_scale=spec.within_study_heterogeneity_scale,
        measurement_scale=spec.measurement_noise_scale,
        cross_study_scale=spec.cross_study_effect_scale,
    )


@dataclass(frozen=True)
class BMRBStudyOperatingShardRange:
    start_cell: int
    stop_cell: int

    def __post_init__(self) -> None:
        if type(self.start_cell) is bool or type(self.stop_cell) is bool:
            raise ValueError("shard bounds must be integers")
        if self.start_cell < 0 or self.stop_cell <= self.start_cell:
            raise ValueError("shard range must satisfy 0 <= start_cell < stop_cell")

    @property
    def cell_count(self) -> int:
        return self.stop_cell - self.start_cell

    def to_mapping(self) -> dict[str, int]:
        return {
            "start_cell": self.start_cell,
            "stop_cell": self.stop_cell,
            "cell_count": self.cell_count,
        }


@dataclass(frozen=True)
class BMRBStudyOperatingShardPlan:
    policy_fingerprint: str
    total_cells: int
    cells_per_shard: int
    ranges: tuple[BMRBStudyOperatingShardRange, ...]

    def __post_init__(self) -> None:
        _sha256("policy_fingerprint", self.policy_fingerprint)
        _positive_int("total_cells", self.total_cells)
        _positive_int("cells_per_shard", self.cells_per_shard)
        cursor = 0
        for item in self.ranges:
            if item.start_cell != cursor:
                raise ValueError("shard plan ranges must be contiguous and ordered")
            cursor = item.stop_cell
        if cursor != self.total_cells:
            raise ValueError("shard plan must cover the complete frozen grid")

    def decision_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "method": BMRB_STUDY_OPERATING_SHARD_PLAN_METHOD,
            "policy_fingerprint": self.policy_fingerprint,
            "total_cells": self.total_cells,
            "cells_per_shard": self.cells_per_shard,
            "ranges": [item.to_mapping() for item in self.ranges],
            "partial_shards_are_scientific_results": False,
            "evaluation_partition_executable": False,
        }

    @property
    def plan_fingerprint(self) -> str:
        return canonical_scientific_fingerprint(
            "quantumbci.bmrb-study-operating-shard-plan.v1", self.decision_payload()
        )

    def to_mapping(self) -> dict[str, Any]:
        return {**self.decision_payload(), "plan_fingerprint": self.plan_fingerprint}


def plan_bmrb_study_operating_shards(
    policy: BMRBStudyOperatingPolicy,
    *,
    cells_per_shard: int,
) -> BMRBStudyOperatingShardPlan:
    """Freeze a deterministic contiguous shard plan for one development policy."""

    _require_development(policy)
    _require_unique_axes(policy)
    width = _positive_int("cells_per_shard", cells_per_shard)
    ranges = tuple(
        BMRBStudyOperatingShardRange(start, min(start + width, policy.grid.cell_count))
        for start in range(0, policy.grid.cell_count, width)
    )
    return BMRBStudyOperatingShardPlan(
        policy_fingerprint=policy.policy_fingerprint,
        total_cells=policy.grid.cell_count,
        cells_per_shard=width,
        ranges=ranges,
    )


@dataclass(frozen=True)
class BMRBStudyOperatingShard:
    policy: BMRBStudyOperatingPolicy
    shard_range: BMRBStudyOperatingShardRange
    entries: tuple[tuple[int, BMRBStudyOperatingCell], ...]

    def __post_init__(self) -> None:
        _require_development(self.policy)
        expected_indices = tuple(range(self.shard_range.start_cell, self.shard_range.stop_cell))
        observed_indices = tuple(index for index, _ in self.entries)
        if observed_indices != expected_indices:
            raise ValueError("study operating shard entries must exactly cover their declared range")

    def decision_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "method": BMRB_STUDY_OPERATING_SHARD_METHOD,
            "policy": self.policy.to_mapping(),
            "range": self.shard_range.to_mapping(),
            "entries": [
                {"cell_index": index, "cell": cell.to_mapping()} for index, cell in self.entries
            ],
            "complete_operating_result": False,
            "qualification_defined": False,
            "evaluation_partition_executed": False,
            "physical_quantum_promotion_eligible": False,
        }

    @property
    def artifact_fingerprint(self) -> str:
        return canonical_scientific_fingerprint(
            "quantumbci.bmrb-study-operating-shard.v1", self.decision_payload()
        )

    def to_mapping(self) -> dict[str, Any]:
        return {**self.decision_payload(), "artifact_fingerprint": self.artifact_fingerprint}


def run_bmrb_study_operating_shard(
    policy: BMRBStudyOperatingPolicy,
    *,
    start_cell: int,
    stop_cell: int,
) -> BMRBStudyOperatingShard:
    """Execute one deterministic development shard."""

    _require_development(policy)
    _require_unique_axes(policy)
    shard_range = BMRBStudyOperatingShardRange(start_cell, stop_cell)
    if shard_range.stop_cell > policy.grid.cell_count:
        raise ValueError("shard stop_cell exceeds the frozen study operating grid")
    entries = tuple(
        (index, run_bmrb_study_operating_cell(policy, cell_index=index))
        for index in range(shard_range.start_cell, shard_range.stop_cell)
    )
    return BMRBStudyOperatingShard(policy=policy, shard_range=shard_range, entries=entries)


def verify_bmrb_study_operating_shard_mapping(
    payload: Mapping[str, Any],
    *,
    policy: BMRBStudyOperatingPolicy,
) -> None:
    """Verify a partial shard against an independently supplied frozen policy."""

    _require_development(policy)
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("study operating shard schema_version must be 1")
    if payload.get("method") != BMRB_STUDY_OPERATING_SHARD_METHOD:
        raise ValueError("study operating shard method mismatch")
    for name in (
        "complete_operating_result",
        "qualification_defined",
        "evaluation_partition_executed",
        "physical_quantum_promotion_eligible",
    ):
        if payload.get(name) is not False:
            raise ValueError(f"study operating shard must keep {name}=false")
    claimed = _sha256("artifact_fingerprint", payload.get("artifact_fingerprint"))
    core = {key: value for key, value in payload.items() if key != "artifact_fingerprint"}
    expected = canonical_scientific_fingerprint(
        "quantumbci.bmrb-study-operating-shard.v1", core
    )
    if claimed != expected:
        raise ValueError("study operating shard fingerprint mismatch")
    if payload.get("policy") != policy.to_mapping():
        raise ValueError("study operating shard policy does not match the frozen execution policy")
    raw_range = payload.get("range")
    if not isinstance(raw_range, Mapping):
        raise ValueError("study operating shard range must be an object")
    shard_range = BMRBStudyOperatingShardRange(
        int(raw_range.get("start_cell", -1)), int(raw_range.get("stop_cell", -1))
    )
    if raw_range != shard_range.to_mapping():
        raise ValueError("study operating shard range is noncanonical")
    if shard_range.stop_cell > policy.grid.cell_count:
        raise ValueError("study operating shard range exceeds the frozen grid")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != shard_range.cell_count:
        raise ValueError("study operating shard entries do not cover the declared range")
    specs = study_operating_cell_specs(policy)
    for expected_index, raw_entry in zip(
        range(shard_range.start_cell, shard_range.stop_cell), entries, strict=True
    ):
        if not isinstance(raw_entry, Mapping):
            raise ValueError("study operating shard entry must be an object")
        if int(raw_entry.get("cell_index", -1)) != expected_index:
            raise ValueError("study operating shard cell indices must be contiguous and ordered")
        raw_cell = raw_entry.get("cell")
        if not isinstance(raw_cell, Mapping):
            raise ValueError("study operating shard cell must be an object")
        spec = specs[expected_index]
        expected_identity = {
            "scenario_id": spec.scenario_id,
            "participant_count": spec.participant_count,
            "within_study_heterogeneity_scale": spec.within_study_heterogeneity_scale,
            "measurement_noise_scale": spec.measurement_noise_scale,
            "cross_study_effect_scale": spec.cross_study_effect_scale,
        }
        for key, value in expected_identity.items():
            if raw_cell.get(key) != value:
                raise ValueError("study operating shard cell identity disagrees with frozen grid")


def load_bmrb_study_operating_shard(
    path: str | Path,
    *,
    policy: BMRBStudyOperatingPolicy,
) -> BMRBStudyOperatingShard:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("study operating shard must contain a JSON object")
    verify_bmrb_study_operating_shard_mapping(payload, policy=policy)
    raw_range = payload["range"]
    shard_range = BMRBStudyOperatingShardRange(
        int(raw_range["start_cell"]), int(raw_range["stop_cell"])
    )
    entries = tuple(
        (int(entry["cell_index"]), BMRBStudyOperatingCell(**dict(entry["cell"])))
        for entry in payload["entries"]
    )
    return BMRBStudyOperatingShard(policy=policy, shard_range=shard_range, entries=entries)


def write_bmrb_study_operating_shard(
    path: str | Path,
    shard: BMRBStudyOperatingShard,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(shard.to_mapping(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def merge_bmrb_study_operating_shards(
    policy: BMRBStudyOperatingPolicy,
    shards: Sequence[BMRBStudyOperatingShard],
) -> BMRBStudyOperatingResult:
    """Recompose the exact complete grid; gaps, overlaps, or policy drift fail closed."""

    _require_development(policy)
    _require_unique_axes(policy)
    if not shards:
        raise ValueError("at least one study operating shard is required")
    cells_by_index: dict[int, BMRBStudyOperatingCell] = {}
    for shard in shards:
        if shard.policy.to_mapping() != policy.to_mapping():
            raise ValueError("study operating shard policy drift detected")
        for index, cell in shard.entries:
            if index in cells_by_index:
                raise ValueError("study operating shard overlap detected")
            cells_by_index[index] = cell
    expected_indices = set(range(policy.grid.cell_count))
    observed_indices = set(cells_by_index)
    if observed_indices != expected_indices:
        missing = sorted(expected_indices - observed_indices)
        extra = sorted(observed_indices - expected_indices)
        raise ValueError(
            f"study operating shard coverage incomplete or invalid: missing={missing[:8]}, extra={extra[:8]}"
        )
    result = BMRBStudyOperatingResult(
        policy=policy,
        cells=tuple(cells_by_index[index] for index in range(policy.grid.cell_count)),
    )
    verify_bmrb_study_operating_mapping(result.to_mapping())
    return result
