"""Frozen operating-characteristics studies for known-truth BMRB validation.

v0.19 established deterministic software qualification on a compact set of declared
known-truth scenarios. This module is the next methods layer: it evaluates the same
production BMRB decision machinery over a predeclared parameter grid and reports Monte
Carlo uncertainty without turning the CI qualification thresholds into biological
significance thresholds.

The development and final-evaluation simulation seed partitions are intentionally
separate. A final methods analysis should freeze its grid and policy fingerprint before
executing the evaluation partition.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from hashlib import sha256
from itertools import product
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .bmrb_validation import (
    BMRBValidationReplicate,
    BMRBValidationScenario,
    default_validation_scenarios,
    run_validation_replicate,
)


BMRB_OPERATING_CHARACTERISTICS_BENCHMARK = "BMRB_KNOWN_TRUTH_OPERATING_CURVES_V1"
BMRB_OPERATING_CHARACTERISTICS_METHOD = "frozen_grid_monte_carlo_v1"
PartitionName = Literal["development", "evaluation"]
NORMAL_95 = 1.959963984540054


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _scientific_fingerprint(label: str, payload: Any) -> str:
    encoded = _canonical_json(payload).encode("utf-8")
    return sha256(label.encode("utf-8") + b"\0" + encoded).hexdigest()


def _positive_int(name: str, value: int) -> int:
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _positive_finite(name: str, value: float) -> float:
    normalized = float(value)
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return normalized


@dataclass(frozen=True)
class SimulationSeedPartition:
    """Deterministic, non-overlapping development/evaluation Monte Carlo authority."""

    development_offset: int = 10_000_000
    evaluation_offset: int = 1_000_000_000_000_000
    cell_stride: int = 10_000_019
    replicate_stride: int = 1009
    max_replicates_per_cell: int = 5000

    def __post_init__(self) -> None:
        for name in (
            "development_offset",
            "evaluation_offset",
            "cell_stride",
            "replicate_stride",
            "max_replicates_per_cell",
        ):
            object.__setattr__(self, name, _positive_int(name, getattr(self, name)))
        if self.development_offset >= self.evaluation_offset:
            raise ValueError("development seed authority must precede evaluation authority")
        maximum_within_cell_span = (self.max_replicates_per_cell - 1) * self.replicate_stride
        if self.cell_stride <= maximum_within_cell_span:
            raise ValueError(
                "cell_stride must exceed the maximum within-cell replicate seed span"
            )

    def offset_for(self, partition: PartitionName) -> int:
        if partition == "development":
            return self.development_offset
        if partition == "evaluation":
            return self.evaluation_offset
        raise ValueError("partition must be development or evaluation")

    def base_seed(self, partition: PartitionName, *, cell_index: int) -> int:
        index = int(cell_index)
        if index < 0:
            raise ValueError("cell_index must be non-negative")
        return self.offset_for(partition) + index * self.cell_stride

    def effective_rng_seed(
        self,
        partition: PartitionName,
        *,
        cell_index: int,
        replicate: int,
    ) -> int:
        replicate_index = int(replicate)
        if not 0 <= replicate_index < self.max_replicates_per_cell:
            raise ValueError("replicate exceeds the preregistered seed-partition capacity")
        return self.base_seed(partition, cell_index=cell_index) + replicate_index * self.replicate_stride

    def partitions_are_disjoint(self, *, cell_count: int, replicates_per_cell: int) -> bool:
        cells = _positive_int("cell_count", cell_count)
        replicates = _positive_int("replicates_per_cell", replicates_per_cell)
        if replicates > self.max_replicates_per_cell:
            return False
        development_maximum = self.effective_rng_seed(
            "development",
            cell_index=cells - 1,
            replicate=replicates - 1,
        )
        evaluation_minimum = self.base_seed("evaluation", cell_index=0)
        return development_maximum < evaluation_minimum

    def to_mapping(self) -> dict[str, Any]:
        return {
            "method": "disjoint_arithmetic_seed_partitions_v1",
            "development_offset": self.development_offset,
            "evaluation_offset": self.evaluation_offset,
            "cell_stride": self.cell_stride,
            "replicate_stride": self.replicate_stride,
            "max_replicates_per_cell": self.max_replicates_per_cell,
        }

    @property
    def fingerprint(self) -> str:
        return _scientific_fingerprint(
            "quantumbci.bmrb-operating-seed-partition.v1",
            self.to_mapping(),
        )


@dataclass(frozen=True)
class OperatingCurveGrid:
    """Predeclared DGM parameter grid for one operating-characteristics study."""

    scenario_ids: tuple[str, ...]
    participant_counts: tuple[int, ...]
    effect_scales: tuple[float, ...]
    heterogeneity_scales: tuple[float, ...]
    measurement_noise_scales: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.scenario_ids:
            raise ValueError("scenario_ids must not be empty")
        normalized_scenarios = tuple(str(value).strip() for value in self.scenario_ids)
        if any(not value for value in normalized_scenarios):
            raise ValueError("scenario_ids must contain non-empty identifiers")
        if len(set(normalized_scenarios)) != len(normalized_scenarios):
            raise ValueError("scenario_ids must be unique")
        object.__setattr__(self, "scenario_ids", normalized_scenarios)

        participants = tuple(_positive_int("participant_count", value) for value in self.participant_counts)
        if not participants or len(set(participants)) != len(participants):
            raise ValueError("participant_counts must be non-empty and unique")
        if any(value < 4 for value in participants):
            raise ValueError("operating-curve cells require at least four participants")
        object.__setattr__(self, "participant_counts", participants)

        for name in (
            "effect_scales",
            "heterogeneity_scales",
            "measurement_noise_scales",
        ):
            values = tuple(_positive_finite(name, value) for value in getattr(self, name))
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must be non-empty and unique")
            object.__setattr__(self, name, values)

    @property
    def cell_count(self) -> int:
        return int(
            len(self.scenario_ids)
            * len(self.participant_counts)
            * len(self.effect_scales)
            * len(self.heterogeneity_scales)
            * len(self.measurement_noise_scales)
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "scenario_ids": list(self.scenario_ids),
            "participant_counts": list(self.participant_counts),
            "effect_scales": list(self.effect_scales),
            "heterogeneity_scales": list(self.heterogeneity_scales),
            "measurement_noise_scales": list(self.measurement_noise_scales),
            "cell_count": self.cell_count,
        }

    @property
    def fingerprint(self) -> str:
        return _scientific_fingerprint(
            "quantumbci.bmrb-operating-grid.v1",
            self.to_mapping(),
        )


def qualification_smoke_grid() -> OperatingCurveGrid:
    """Small deterministic grid suitable for CI and API qualification only."""

    return OperatingCurveGrid(
        scenario_ids=(
            "effect-null",
            "equivalence-null",
            "shared-mechanism-positive",
            "calibration-reversal",
        ),
        participant_counts=(4,),
        effect_scales=(1.0,),
        heterogeneity_scales=(1.0,),
        measurement_noise_scales=(1.0,),
    )


def recommended_development_grid() -> OperatingCurveGrid:
    """A substantive development grid that remains distinct from final evaluation authority."""

    return OperatingCurveGrid(
        scenario_ids=tuple(scenario.scenario_id for scenario in default_validation_scenarios()),
        participant_counts=(4, 8, 16, 32),
        effect_scales=(0.50, 0.75, 1.00, 1.25),
        heterogeneity_scales=(0.50, 1.00, 2.00),
        measurement_noise_scales=(0.50, 1.00, 2.00),
    )


@dataclass(frozen=True)
class BMRBOperatingStudyPolicy:
    """Frozen authority for one development or final-evaluation simulation study."""

    study_id: str
    source_sha: str
    partition: PartitionName
    grid: OperatingCurveGrid
    replicates_per_cell: int
    bootstrap_resamples: int = 300
    primary_calibration_per_class: int = 10
    seed_partition: SimulationSeedPartition = field(default_factory=SimulationSeedPartition)

    def __post_init__(self) -> None:
        for name in ("study_id", "source_sha"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty")
            object.__setattr__(self, name, value)
        if self.partition not in {"development", "evaluation"}:
            raise ValueError("partition must be development or evaluation")
        object.__setattr__(
            self,
            "replicates_per_cell",
            _positive_int("replicates_per_cell", self.replicates_per_cell),
        )
        if self.replicates_per_cell > self.seed_partition.max_replicates_per_cell:
            raise ValueError("replicates_per_cell exceeds seed-partition capacity")
        if int(self.bootstrap_resamples) < 100:
            raise ValueError("bootstrap_resamples must be at least 100")
        object.__setattr__(self, "bootstrap_resamples", int(self.bootstrap_resamples))
        if int(self.primary_calibration_per_class) < 0:
            raise ValueError("primary_calibration_per_class must be non-negative")
        object.__setattr__(
            self,
            "primary_calibration_per_class",
            int(self.primary_calibration_per_class),
        )
        if not self.seed_partition.partitions_are_disjoint(
            cell_count=self.grid.cell_count,
            replicates_per_cell=self.replicates_per_cell,
        ):
            raise ValueError("development and evaluation seed authorities overlap")

    def decision_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "benchmark": BMRB_OPERATING_CHARACTERISTICS_BENCHMARK,
            "method": BMRB_OPERATING_CHARACTERISTICS_METHOD,
            "study_id": self.study_id,
            "source_sha": self.source_sha,
            "partition": self.partition,
            "grid": self.grid.to_mapping(),
            "grid_fingerprint": self.grid.fingerprint,
            "replicates_per_cell": self.replicates_per_cell,
            "bootstrap_resamples": self.bootstrap_resamples,
            "primary_calibration_per_class": self.primary_calibration_per_class,
            "seed_partition": self.seed_partition.to_mapping(),
            "seed_partition_fingerprint": self.seed_partition.fingerprint,
        }

    @property
    def policy_fingerprint(self) -> str:
        return _scientific_fingerprint(
            "quantumbci.bmrb-operating-policy.v1",
            self.decision_payload(),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            **self.decision_payload(),
            "policy_fingerprint": self.policy_fingerprint,
        }


@dataclass(frozen=True)
class OperatingCurveCellResult:
    cell_index: int
    scenario_id: str
    truth_class: str
    expected_scientific_pass: bool
    expected_failure_component: str | None
    participant_count: int
    effect_scale: float
    heterogeneity_scale: float
    measurement_noise_scale: float
    base_seed: int
    replicates: int
    observed_passes: int
    observed_pass_rate: float
    decision_error_rate: float
    monte_carlo_se: float
    pass_rate_ci_lower: float
    pass_rate_ci_upper: float
    expected_failure_localization_rate: float
    mean_reference_effect_bias: float
    reference_effect_rmse: float
    reference_ci_coverage: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "cell_index": self.cell_index,
            "scenario_id": self.scenario_id,
            "truth_class": self.truth_class,
            "expected_scientific_pass": self.expected_scientific_pass,
            "expected_failure_component": self.expected_failure_component,
            "participant_count": self.participant_count,
            "effect_scale": self.effect_scale,
            "heterogeneity_scale": self.heterogeneity_scale,
            "measurement_noise_scale": self.measurement_noise_scale,
            "base_seed": self.base_seed,
            "replicates": self.replicates,
            "observed_passes": self.observed_passes,
            "observed_pass_rate": self.observed_pass_rate,
            "decision_error_rate": self.decision_error_rate,
            "monte_carlo_se": self.monte_carlo_se,
            "pass_rate_ci_lower": self.pass_rate_ci_lower,
            "pass_rate_ci_upper": self.pass_rate_ci_upper,
            "expected_failure_localization_rate": self.expected_failure_localization_rate,
            "mean_reference_effect_bias": self.mean_reference_effect_bias,
            "reference_effect_rmse": self.reference_effect_rmse,
            "reference_ci_coverage": self.reference_ci_coverage,
        }


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    n = _positive_int("trials", trials)
    k = int(successes)
    if not 0 <= k <= n:
        raise ValueError("successes must lie between zero and trials")
    proportion = k / n
    z2 = NORMAL_95**2
    denominator = 1.0 + z2 / n
    center = (proportion + z2 / (2.0 * n)) / denominator
    half_width = (
        NORMAL_95
        * np.sqrt(proportion * (1.0 - proportion) / n + z2 / (4.0 * n * n))
        / denominator
    )
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def _scaled_scenario(
    scenario: BMRBValidationScenario,
    *,
    effect_scale: float,
    heterogeneity_scale: float,
    measurement_noise_scale: float,
) -> BMRBValidationScenario:
    effect = _positive_finite("effect_scale", effect_scale)
    heterogeneity = _positive_finite("heterogeneity_scale", heterogeneity_scale)
    measurement = _positive_finite("measurement_noise_scale", measurement_noise_scale)
    return replace(
        scenario,
        reference_effect=float(scenario.reference_effect) * effect,
        alternate_effect=float(scenario.alternate_effect) * effect,
        reference_ablation=float(scenario.reference_ablation) * effect,
        alternate_ablation=float(scenario.alternate_ablation) * effect,
        participant_effect_sd=float(scenario.participant_effect_sd) * heterogeneity,
        measurement_sd=float(scenario.measurement_sd) * measurement,
        secondary_budget_effect=(
            None
            if scenario.secondary_budget_effect is None
            else float(scenario.secondary_budget_effect) * effect
        ),
        secondary_budget_ablation=(
            None
            if scenario.secondary_budget_ablation is None
            else float(scenario.secondary_budget_ablation) * effect
        ),
    )


def _summarize_cell(
    *,
    cell_index: int,
    scenario: BMRBValidationScenario,
    participant_count: int,
    effect_scale: float,
    heterogeneity_scale: float,
    measurement_noise_scale: float,
    base_seed: int,
    rows: tuple[BMRBValidationReplicate, ...],
) -> OperatingCurveCellResult:
    if not rows:
        raise ValueError("operating-curve cell requires at least one replicate")
    passes = np.asarray([row.scientific_criteria_passed for row in rows], dtype=bool)
    observed_passes = int(np.count_nonzero(passes))
    pass_rate = float(observed_passes / len(rows))
    decision_error = (
        1.0 - pass_rate if scenario.expected_scientific_pass else pass_rate
    )
    monte_carlo_se = float(np.sqrt(pass_rate * (1.0 - pass_rate) / len(rows)))
    ci_lower, ci_upper = _wilson_interval(observed_passes, len(rows))
    biases = np.asarray([row.reference_effect_bias for row in rows], dtype=float)
    return OperatingCurveCellResult(
        cell_index=int(cell_index),
        scenario_id=scenario.scenario_id,
        truth_class=scenario.truth_class,
        expected_scientific_pass=scenario.expected_scientific_pass,
        expected_failure_component=scenario.expected_failure_component,
        participant_count=int(participant_count),
        effect_scale=float(effect_scale),
        heterogeneity_scale=float(heterogeneity_scale),
        measurement_noise_scale=float(measurement_noise_scale),
        base_seed=int(base_seed),
        replicates=len(rows),
        observed_passes=observed_passes,
        observed_pass_rate=pass_rate,
        decision_error_rate=float(decision_error),
        monte_carlo_se=monte_carlo_se,
        pass_rate_ci_lower=ci_lower,
        pass_rate_ci_upper=ci_upper,
        expected_failure_localization_rate=float(
            np.mean([row.expected_failure_localized for row in rows])
        ),
        mean_reference_effect_bias=float(np.mean(biases)),
        reference_effect_rmse=float(np.sqrt(np.mean(biases**2))),
        reference_ci_coverage=float(np.mean([row.reference_ci_covers_truth for row in rows])),
    )


@dataclass(frozen=True)
class BMRBOperatingCharacteristicsResult:
    policy: BMRBOperatingStudyPolicy
    cells: tuple[OperatingCurveCellResult, ...]

    def __post_init__(self) -> None:
        if len(self.cells) != self.policy.grid.cell_count:
            raise ValueError("operating result does not cover the complete frozen grid")
        indices = tuple(cell.cell_index for cell in self.cells)
        if indices != tuple(range(len(self.cells))):
            raise ValueError("operating-curve cell indices must be complete and ordered")

    def aggregate_mapping(self) -> dict[str, Any]:
        null_cells = tuple(cell for cell in self.cells if not cell.expected_scientific_pass)
        positive_cells = tuple(cell for cell in self.cells if cell.expected_scientific_pass)
        null_trials = sum(cell.replicates for cell in null_cells)
        positive_trials = sum(cell.replicates for cell in positive_cells)
        false_promotions = sum(cell.observed_passes for cell in null_cells)
        positive_recoveries = sum(cell.observed_passes for cell in positive_cells)
        worst = max(self.cells, key=lambda cell: cell.decision_error_rate)
        return {
            "false_promotion_rate": (
                None if null_trials == 0 else float(false_promotions / null_trials)
            ),
            "known_positive_recovery_rate": (
                None if positive_trials == 0 else float(positive_recoveries / positive_trials)
            ),
            "mean_cell_decision_error_rate": float(
                np.mean([cell.decision_error_rate for cell in self.cells])
            ),
            "mean_failure_localization_rate": float(
                np.mean([cell.expected_failure_localization_rate for cell in self.cells])
            ),
            "mean_reference_ci_coverage": float(
                np.mean([cell.reference_ci_coverage for cell in self.cells])
            ),
            "worst_cell": {
                "cell_index": worst.cell_index,
                "scenario_id": worst.scenario_id,
                "participant_count": worst.participant_count,
                "effect_scale": worst.effect_scale,
                "heterogeneity_scale": worst.heterogeneity_scale,
                "measurement_noise_scale": worst.measurement_noise_scale,
                "decision_error_rate": worst.decision_error_rate,
            },
        }

    def scientific_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "benchmark": BMRB_OPERATING_CHARACTERISTICS_BENCHMARK,
            "method": BMRB_OPERATING_CHARACTERISTICS_METHOD,
            "policy": self.policy.to_mapping(),
            "aggregate": self.aggregate_mapping(),
            "cells": [cell.to_mapping() for cell in self.cells],
            "qualification_defined": False,
            "interpretation": (
                "Operating characteristics quantify BMRB behavior under declared synthetic "
                "data-generating mechanisms and a frozen simulation authority. They do not "
                "validate biological truth, define universal scientific thresholds, or "
                "authorize a physical-quantum interpretation."
            ),
        }

    @property
    def artifact_fingerprint(self) -> str:
        return _scientific_fingerprint(
            "quantumbci.bmrb-operating-result.v1",
            self.scientific_payload(),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            **self.scientific_payload(),
            "artifact_fingerprint": self.artifact_fingerprint,
        }


def run_bmrb_operating_characteristics(
    policy: BMRBOperatingStudyPolicy,
) -> BMRBOperatingCharacteristicsResult:
    """Execute every frozen grid cell against production BMRB confirmatory semantics."""

    scenarios = {scenario.scenario_id: scenario for scenario in default_validation_scenarios()}
    unknown = sorted(set(policy.grid.scenario_ids) - set(scenarios))
    if unknown:
        raise ValueError(f"unknown operating-grid scenario ids: {unknown}")

    cells: list[OperatingCurveCellResult] = []
    combinations = product(
        policy.grid.scenario_ids,
        policy.grid.participant_counts,
        policy.grid.effect_scales,
        policy.grid.heterogeneity_scales,
        policy.grid.measurement_noise_scales,
    )
    for cell_index, (
        scenario_id,
        participant_count,
        effect_scale,
        heterogeneity_scale,
        measurement_noise_scale,
    ) in enumerate(combinations):
        scenario = _scaled_scenario(
            scenarios[scenario_id],
            effect_scale=effect_scale,
            heterogeneity_scale=heterogeneity_scale,
            measurement_noise_scale=measurement_noise_scale,
        )
        base_seed = policy.seed_partition.base_seed(
            policy.partition,
            cell_index=cell_index,
        )
        rows = tuple(
            run_validation_replicate(
                scenario,
                replicate=replicate,
                seed=base_seed,
                participants=participant_count,
                primary_calibration_per_class=policy.primary_calibration_per_class,
                bootstrap_resamples=policy.bootstrap_resamples,
            )
            for replicate in range(policy.replicates_per_cell)
        )
        cells.append(
            _summarize_cell(
                cell_index=cell_index,
                scenario=scenario,
                participant_count=participant_count,
                effect_scale=effect_scale,
                heterogeneity_scale=heterogeneity_scale,
                measurement_noise_scale=measurement_noise_scale,
                base_seed=base_seed,
                rows=rows,
            )
        )
    return BMRBOperatingCharacteristicsResult(policy=policy, cells=tuple(cells))


def write_bmrb_operating_characteristics(
    result: BMRBOperatingCharacteristicsResult,
    output: str | Path,
) -> Path:
    """Write one fingerprinted operating-characteristics artifact as canonical JSON."""

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(result.to_mapping()) + "\n", encoding="utf-8")
    return path