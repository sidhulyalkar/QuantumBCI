"""Known-truth operating characteristics for BMRB study-level replication.

This validates the software hierarchy, not biology. Every simulated study first runs
through the production participant-level BMRB known-truth evaluator. Exactly one bounded
study evidence object then enters the production study-replication layer, whose completed
decision enters the production sensitivity layer. Participant rows are never pooled across studies.

Only the development seed partition is executable in v1. The evaluation partition is
fingerprinted but remains sealed and unexecuted.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Any, Literal, Sequence

import numpy as np

from .bmrb_study_replication import (
    BMRBStudyEvidence,
    BMRBStudyReplicationPolicy,
    BMRBStudyReplicationSlot,
    evaluate_study_replication,
)
from .bmrb_study_sensitivity import BMRBStudySensitivityPolicy, assess_study_sensitivity
from .bmrb_validation import (
    BMRBValidationReplicate,
    BMRBValidationScenario,
    default_validation_scenarios,
    run_validation_replicate,
)
from .preregistration import PreregistrationEvidence, canonical_scientific_fingerprint

BMRB_STUDY_OPERATING_BENCHMARK = "BMRB_STUDY_KNOWN_TRUTH_OPERATING_CURVES_V1"
BMRB_STUDY_OPERATING_METHOD = "hierarchical_frozen_grid_monte_carlo_v1"
Partition = Literal["development", "evaluation"]


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if number < 1 or number != value:
        raise ValueError(f"{name} must be a positive integer")
    return number


def _nonnegative(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return number


def _required_text(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _hex_identity(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if trials < 1:
        raise ValueError("Wilson interval requires at least one trial")
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * trials)) / trials) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


@dataclass(frozen=True)
class StudySimulationSeedPartition:
    """Disjoint deterministic seed authority for cells, replicates, and studies."""

    development_offset: int = 31_000_000
    evaluation_offset: int = 2_031_000_000
    cell_stride: int = 1_000_000
    replicate_stride: int = 10_000
    study_stride: int = 101
    max_cells: int = 1024
    max_replicates_per_cell: int = 64
    max_studies_per_replicate: int = 16

    def __post_init__(self) -> None:
        for name in (
            "development_offset",
            "evaluation_offset",
            "cell_stride",
            "replicate_stride",
            "study_stride",
            "max_cells",
            "max_replicates_per_cell",
            "max_studies_per_replicate",
        ):
            object.__setattr__(self, name, _positive_int(name, getattr(self, name)))
        max_study_span = (self.max_studies_per_replicate - 1) * self.study_stride
        if self.replicate_stride <= max_study_span:
            raise ValueError("replicate_stride must exceed the maximum study seed span")
        max_replicate_span = (
            (self.max_replicates_per_cell - 1) * self.replicate_stride + max_study_span
        )
        if self.cell_stride <= max_replicate_span:
            raise ValueError("cell_stride must exceed the maximum replicate seed span")
        full_partition_span = (self.max_cells - 1) * self.cell_stride + max_replicate_span
        if abs(self.evaluation_offset - self.development_offset) <= full_partition_span:
            raise ValueError("development and evaluation seed spaces overlap")

    def effective_seed(
        self,
        partition: Partition,
        *,
        cell_index: int,
        replicate: int,
        study_index: int,
    ) -> int:
        if partition not in {"development", "evaluation"}:
            raise ValueError("partition must be development or evaluation")
        if not 0 <= cell_index < self.max_cells:
            raise ValueError("cell_index exceeds seed authority capacity")
        if not 0 <= replicate < self.max_replicates_per_cell:
            raise ValueError("replicate exceeds seed authority capacity")
        if not 0 <= study_index < self.max_studies_per_replicate:
            raise ValueError("study_index exceeds seed authority capacity")
        offset = self.development_offset if partition == "development" else self.evaluation_offset
        return int(
            offset
            + cell_index * self.cell_stride
            + replicate * self.replicate_stride
            + study_index * self.study_stride
        )

    @property
    def fingerprint(self) -> str:
        return canonical_scientific_fingerprint(
            "quantumbci.bmrb-study-operating-seeds.v1",
            {
                "development_offset": self.development_offset,
                "evaluation_offset": self.evaluation_offset,
                "cell_stride": self.cell_stride,
                "replicate_stride": self.replicate_stride,
                "study_stride": self.study_stride,
                "max_cells": self.max_cells,
                "max_replicates_per_cell": self.max_replicates_per_cell,
                "max_studies_per_replicate": self.max_studies_per_replicate,
            },
        )


@dataclass(frozen=True)
class BMRBStudyOperatingScenario:
    """One declared cross-study truth pattern."""

    scenario_id: str
    study_truths: tuple[str, ...]
    min_successful_replications: int
    expected_replication_pass: bool
    expected_context_specific_only: bool
    expected_sensitivity_warning: bool
    rationale: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_id", _required_text("scenario_id", self.scenario_id))
        truths = tuple(_required_text("study truth", item) for item in self.study_truths)
        object.__setattr__(self, "study_truths", truths)
        if len(truths) < 3:
            raise ValueError("study operating scenarios require at least three studies")
        unknown = sorted(set(truths) - {"positive", "null", "reversal"})
        if unknown:
            raise ValueError(f"unknown study truth labels: {unknown}")
        minimum = _positive_int("min_successful_replications", self.min_successful_replications)
        if minimum > len(truths) - 1:
            raise ValueError("min_successful_replications exceeds replication count")
        object.__setattr__(self, "min_successful_replications", minimum)
        for name in (
            "expected_replication_pass",
            "expected_context_specific_only",
            "expected_sensitivity_warning",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        object.__setattr__(self, "rationale", _required_text("rationale", self.rationale))

    @property
    def study_count(self) -> int:
        return len(self.study_truths)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "study_truths": list(self.study_truths),
            "study_count": self.study_count,
            "min_successful_replications": self.min_successful_replications,
            "expected_replication_pass": self.expected_replication_pass,
            "expected_context_specific_only": self.expected_context_specific_only,
            "expected_sensitivity_warning": self.expected_sensitivity_warning,
            "rationale": self.rationale,
        }


def default_study_operating_scenarios() -> tuple[BMRBStudyOperatingScenario, ...]:
    """Declared higher-level truths; all minima are software fixtures, not biological defaults."""

    return (
        BMRBStudyOperatingScenario(
            "homogeneous-positive-3",
            ("positive", "positive", "positive"),
            2,
            True,
            False,
            True,
            "All studies are positive, but requiring both replications leaves zero redundancy margin.",
        ),
        BMRBStudyOperatingScenario(
            "homogeneous-null-3",
            ("null", "null", "null"),
            2,
            False,
            False,
            False,
            "All three studies carry a declared effect-null pattern without heterogeneity.",
        ),
        BMRBStudyOperatingScenario(
            "homogeneous-positive-4",
            ("positive", "positive", "positive", "positive"),
            2,
            True,
            False,
            False,
            "Four-study positive control with one redundant replication beyond the minimum.",
        ),
        BMRBStudyOperatingScenario(
            "homogeneous-null-4",
            ("null", "null", "null", "null"),
            2,
            False,
            False,
            False,
            "Four independent effect-null studies must not promote and need not warn for heterogeneity.",
        ),
        BMRBStudyOperatingScenario(
            "primary-only-positive-4",
            ("positive", "null", "null", "null"),
            2,
            False,
            True,
            True,
            "A positive primary remains visible but cannot become a broad replication claim.",
        ),
        BMRBStudyOperatingScenario(
            "primary-fail-replications-positive-4",
            ("null", "positive", "positive", "positive"),
            2,
            False,
            True,
            True,
            "Successful replications cannot retroactively replace the frozen failed primary.",
        ),
        BMRBStudyOperatingScenario(
            "fragile-one-conflict-4",
            ("positive", "positive", "positive", "reversal"),
            2,
            True,
            False,
            True,
            "A zero-margin broad PASS with one directional context reversal should warn.",
        ),
        BMRBStudyOperatingScenario(
            "redundant-one-conflict-5",
            ("positive", "positive", "positive", "positive", "reversal"),
            2,
            True,
            False,
            True,
            "A broad PASS can have positive replication margin and still warrant a heterogeneity warning.",
        ),
    )


@dataclass(frozen=True)
class BMRBStudyOperatingGrid:
    scenario_ids: tuple[str, ...]
    participant_counts: tuple[int, ...]
    within_study_heterogeneity_scales: tuple[float, ...]
    measurement_noise_scales: tuple[float, ...]
    cross_study_effect_scales: tuple[float, ...]

    def __post_init__(self) -> None:
        scenario_ids = tuple(_required_text("scenario_id", item) for item in self.scenario_ids)
        if not scenario_ids or len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("scenario_ids must be non-empty and unique")
        object.__setattr__(self, "scenario_ids", scenario_ids)
        participants = tuple(_positive_int("participant_count", item) for item in self.participant_counts)
        if not participants or any(item < 4 for item in participants):
            raise ValueError("participant counts must be non-empty and at least four")
        object.__setattr__(self, "participant_counts", participants)
        for name in (
            "within_study_heterogeneity_scales",
            "measurement_noise_scales",
            "cross_study_effect_scales",
        ):
            values = tuple(_nonnegative(name, item) for item in getattr(self, name))
            if not values:
                raise ValueError(f"{name} must not be empty")
            object.__setattr__(self, name, values)

    @property
    def cell_count(self) -> int:
        return (
            len(self.scenario_ids)
            * len(self.participant_counts)
            * len(self.within_study_heterogeneity_scales)
            * len(self.measurement_noise_scales)
            * len(self.cross_study_effect_scales)
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "scenario_ids": list(self.scenario_ids),
            "participant_counts": list(self.participant_counts),
            "within_study_heterogeneity_scales": list(self.within_study_heterogeneity_scales),
            "measurement_noise_scales": list(self.measurement_noise_scales),
            "cross_study_effect_scales": list(self.cross_study_effect_scales),
            "cell_count": self.cell_count,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_scientific_fingerprint(
            "quantumbci.bmrb-study-operating-grid.v1", self.to_mapping()
        )


def qualification_smoke_grid() -> BMRBStudyOperatingGrid:
    return BMRBStudyOperatingGrid(
        scenario_ids=tuple(item.scenario_id for item in default_study_operating_scenarios()),
        participant_counts=(8,),
        within_study_heterogeneity_scales=(0.0,),
        measurement_noise_scales=(0.0,),
        cross_study_effect_scales=(0.0,),
    )


def recommended_development_grid() -> BMRBStudyOperatingGrid:
    return BMRBStudyOperatingGrid(
        scenario_ids=tuple(item.scenario_id for item in default_study_operating_scenarios()),
        participant_counts=(4, 8, 16),
        within_study_heterogeneity_scales=(0.5, 1.0, 2.0),
        measurement_noise_scales=(0.5, 1.0, 2.0),
        cross_study_effect_scales=(0.0, 1.0, 2.0),
    )


@dataclass(frozen=True)
class BMRBStudyOperatingPolicy:
    study_id: str
    source_sha: str
    partition: Partition
    grid: BMRBStudyOperatingGrid
    replicates_per_cell: int = 8
    bootstrap_resamples: int = 100
    seed_partition: StudySimulationSeedPartition = StudySimulationSeedPartition()
    sensitivity_min_direction_agreement: float = 0.75
    sensitivity_max_effect_range: float = 0.08
    sensitivity_max_leave_one_out_mean_shift: float = 0.04

    def __post_init__(self) -> None:
        object.__setattr__(self, "study_id", _required_text("study_id", self.study_id))
        object.__setattr__(self, "source_sha", _required_text("source_sha", self.source_sha))
        if self.partition not in {"development", "evaluation"}:
            raise ValueError("partition must be development or evaluation")
        reps = _positive_int("replicates_per_cell", self.replicates_per_cell)
        if reps > self.seed_partition.max_replicates_per_cell:
            raise ValueError("replicates_per_cell exceeds seed authority capacity")
        if self.grid.cell_count > self.seed_partition.max_cells:
            raise ValueError("operating grid exceeds seed authority cell capacity")
        object.__setattr__(self, "replicates_per_cell", reps)
        object.__setattr__(
            self,
            "bootstrap_resamples",
            _positive_int("bootstrap_resamples", self.bootstrap_resamples),
        )
        direction = float(self.sensitivity_min_direction_agreement)
        if not math.isfinite(direction) or not 0.0 <= direction <= 1.0:
            raise ValueError("sensitivity_min_direction_agreement must lie in [0, 1]")
        object.__setattr__(self, "sensitivity_min_direction_agreement", direction)
        for name in (
            "sensitivity_max_effect_range",
            "sensitivity_max_leave_one_out_mean_shift",
        ):
            object.__setattr__(self, name, _nonnegative(name, getattr(self, name)))

    def decision_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "benchmark": BMRB_STUDY_OPERATING_BENCHMARK,
            "method": BMRB_STUDY_OPERATING_METHOD,
            "study_id": self.study_id,
            "source_sha": self.source_sha,
            "partition": self.partition,
            "grid": self.grid.to_mapping(),
            "grid_fingerprint": self.grid.fingerprint,
            "replicates_per_cell": self.replicates_per_cell,
            "bootstrap_resamples": self.bootstrap_resamples,
            "seed_partition_fingerprint": self.seed_partition.fingerprint,
            "sensitivity_min_direction_agreement": self.sensitivity_min_direction_agreement,
            "sensitivity_max_effect_range": self.sensitivity_max_effect_range,
            "sensitivity_max_leave_one_out_mean_shift": self.sensitivity_max_leave_one_out_mean_shift,
            "evaluation_partition_executable": False,
        }

    @property
    def policy_fingerprint(self) -> str:
        return canonical_scientific_fingerprint(
            "quantumbci.bmrb-study-operating-policy.v1", self.decision_payload()
        )

    def to_mapping(self) -> dict[str, Any]:
        return {**self.decision_payload(), "policy_fingerprint": self.policy_fingerprint}


@dataclass(frozen=True)
class BMRBStudyOperatingReplicate:
    scenario_id: str
    cell_index: int
    replicate: int
    replication_criteria_passed: bool
    context_specific_only: bool
    sensitivity_warning: bool
    single_successful_replication_removal_flips_claim: bool
    successful_replication_margin: int
    study_passes: tuple[bool, ...]
    study_effects: tuple[float, ...]


@dataclass(frozen=True)
class BMRBStudyOperatingCell:
    scenario_id: str
    study_count: int
    participant_count: int
    within_study_heterogeneity_scale: float
    measurement_noise_scale: float
    cross_study_effect_scale: float
    expected_replication_pass: bool
    expected_context_specific_only: bool
    expected_sensitivity_warning: bool
    replicates: int
    observed_replication_pass_rate: float
    decision_error_rate: float
    context_specific_match_rate: float
    sensitivity_warning_match_rate: float
    primary_role_protection_rate: float
    fragile_claim_detection_rate: float
    pass_rate_ci_lower: float
    pass_rate_ci_upper: float
    mean_successful_replication_margin: float
    mean_study_effect_range: float

    def to_mapping(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class BMRBStudyOperatingResult:
    policy: BMRBStudyOperatingPolicy
    cells: tuple[BMRBStudyOperatingCell, ...]

    def aggregate_mapping(self) -> dict[str, Any]:
        negative = [cell for cell in self.cells if not cell.expected_replication_pass]
        positive = [cell for cell in self.cells if cell.expected_replication_pass]
        warning = [cell for cell in self.cells if cell.expected_sensitivity_warning]
        no_warning = [cell for cell in self.cells if not cell.expected_sensitivity_warning]
        return {
            "mean_false_promotion_rate": float(
                np.mean([cell.observed_replication_pass_rate for cell in negative])
            ),
            "mean_known_positive_recovery_rate": float(
                np.mean([cell.observed_replication_pass_rate for cell in positive])
            ),
            "mean_context_semantics_match_rate": float(
                np.mean([cell.context_specific_match_rate for cell in self.cells])
            ),
            "mean_expected_warning_match_rate": float(
                np.mean([cell.sensitivity_warning_match_rate for cell in warning])
            ),
            "mean_expected_no_warning_match_rate": float(
                np.mean([cell.sensitivity_warning_match_rate for cell in no_warning])
            ),
            "qualification_defined": False,
        }

    def to_mapping(self) -> dict[str, Any]:
        core = {
            "schema_version": 1,
            "benchmark": BMRB_STUDY_OPERATING_BENCHMARK,
            "method": BMRB_STUDY_OPERATING_METHOD,
            "policy": self.policy.to_mapping(),
            "cells": [cell.to_mapping() for cell in self.cells],
            "aggregate": self.aggregate_mapping(),
            "qualification_defined": False,
            "evaluation_partition_executed": False,
            "physical_quantum_promotion_eligible": False,
            "interpretation": (
                "Known-truth study-level simulation validates hierarchical BMRB software behavior. "
                "It does not validate biological truth, define a universal replication threshold, "
                "or authorize physical-quantum claims."
            ),
        }
        return {
            **core,
            "artifact_fingerprint": canonical_scientific_fingerprint(
                "quantumbci.bmrb-study-operating-result.v1", core
            ),
        }


def _truth_scenario(label: str) -> BMRBValidationScenario:
    base = {item.scenario_id: item for item in default_validation_scenarios()}
    if label == "positive":
        return base["shared-mechanism-positive"]
    if label == "null":
        return base["effect-null"]
    if label == "reversal":
        return BMRBValidationScenario(
            scenario_id="study-context-reversal",
            truth_class="adversarial",
            expected_scientific_pass=False,
            expected_failure_component="effect",
            reference_effect=-0.10,
            alternate_effect=-0.095,
            reference_ablation=0.10,
            alternate_ablation=0.095,
            information_novel=True,
            participant_effect_sd=0.015,
            measurement_sd=0.005,
        )
    raise ValueError(f"unknown study truth label: {label!r}")


def _scaled_truth_scenario(
    label: str,
    *,
    study_index: int,
    study_count: int,
    within_scale: float,
    measurement_scale: float,
    cross_study_scale: float,
) -> BMRBValidationScenario:
    scenario = _truth_scenario(label)
    center = (study_count - 1) / 2.0
    position = 0.0 if center == 0.0 else (study_index - center) / center
    effect_offset = 0.01 * cross_study_scale * position if label == "positive" else 0.0
    return replace(
        scenario,
        reference_effect=scenario.reference_effect + effect_offset,
        alternate_effect=scenario.alternate_effect + effect_offset,
        participant_effect_sd=scenario.participant_effect_sd * within_scale,
        measurement_sd=scenario.measurement_sd * measurement_scale,
    )


def _registered_replication_policy(
    scenario: BMRBStudyOperatingScenario,
    *,
    cell_index: int,
) -> BMRBStudyReplicationPolicy:
    mechanism_id = f"study-operating:{scenario.scenario_id}"
    provisional = BMRBStudyReplicationPolicy(
        policy_id=f"study-operating-replication:{scenario.scenario_id}:{cell_index}",
        mechanism_id=mechanism_id,
        studies=tuple(
            BMRBStudyReplicationSlot(
                study_id=f"study-{index}",
                dataset_id=f"dataset-{scenario.scenario_id}-{index}",
                role="primary" if index == 0 else "replication",
                order=index,
                rationale="Frozen known-truth study operating fixture.",
            )
            for index in range(scenario.study_count)
        ),
        min_successful_replications=scenario.min_successful_replications,
        scientific_rationale=(
            "Synthetic higher-level operating policy. The replication minimum is a "
            "scenario-specific software truth, not a universal biological threshold."
        ),
    )
    registration = PreregistrationEvidence(
        registration_uri=(
            f"https://osf.io/example/register/study-operating-{scenario.scenario_id}-{cell_index}"
        ),
        registered_at="2026-08-31T00:00:00Z",
        registration_document_sha256=_hex_identity(
            f"study-operating-registration:{scenario.scenario_id}:{cell_index}"
        ),
        registered_policy_sha256=provisional.decision_fingerprint,
        registry="synthetic-software-fixture",
    )
    return BMRBStudyReplicationPolicy(
        **{**provisional.__dict__, "preregistration": registration}
    )


def _sensitivity_policy(
    policy: BMRBStudyOperatingPolicy,
    *,
    cell_index: int,
) -> BMRBStudySensitivityPolicy:
    return BMRBStudySensitivityPolicy(
        policy_id=f"study-operating-sensitivity:{cell_index}",
        min_direction_agreement_fraction=policy.sensitivity_min_direction_agreement,
        max_effect_range=policy.sensitivity_max_effect_range,
        max_leave_one_out_mean_shift=policy.sensitivity_max_leave_one_out_mean_shift,
        scientific_rationale="Known-truth software thresholds for sensitivity calibration only.",
    )


def _study_evidence_from_validation(
    row: BMRBValidationReplicate,
    *,
    scenario: BMRBStudyOperatingScenario,
    study_index: int,
    participants: int,
    seed: int,
) -> BMRBStudyEvidence:
    passed = bool(row.scientific_criteria_passed)
    return BMRBStudyEvidence(
        study_id=f"study-{study_index}",
        dataset_id=f"dataset-{scenario.scenario_id}-{study_index}",
        mechanism_id=f"study-operating:{scenario.scenario_id}",
        participant_count=participants,
        scientific_criteria_passed=passed,
        confirmatory_authority=True,
        promotion_eligible=passed,
        reference_effect=row.reference_observed_effect,
        reference_ci_lower=row.reference_effect_ci_lower,
        reference_ci_upper=row.reference_effect_ci_upper,
        source_fingerprint=_hex_identity(
            f"study-operating-source:{scenario.scenario_id}:{study_index}:{seed}"
        ),
    )


def run_study_operating_replicate(
    policy: BMRBStudyOperatingPolicy,
    scenario: BMRBStudyOperatingScenario,
    *,
    cell_index: int,
    replicate: int,
    participants: int,
    within_scale: float,
    measurement_scale: float,
    cross_study_scale: float,
) -> BMRBStudyOperatingReplicate:
    """Run one outer Monte Carlo replicate through the complete BMRB hierarchy."""

    if policy.partition != "development":
        raise RuntimeError("study-level evaluation partition remains sealed in v1")
    replication_policy = _registered_replication_policy(scenario, cell_index=cell_index)
    evidence: list[BMRBStudyEvidence] = []
    for study_index, truth_label in enumerate(scenario.study_truths):
        seed = policy.seed_partition.effective_seed(
            policy.partition,
            cell_index=cell_index,
            replicate=replicate,
            study_index=study_index,
        )
        truth = _scaled_truth_scenario(
            truth_label,
            study_index=study_index,
            study_count=scenario.study_count,
            within_scale=within_scale,
            measurement_scale=measurement_scale,
            cross_study_scale=cross_study_scale,
        )
        row = run_validation_replicate(
            truth,
            replicate=replicate,
            seed=seed,
            participants=participants,
            bootstrap_resamples=policy.bootstrap_resamples,
        )
        evidence.append(
            _study_evidence_from_validation(
                row,
                scenario=scenario,
                study_index=study_index,
                participants=participants,
                seed=seed,
            )
        )
    replication_decision = evaluate_study_replication(replication_policy, tuple(evidence))
    sensitivity = assess_study_sensitivity(
        replication_decision,
        policy=_sensitivity_policy(policy, cell_index=cell_index),
    )
    return BMRBStudyOperatingReplicate(
        scenario_id=scenario.scenario_id,
        cell_index=cell_index,
        replicate=replicate,
        replication_criteria_passed=replication_decision.replication_criteria_passed,
        context_specific_only=replication_decision.context_specific_only,
        sensitivity_warning=sensitivity.sensitivity_warning,
        single_successful_replication_removal_flips_claim=(
            sensitivity.single_successful_replication_removal_flips_claim
        ),
        successful_replication_margin=sensitivity.successful_replication_margin,
        study_passes=tuple(item.scientific_criteria_passed for item in evidence),
        study_effects=tuple(item.reference_effect for item in evidence),
    )


def _summarize_cell(
    scenario: BMRBStudyOperatingScenario,
    rows: Sequence[BMRBStudyOperatingReplicate],
    *,
    participants: int,
    within_scale: float,
    measurement_scale: float,
    cross_study_scale: float,
) -> BMRBStudyOperatingCell:
    if not rows:
        raise ValueError("study operating cell requires replicates")
    pass_flags = [row.replication_criteria_passed for row in rows]
    context_matches = [
        row.context_specific_only == scenario.expected_context_specific_only for row in rows
    ]
    warning_matches = [
        row.sensitivity_warning == scenario.expected_sensitivity_warning for row in rows
    ]
    pass_rate = float(np.mean(pass_flags))
    decision_error = float(
        np.mean([flag != scenario.expected_replication_pass for flag in pass_flags])
    )
    lower, upper = _wilson_interval(sum(pass_flags), len(rows))
    primary_failed_later_positive = scenario.study_truths[0] == "null" and any(
        item == "positive" for item in scenario.study_truths[1:]
    )
    primary_role = (
        float(np.mean([not row.replication_criteria_passed for row in rows]))
        if primary_failed_later_positive
        else 1.0
    )
    if scenario.expected_sensitivity_warning and scenario.expected_replication_pass:
        fragile_detection = float(
            np.mean(
                [
                    row.sensitivity_warning
                    or row.single_successful_replication_removal_flips_claim
                    for row in rows
                ]
            )
        )
    else:
        fragile_detection = 1.0
    effect_ranges = [max(row.study_effects) - min(row.study_effects) for row in rows]
    return BMRBStudyOperatingCell(
        scenario_id=scenario.scenario_id,
        study_count=scenario.study_count,
        participant_count=participants,
        within_study_heterogeneity_scale=within_scale,
        measurement_noise_scale=measurement_scale,
        cross_study_effect_scale=cross_study_scale,
        expected_replication_pass=scenario.expected_replication_pass,
        expected_context_specific_only=scenario.expected_context_specific_only,
        expected_sensitivity_warning=scenario.expected_sensitivity_warning,
        replicates=len(rows),
        observed_replication_pass_rate=pass_rate,
        decision_error_rate=decision_error,
        context_specific_match_rate=float(np.mean(context_matches)),
        sensitivity_warning_match_rate=float(np.mean(warning_matches)),
        primary_role_protection_rate=primary_role,
        fragile_claim_detection_rate=fragile_detection,
        pass_rate_ci_lower=lower,
        pass_rate_ci_upper=upper,
        mean_successful_replication_margin=float(
            np.mean([row.successful_replication_margin for row in rows])
        ),
        mean_study_effect_range=float(np.mean(effect_ranges)),
    )


def run_bmrb_study_operating_characteristics(
    policy: BMRBStudyOperatingPolicy,
) -> BMRBStudyOperatingResult:
    """Run the frozen development grid. Evaluation seeds are intentionally inaccessible."""

    if policy.partition != "development":
        raise RuntimeError("study-level evaluation partition remains sealed in v1")
    scenarios = {item.scenario_id: item for item in default_study_operating_scenarios()}
    unknown = sorted(set(policy.grid.scenario_ids) - set(scenarios))
    if unknown:
        raise ValueError(f"unknown study operating scenario ids: {unknown}")

    cells: list[BMRBStudyOperatingCell] = []
    cell_index = 0
    for scenario_id in policy.grid.scenario_ids:
        scenario = scenarios[scenario_id]
        if scenario.study_count > policy.seed_partition.max_studies_per_replicate:
            raise ValueError("scenario study count exceeds seed authority capacity")
        for participants in policy.grid.participant_counts:
            for within_scale in policy.grid.within_study_heterogeneity_scales:
                for measurement_scale in policy.grid.measurement_noise_scales:
                    for cross_study_scale in policy.grid.cross_study_effect_scales:
                        rows = tuple(
                            run_study_operating_replicate(
                                policy,
                                scenario,
                                cell_index=cell_index,
                                replicate=replicate,
                                participants=participants,
                                within_scale=within_scale,
                                measurement_scale=measurement_scale,
                                cross_study_scale=cross_study_scale,
                            )
                            for replicate in range(policy.replicates_per_cell)
                        )
                        cells.append(
                            _summarize_cell(
                                scenario,
                                rows,
                                participants=participants,
                                within_scale=within_scale,
                                measurement_scale=measurement_scale,
                                cross_study_scale=cross_study_scale,
                            )
                        )
                        cell_index += 1
    if cell_index != policy.grid.cell_count:
        raise RuntimeError("study operating grid coverage mismatch")
    return BMRBStudyOperatingResult(policy=policy, cells=tuple(cells))
