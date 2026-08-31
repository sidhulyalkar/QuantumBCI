"""Preregisterable authority for a future BMRB study-level operating evaluation.

This module does not execute the study-level evaluation partition. It verifies development
operating evidence, binds an evaluation policy with identical scientific semantics, freezes
explicit acceptance criteria, and records machine-readable hierarchy, multiplicity, and
adaptive-search authority before an external preregistration can authorize future evaluation.

No numeric acceptance threshold is supplied as a scientific default. Sensitivity remains
non-promotion-authoritative, adaptive discovery never defines the confirmatory evidence set,
and this seal cannot authorize biological or physical-quantum claims.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .bmrb_adaptive_search import BMRBAdaptiveSearchPlan
from .bmrb_multiplicity import BMRBMultiplicityPlan
from .bmrb_study_operating import (
    BMRB_STUDY_OPERATING_BENCHMARK,
    BMRB_STUDY_OPERATING_METHOD,
    BMRBStudyOperatingGrid,
    BMRBStudyOperatingPolicy,
    StudySimulationSeedPartition,
    default_study_operating_scenarios,
)
from .bmrb_study_operating_artifacts import verify_bmrb_study_operating_mapping
from .bmrb_study_replication import BMRB_STUDY_REPLICATION_METHOD
from .bmrb_study_sensitivity import BMRB_STUDY_SENSITIVITY_METHOD
from .preregistration import PreregistrationEvidence, canonical_scientific_fingerprint

BMRB_STUDY_EVALUATION_PLAN_METHOD = "preregistered_study_operating_acceptance_plan_v1"
BMRB_STUDY_EVALUATION_SEAL_ROLE = "bmrb_study_operating_evaluation_seal_v1"
BMRB_STUDY_EVALUATION_SEAL_SCHEMA = 1
BMRB_STUDY_SEARCH_AUTHORITY_METHOD = "closed_family_search_authority_v1"
BMRB_STUDY_HIERARCHY_AUTHORITY_METHOD = "study_replication_sensitivity_authority_v1"

AGGREGATE_ACCEPTANCE_METRICS = frozenset(
    {
        "aggregate.mean_false_promotion_rate",
        "aggregate.mean_known_positive_recovery_rate",
        "aggregate.mean_context_semantics_match_rate",
        "aggregate.mean_expected_warning_match_rate",
        "aggregate.mean_expected_no_warning_match_rate",
    }
)
SCENARIO_ACCEPTANCE_METRICS = frozenset(
    {
        "scenario.observed_replication_pass_rate",
        "scenario.decision_error_rate",
        "scenario.context_specific_match_rate",
        "scenario.sensitivity_warning_match_rate",
        "scenario.primary_role_protection_rate",
        "scenario.fragile_claim_detection_rate",
    }
)
SCENARIO_REDUCERS = frozenset({"minimum", "mean", "maximum"})
REQUIRED_AGGREGATE_BOUNDS = {
    "aggregate.mean_false_promotion_rate": "upper",
    "aggregate.mean_known_positive_recovery_rate": "lower",
}
REQUIRED_SCENARIO_BOUNDS = {
    ("scenario.observed_replication_pass_rate", "homogeneous-positive-4"): "lower",
    ("scenario.observed_replication_pass_rate", "homogeneous-null-4"): "upper",
    ("scenario.primary_role_protection_rate", "primary-fail-replications-positive-4"): "lower",
    ("scenario.fragile_claim_detection_rate", "fragile-one-conflict-4"): "lower",
    ("scenario.sensitivity_warning_match_rate", "redundant-one-conflict-5"): "lower",
}


def _required_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _required_text(name: str, value: Any) -> str:
    if value is None:
        raise ValueError(f"{name} must not be empty")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _sha256(name: str, value: Any) -> str:
    text = _required_text(name, value).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a 64-character SHA-256 hexadecimal digest")
    return text


def _optional_fraction(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return number


def _seed_mapping(seed: StudySimulationSeedPartition) -> dict[str, Any]:
    core = {
        "schema_version": 1,
        "method": "nested_disjoint_study_seed_partitions_v1",
        "development_offset": seed.development_offset,
        "evaluation_offset": seed.evaluation_offset,
        "cell_stride": seed.cell_stride,
        "replicate_stride": seed.replicate_stride,
        "study_stride": seed.study_stride,
        "max_cells": seed.max_cells,
        "max_replicates_per_cell": seed.max_replicates_per_cell,
        "max_studies_per_replicate": seed.max_studies_per_replicate,
    }
    return {**core, "seed_partition_fingerprint": seed.fingerprint}


def _seed_from_mapping(payload: Mapping[str, Any]) -> StudySimulationSeedPartition:
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("study seed authority schema_version must be 1")
    if payload.get("method") != "nested_disjoint_study_seed_partitions_v1":
        raise ValueError("study seed authority method mismatch")
    seed = StudySimulationSeedPartition(
        development_offset=int(payload.get("development_offset", 0)),
        evaluation_offset=int(payload.get("evaluation_offset", 0)),
        cell_stride=int(payload.get("cell_stride", 0)),
        replicate_stride=int(payload.get("replicate_stride", 0)),
        study_stride=int(payload.get("study_stride", 0)),
        max_cells=int(payload.get("max_cells", 0)),
        max_replicates_per_cell=int(payload.get("max_replicates_per_cell", 0)),
        max_studies_per_replicate=int(payload.get("max_studies_per_replicate", 0)),
    )
    if _sha256(
        "seed_partition_fingerprint", payload.get("seed_partition_fingerprint")
    ) != seed.fingerprint:
        raise ValueError("study seed authority fingerprint mismatch")
    if _seed_mapping(seed) != dict(payload):
        raise ValueError("study seed authority is noncanonical")
    return seed


def _grid_from_mapping(payload: Mapping[str, Any]) -> BMRBStudyOperatingGrid:
    grid = BMRBStudyOperatingGrid(
        scenario_ids=tuple(str(value) for value in payload.get("scenario_ids", ())),
        participant_counts=tuple(int(value) for value in payload.get("participant_counts", ())),
        within_study_heterogeneity_scales=tuple(
            float(value) for value in payload.get("within_study_heterogeneity_scales", ())
        ),
        measurement_noise_scales=tuple(
            float(value) for value in payload.get("measurement_noise_scales", ())
        ),
        cross_study_effect_scales=tuple(
            float(value) for value in payload.get("cross_study_effect_scales", ())
        ),
    )
    if grid.to_mapping() != dict(payload):
        raise ValueError("study operating grid is noncanonical")
    return grid


def _policy_from_mapping(
    payload: Mapping[str, Any],
    *,
    seed_partition: StudySimulationSeedPartition,
) -> BMRBStudyOperatingPolicy:
    if payload.get("benchmark") != BMRB_STUDY_OPERATING_BENCHMARK:
        raise ValueError("study operating policy benchmark mismatch")
    if payload.get("method") != BMRB_STUDY_OPERATING_METHOD:
        raise ValueError("study operating policy method mismatch")
    supplied_seed_fingerprint = _sha256(
        "policy.seed_partition_fingerprint", payload.get("seed_partition_fingerprint")
    )
    if supplied_seed_fingerprint != seed_partition.fingerprint:
        raise ValueError(
            "policy.seed_partition_fingerprint does not match supplied seed authority"
        )
    grid = _grid_from_mapping(_required_mapping("policy.grid", payload.get("grid")))
    policy = BMRBStudyOperatingPolicy(
        study_id=_required_text("policy.study_id", payload.get("study_id")),
        source_sha=_required_text("policy.source_sha", payload.get("source_sha")),
        partition=str(payload.get("partition", "")),
        grid=grid,
        replicates_per_cell=int(payload.get("replicates_per_cell", 0)),
        bootstrap_resamples=int(payload.get("bootstrap_resamples", 0)),
        seed_partition=seed_partition,
        sensitivity_min_direction_agreement=float(
            payload.get("sensitivity_min_direction_agreement", math.nan)
        ),
        sensitivity_max_effect_range=float(payload.get("sensitivity_max_effect_range", math.nan)),
        sensitivity_max_leave_one_out_mean_shift=float(
            payload.get("sensitivity_max_leave_one_out_mean_shift", math.nan)
        ),
    )
    if policy.to_mapping() != dict(payload):
        raise ValueError("study operating policy is noncanonical or has stale fingerprints")
    return policy


def _normalized_policy_semantics(policy: BMRBStudyOperatingPolicy) -> dict[str, Any]:
    payload = policy.to_mapping()
    payload.pop("policy_fingerprint")
    payload["partition"] = "frozen-development-evaluation-pair"
    return payload


@dataclass(frozen=True)
class StudyOperatingAcceptanceCriterion:
    """One explicit future evaluation bound. No threshold defaults are supplied."""

    criterion_id: str
    metric: str
    rationale: str
    lower_bound: float | None = None
    upper_bound: float | None = None
    scenario_id: str | None = None
    reducer: str = "identity"

    def __post_init__(self) -> None:
        object.__setattr__(self, "criterion_id", _required_text("criterion_id", self.criterion_id))
        object.__setattr__(self, "metric", _required_text("metric", self.metric))
        object.__setattr__(self, "rationale", _required_text("rationale", self.rationale))
        lower = _optional_fraction("lower_bound", self.lower_bound)
        upper = _optional_fraction("upper_bound", self.upper_bound)
        if lower is None and upper is None:
            raise ValueError("an acceptance criterion must declare at least one bound")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("acceptance lower_bound must not exceed upper_bound")
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        if self.metric in AGGREGATE_ACCEPTANCE_METRICS:
            if self.scenario_id is not None:
                raise ValueError("aggregate acceptance criteria must not declare scenario_id")
            if self.reducer != "identity":
                raise ValueError("aggregate acceptance criteria must use reducer='identity'")
            return
        if self.metric not in SCENARIO_ACCEPTANCE_METRICS:
            raise ValueError(f"unsupported study operating acceptance metric: {self.metric!r}")
        object.__setattr__(self, "scenario_id", _required_text("scenario_id", self.scenario_id))
        if self.reducer not in SCENARIO_REDUCERS:
            raise ValueError("scenario criteria require reducer minimum, mean, or maximum")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "metric": self.metric,
            "scenario_id": self.scenario_id,
            "reducer": self.reducer,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "rationale": self.rationale,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StudyOperatingAcceptanceCriterion":
        return cls(
            criterion_id=_required_text("criterion_id", payload.get("criterion_id")),
            metric=_required_text("metric", payload.get("metric")),
            rationale=_required_text("rationale", payload.get("rationale")),
            lower_bound=None if payload.get("lower_bound") is None else float(payload["lower_bound"]),
            upper_bound=None if payload.get("upper_bound") is None else float(payload["upper_bound"]),
            scenario_id=None if payload.get("scenario_id") is None else str(payload["scenario_id"]),
            reducer=str(payload.get("reducer", "identity")),
        )


@dataclass(frozen=True)
class BMRBStudySearchAuthority:
    """Machine-readable multiplicity and optional adaptive discovery authority."""

    authority_id: str
    multiplicity_plan: BMRBMultiplicityPlan
    adaptive_search_mode: str
    scientific_rationale: str
    adaptive_search_plan: BMRBAdaptiveSearchPlan | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "authority_id", _required_text("authority_id", self.authority_id))
        object.__setattr__(
            self,
            "scientific_rationale",
            _required_text("scientific_rationale", self.scientific_rationale),
        )
        if self.adaptive_search_mode not in {"forbidden", "predeclared_plan"}:
            raise ValueError("adaptive_search_mode must be forbidden or predeclared_plan")
        if self.adaptive_search_mode == "forbidden":
            if self.adaptive_search_plan is not None:
                raise ValueError("forbidden adaptive search must not carry an adaptive plan")
        else:
            if self.adaptive_search_plan is None:
                raise ValueError("predeclared adaptive search requires an exact plan")
            if (
                self.adaptive_search_plan.multiplicity_plan.plan_fingerprint
                != self.multiplicity_plan.plan_fingerprint
            ):
                raise ValueError("adaptive and multiplicity plans must bind the same closed family")

    def decision_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "method": BMRB_STUDY_SEARCH_AUTHORITY_METHOD,
            "authority_id": self.authority_id,
            "multiplicity_plan": self.multiplicity_plan.to_mapping(),
            "adaptive_search_mode": self.adaptive_search_mode,
            "adaptive_search_plan": (
                None if self.adaptive_search_plan is None else self.adaptive_search_plan.to_mapping()
            ),
            "confirmatory_evidence_set": "complete_closed_multiplicity_family",
            "adaptive_discovery_defines_confirmatory_evidence_set": False,
            "promotion_rule": "multiplicity_predeclared_primary_only",
            "scientific_rationale": self.scientific_rationale,
        }

    @property
    def authority_fingerprint(self) -> str:
        return canonical_scientific_fingerprint(
            "quantumbci.bmrb-study-search-authority.v1", self.decision_payload()
        )

    def to_mapping(self) -> dict[str, Any]:
        return {**self.decision_payload(), "authority_fingerprint": self.authority_fingerprint}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BMRBStudySearchAuthority":
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("study search authority schema_version must be 1")
        if payload.get("method") != BMRB_STUDY_SEARCH_AUTHORITY_METHOD:
            raise ValueError("study search authority method mismatch")
        if payload.get("confirmatory_evidence_set") != "complete_closed_multiplicity_family":
            raise ValueError("confirmatory evidence set must remain the complete closed family")
        if payload.get("adaptive_discovery_defines_confirmatory_evidence_set") is not False:
            raise ValueError("adaptive discovery must never define the confirmatory evidence set")
        if payload.get("promotion_rule") != "multiplicity_predeclared_primary_only":
            raise ValueError("study search promotion rule mismatch")
        multiplicity = BMRBMultiplicityPlan.from_mapping(
            _required_mapping("multiplicity_plan", payload.get("multiplicity_plan"))
        )
        raw_adaptive = payload.get("adaptive_search_plan")
        adaptive = (
            None
            if raw_adaptive is None
            else BMRBAdaptiveSearchPlan.from_mapping(
                _required_mapping("adaptive_search_plan", raw_adaptive)
            )
        )
        authority = cls(
            authority_id=_required_text("authority_id", payload.get("authority_id")),
            multiplicity_plan=multiplicity,
            adaptive_search_mode=_required_text(
                "adaptive_search_mode", payload.get("adaptive_search_mode")
            ),
            adaptive_search_plan=adaptive,
            scientific_rationale=_required_text(
                "scientific_rationale", payload.get("scientific_rationale")
            ),
        )
        if (
            _sha256("authority_fingerprint", payload.get("authority_fingerprint"))
            != authority.authority_fingerprint
        ):
            raise ValueError("study search authority fingerprint mismatch")
        if authority.to_mapping() != dict(payload):
            raise ValueError("study search authority is noncanonical")
        return authority


@dataclass(frozen=True)
class BMRBStudyHierarchyAuthority:
    """Exact hierarchy contract consumed by the study operating benchmark."""

    scenario_contract_fingerprint: str
    sensitivity_min_direction_agreement: float
    sensitivity_max_effect_range: float
    sensitivity_max_leave_one_out_mean_shift: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scenario_contract_fingerprint",
            _sha256("scenario_contract_fingerprint", self.scenario_contract_fingerprint),
        )
        direction = _optional_fraction(
            "sensitivity_min_direction_agreement", self.sensitivity_min_direction_agreement
        )
        if direction is None:
            raise ValueError("sensitivity_min_direction_agreement is required")
        object.__setattr__(self, "sensitivity_min_direction_agreement", direction)
        for name in (
            "sensitivity_max_effect_range",
            "sensitivity_max_leave_one_out_mean_shift",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)

    @classmethod
    def from_operating_policy(
        cls, policy: BMRBStudyOperatingPolicy
    ) -> "BMRBStudyHierarchyAuthority":
        scenario_by_id = {
            scenario.scenario_id: scenario for scenario in default_study_operating_scenarios()
        }
        unknown = sorted(set(policy.grid.scenario_ids) - set(scenario_by_id))
        if unknown:
            raise ValueError(f"unknown study operating scenario ids: {unknown}")
        payload = [scenario_by_id[scenario_id].to_mapping() for scenario_id in policy.grid.scenario_ids]
        return cls(
            scenario_contract_fingerprint=canonical_scientific_fingerprint(
                "quantumbci.bmrb-study-operating-scenario-contract.v1",
                {"scenarios": payload},
            ),
            sensitivity_min_direction_agreement=policy.sensitivity_min_direction_agreement,
            sensitivity_max_effect_range=policy.sensitivity_max_effect_range,
            sensitivity_max_leave_one_out_mean_shift=policy.sensitivity_max_leave_one_out_mean_shift,
        )

    def decision_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "method": BMRB_STUDY_HIERARCHY_AUTHORITY_METHOD,
            "scenario_contract_fingerprint": self.scenario_contract_fingerprint,
            "replication_method": BMRB_STUDY_REPLICATION_METHOD,
            "primary_must_pass": True,
            "study_weighting": "one_independent_study_one_vote",
            "participant_weighting_role": "diagnostic_only",
            "sensitivity_method": BMRB_STUDY_SENSITIVITY_METHOD,
            "sensitivity_min_direction_agreement": self.sensitivity_min_direction_agreement,
            "sensitivity_max_effect_range": self.sensitivity_max_effect_range,
            "sensitivity_max_leave_one_out_mean_shift": self.sensitivity_max_leave_one_out_mean_shift,
            "sensitivity_promotion_authoritative": False,
        }

    @property
    def authority_fingerprint(self) -> str:
        return canonical_scientific_fingerprint(
            "quantumbci.bmrb-study-hierarchy-authority.v1", self.decision_payload()
        )

    def to_mapping(self) -> dict[str, Any]:
        return {**self.decision_payload(), "authority_fingerprint": self.authority_fingerprint}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BMRBStudyHierarchyAuthority":
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("study hierarchy authority schema_version must be 1")
        if payload.get("method") != BMRB_STUDY_HIERARCHY_AUTHORITY_METHOD:
            raise ValueError("study hierarchy authority method mismatch")
        if payload.get("replication_method") != BMRB_STUDY_REPLICATION_METHOD:
            raise ValueError("study hierarchy replication method mismatch")
        if payload.get("primary_must_pass") is not True:
            raise ValueError("study hierarchy must keep primary_must_pass=true")
        if payload.get("study_weighting") != "one_independent_study_one_vote":
            raise ValueError("study hierarchy weighting mismatch")
        if payload.get("participant_weighting_role") != "diagnostic_only":
            raise ValueError("participant weighting must remain diagnostic only")
        if payload.get("sensitivity_method") != BMRB_STUDY_SENSITIVITY_METHOD:
            raise ValueError("study hierarchy sensitivity method mismatch")
        if payload.get("sensitivity_promotion_authoritative") is not False:
            raise ValueError("study sensitivity v1 must remain non-promotion-authoritative")
        authority = cls(
            scenario_contract_fingerprint=_sha256(
                "scenario_contract_fingerprint", payload.get("scenario_contract_fingerprint")
            ),
            sensitivity_min_direction_agreement=float(
                payload.get("sensitivity_min_direction_agreement", math.nan)
            ),
            sensitivity_max_effect_range=float(
                payload.get("sensitivity_max_effect_range", math.nan)
            ),
            sensitivity_max_leave_one_out_mean_shift=float(
                payload.get("sensitivity_max_leave_one_out_mean_shift", math.nan)
            ),
        )
        if (
            _sha256("authority_fingerprint", payload.get("authority_fingerprint"))
            != authority.authority_fingerprint
        ):
            raise ValueError("study hierarchy authority fingerprint mismatch")
        if authority.to_mapping() != dict(payload):
            raise ValueError("study hierarchy authority is noncanonical")
        return authority


@dataclass(frozen=True)
class BMRBStudyOperatingAcceptancePlan:
    """Everything that must be frozen before a study-level final evaluation can be run."""

    study_id: str
    development_evidence_ref: str
    development_artifact_fingerprint: str
    development_policy: BMRBStudyOperatingPolicy
    evaluation_policy: BMRBStudyOperatingPolicy
    hierarchy_authority: BMRBStudyHierarchyAuthority
    search_authority: BMRBStudySearchAuthority
    criteria: tuple[StudyOperatingAcceptanceCriterion, ...]
    scientific_rationale: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "study_id", _required_text("study_id", self.study_id))
        object.__setattr__(
            self,
            "development_evidence_ref",
            _required_text("development_evidence_ref", self.development_evidence_ref),
        )
        object.__setattr__(
            self,
            "development_artifact_fingerprint",
            _sha256("development_artifact_fingerprint", self.development_artifact_fingerprint),
        )
        object.__setattr__(
            self,
            "scientific_rationale",
            _required_text("scientific_rationale", self.scientific_rationale),
        )
        if self.development_policy.partition != "development":
            raise ValueError("study evaluation plan requires a development policy")
        if self.evaluation_policy.partition != "evaluation":
            raise ValueError("study evaluation plan must bind partition='evaluation'")
        if (
            self.development_policy.seed_partition.fingerprint
            != self.evaluation_policy.seed_partition.fingerprint
        ):
            raise ValueError("development and evaluation must share one seed-partition authority")
        if _normalized_policy_semantics(self.development_policy) != _normalized_policy_semantics(
            self.evaluation_policy
        ):
            raise ValueError(
                "development and evaluation policies must have identical scientific semantics"
            )
        expected_hierarchy = BMRBStudyHierarchyAuthority.from_operating_policy(
            self.evaluation_policy
        )
        if self.hierarchy_authority.authority_fingerprint != expected_hierarchy.authority_fingerprint:
            raise ValueError("hierarchy authority does not match the frozen evaluation policy")

        criteria = tuple(sorted(self.criteria, key=lambda criterion: criterion.criterion_id))
        if not criteria:
            raise ValueError("study evaluation plan requires explicit acceptance criteria")
        if len({criterion.criterion_id for criterion in criteria}) != len(criteria):
            raise ValueError("study acceptance criterion_id values must be unique")
        object.__setattr__(self, "criteria", criteria)
        grid_scenarios = set(self.evaluation_policy.grid.scenario_ids)
        for criterion in criteria:
            if criterion.scenario_id is not None and criterion.scenario_id not in grid_scenarios:
                raise ValueError("study acceptance criterion targets a scenario outside evaluation grid")
        self._require_bounds(criteria)

    @staticmethod
    def _require_bounds(criteria: tuple[StudyOperatingAcceptanceCriterion, ...]) -> None:
        by_aggregate = {
            criterion.metric: criterion for criterion in criteria if criterion.scenario_id is None
        }
        for metric, direction in REQUIRED_AGGREGATE_BOUNDS.items():
            criterion = by_aggregate.get(metric)
            if criterion is None:
                raise ValueError(f"missing required aggregate acceptance metric: {metric}")
            if direction == "upper" and criterion.upper_bound is None:
                raise ValueError(f"{metric} requires an explicit upper bound")
            if direction == "lower" and criterion.lower_bound is None:
                raise ValueError(f"{metric} requires an explicit lower bound")
        by_scenario = {
            (criterion.metric, criterion.scenario_id): criterion
            for criterion in criteria
            if criterion.scenario_id is not None
        }
        for key, direction in REQUIRED_SCENARIO_BOUNDS.items():
            criterion = by_scenario.get(key)
            if criterion is None:
                raise ValueError(f"missing required scenario acceptance criterion: {key}")
            if direction == "upper" and criterion.upper_bound is None:
                raise ValueError(f"{key} requires an explicit upper bound")
            if direction == "lower" and criterion.lower_bound is None:
                raise ValueError(f"{key} requires an explicit lower bound")

    def decision_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "method": BMRB_STUDY_EVALUATION_PLAN_METHOD,
            "study_id": self.study_id,
            "development_evidence_ref": self.development_evidence_ref,
            "development_artifact_fingerprint": self.development_artifact_fingerprint,
            "seed_partition_authority": _seed_mapping(self.evaluation_policy.seed_partition),
            "development_policy": self.development_policy.to_mapping(),
            "evaluation_policy": self.evaluation_policy.to_mapping(),
            "hierarchy_authority": self.hierarchy_authority.to_mapping(),
            "search_authority": self.search_authority.to_mapping(),
            "criteria": [criterion.to_mapping() for criterion in self.criteria],
            "acceptance_criteria_frozen_before_evaluation": True,
            "evaluation_executed": False,
            "scientific_rationale": self.scientific_rationale,
        }

    @property
    def plan_fingerprint(self) -> str:
        return canonical_scientific_fingerprint(
            "quantumbci.bmrb-study-operating-acceptance-plan.v1",
            self.decision_payload(),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {**self.decision_payload(), "plan_fingerprint": self.plan_fingerprint}

    @classmethod
    def from_verified_development_artifact(
        cls,
        *,
        study_id: str,
        development_evidence_ref: str,
        development_artifact: Mapping[str, Any],
        evaluation_policy: BMRBStudyOperatingPolicy,
        hierarchy_authority: BMRBStudyHierarchyAuthority,
        search_authority: BMRBStudySearchAuthority,
        criteria: tuple[StudyOperatingAcceptanceCriterion, ...],
        scientific_rationale: str,
    ) -> "BMRBStudyOperatingAcceptancePlan":
        verify_bmrb_study_operating_mapping(development_artifact)
        development_policy = _policy_from_mapping(
            _required_mapping("development_artifact.policy", development_artifact.get("policy")),
            seed_partition=evaluation_policy.seed_partition,
        )
        return cls(
            study_id=study_id,
            development_evidence_ref=development_evidence_ref,
            development_artifact_fingerprint=_sha256(
                "development_artifact.artifact_fingerprint",
                development_artifact.get("artifact_fingerprint"),
            ),
            development_policy=development_policy,
            evaluation_policy=evaluation_policy,
            hierarchy_authority=hierarchy_authority,
            search_authority=search_authority,
            criteria=criteria,
            scientific_rationale=scientific_rationale,
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BMRBStudyOperatingAcceptancePlan":
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("study operating acceptance plan schema_version must be 1")
        if payload.get("method") != BMRB_STUDY_EVALUATION_PLAN_METHOD:
            raise ValueError("study operating acceptance plan method mismatch")
        if payload.get("acceptance_criteria_frozen_before_evaluation") is not True:
            raise ValueError("acceptance criteria must be frozen before evaluation")
        if payload.get("evaluation_executed") is not False:
            raise ValueError("acceptance plan must precede study-level evaluation execution")
        seed = _seed_from_mapping(
            _required_mapping("seed_partition_authority", payload.get("seed_partition_authority"))
        )
        development_policy = _policy_from_mapping(
            _required_mapping("development_policy", payload.get("development_policy")),
            seed_partition=seed,
        )
        evaluation_policy = _policy_from_mapping(
            _required_mapping("evaluation_policy", payload.get("evaluation_policy")),
            seed_partition=seed,
        )
        raw_criteria = payload.get("criteria")
        if not isinstance(raw_criteria, list):
            raise ValueError("study acceptance criteria must be a list")
        plan = cls(
            study_id=_required_text("study_id", payload.get("study_id")),
            development_evidence_ref=_required_text(
                "development_evidence_ref", payload.get("development_evidence_ref")
            ),
            development_artifact_fingerprint=_sha256(
                "development_artifact_fingerprint", payload.get("development_artifact_fingerprint")
            ),
            development_policy=development_policy,
            evaluation_policy=evaluation_policy,
            hierarchy_authority=BMRBStudyHierarchyAuthority.from_mapping(
                _required_mapping("hierarchy_authority", payload.get("hierarchy_authority"))
            ),
            search_authority=BMRBStudySearchAuthority.from_mapping(
                _required_mapping("search_authority", payload.get("search_authority"))
            ),
            criteria=tuple(
                StudyOperatingAcceptanceCriterion.from_mapping(
                    _required_mapping("criterion", item)
                )
                for item in raw_criteria
            ),
            scientific_rationale=_required_text(
                "scientific_rationale", payload.get("scientific_rationale")
            ),
        )
        if _sha256("plan_fingerprint", payload.get("plan_fingerprint")) != plan.plan_fingerprint:
            raise ValueError("study operating acceptance plan fingerprint mismatch")
        if plan.to_mapping() != dict(payload):
            raise ValueError("study operating acceptance plan is noncanonical")
        return plan


@dataclass(frozen=True)
class BMRBStudyEvaluationSeal:
    """Externally preregistered, tamper-evident authorization for future evaluation."""

    plan: BMRBStudyOperatingAcceptancePlan
    preregistration: PreregistrationEvidence

    def __post_init__(self) -> None:
        if not self.preregistration.matches_policy(self.plan.plan_fingerprint):
            raise ValueError("external preregistration does not bind the exact study evaluation plan")

    def scientific_payload(self) -> dict[str, Any]:
        return {
            "schema_version": BMRB_STUDY_EVALUATION_SEAL_SCHEMA,
            "artifact_role": BMRB_STUDY_EVALUATION_SEAL_ROLE,
            "plan": self.plan.to_mapping(),
            "preregistration": self.preregistration.to_mapping(),
            "evaluation_executed": False,
            "sensitivity_promotion_authoritative": False,
            "physical_quantum_promotion_eligible": False,
            "claim_boundary": (
                "This seal records a preregistered synthetic study-level benchmark evaluation plan. "
                "It does not execute evaluation, validate biological truth, establish universal "
                "replication or heterogeneity thresholds, create meta-analytic authority, or "
                "authorize a physical-quantum interpretation."
            ),
        }

    @property
    def artifact_fingerprint(self) -> str:
        return canonical_scientific_fingerprint(
            "quantumbci.bmrb-study-operating-evaluation-seal.v1",
            self.scientific_payload(),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {**self.scientific_payload(), "artifact_fingerprint": self.artifact_fingerprint}


def verify_bmrb_study_evaluation_seal_mapping(
    payload: Mapping[str, Any],
) -> BMRBStudyEvaluationSeal:
    """Verify a serialized study evaluation seal without running final evaluation."""

    if int(payload.get("schema_version", 0)) != BMRB_STUDY_EVALUATION_SEAL_SCHEMA:
        raise ValueError("study evaluation seal schema_version mismatch")
    if payload.get("artifact_role") != BMRB_STUDY_EVALUATION_SEAL_ROLE:
        raise ValueError("artifact is not a BMRB study operating evaluation seal")
    if payload.get("evaluation_executed") is not False:
        raise ValueError("study evaluation seal must precede evaluation execution")
    if payload.get("sensitivity_promotion_authoritative") is not False:
        raise ValueError("study sensitivity v1 must remain non-promotion-authoritative")
    if payload.get("physical_quantum_promotion_eligible") is not False:
        raise ValueError("study evaluation seal cannot authorize physical-quantum claims")
    claimed = _sha256("artifact_fingerprint", payload.get("artifact_fingerprint"))
    core = {key: value for key, value in payload.items() if key != "artifact_fingerprint"}
    expected = canonical_scientific_fingerprint(
        "quantumbci.bmrb-study-operating-evaluation-seal.v1", core
    )
    if claimed != expected:
        raise ValueError("study evaluation seal artifact fingerprint mismatch")
    plan = BMRBStudyOperatingAcceptancePlan.from_mapping(
        _required_mapping("plan", payload.get("plan"))
    )
    preregistration = PreregistrationEvidence.from_mapping(
        _required_mapping("preregistration", payload.get("preregistration"))
    )
    seal = BMRBStudyEvaluationSeal(plan=plan, preregistration=preregistration)
    if seal.to_mapping() != dict(payload):
        raise ValueError("study evaluation seal is noncanonical or internally inconsistent")
    return seal
