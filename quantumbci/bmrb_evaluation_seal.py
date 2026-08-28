"""Preregisterable seals for frozen BMRB operating-characteristics evaluation.

This module deliberately does not execute the final evaluation partition. It defines a
machine-readable acceptance plan, binds that plan to development evidence and an exact
evaluation policy, and then requires an external preregistration record whose registered
policy hash matches the plan fingerprint.

Numeric acceptance thresholds are never supplied as scientific defaults by QuantumBCI.
They must be explicitly justified and provided before the evaluation seed partition is
observed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .bmrb_validation_operating import (
    BMRBOperatingStudyPolicy,
    OperatingCurveGrid,
    SimulationSeedPartition,
)
from .preregistration import PreregistrationEvidence, canonical_scientific_fingerprint

BMRB_EVALUATION_SEAL_ROLE = "bmrb_operating_evaluation_seal"
BMRB_EVALUATION_PLAN_METHOD = "preregistered_operating_acceptance_plan_v1"
BMRB_EVALUATION_SEAL_SCHEMA = 1

AGGREGATE_ACCEPTANCE_METRICS = frozenset(
    {
        "aggregate.false_promotion_rate",
        "aggregate.known_positive_recovery_rate",
        "aggregate.mean_cell_decision_error_rate",
        "aggregate.mean_failure_localization_rate",
        "aggregate.mean_reference_ci_coverage",
        "aggregate.worst_cell.decision_error_rate",
    }
)
SCENARIO_ACCEPTANCE_METRICS = frozenset(
    {
        "scenario.observed_pass_rate",
        "scenario.decision_error_rate",
        "scenario.expected_failure_localization_rate",
        "scenario.reference_ci_coverage",
    }
)
SCENARIO_REDUCERS = frozenset({"minimum", "mean", "maximum"})
REQUIRED_AGGREGATE_METRICS = frozenset(
    {
        "aggregate.false_promotion_rate",
        "aggregate.known_positive_recovery_rate",
    }
)
REQUIRED_SCENARIO_PASS_CRITERIA = frozenset(
    {
        "equivalence-null",
        "predictive-shortcut",
        "shared-mechanism-positive",
    }
)


def _required_text(name: str, value: Any) -> str:
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
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return number


def _required_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True)
class OperatingAcceptanceCriterion:
    """One explicitly justified bound in a future frozen evaluation decision rule."""

    criterion_id: str
    metric: str
    rationale: str
    lower_bound: float | None = None
    upper_bound: float | None = None
    scenario_id: str | None = None
    reducer: str = "identity"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "criterion_id",
            _required_text("criterion_id", self.criterion_id),
        )
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

        metric = self.metric
        if metric in AGGREGATE_ACCEPTANCE_METRICS:
            if self.scenario_id is not None:
                raise ValueError("aggregate acceptance criteria must not declare scenario_id")
            if self.reducer != "identity":
                raise ValueError("aggregate acceptance criteria must use reducer='identity'")
            return
        if metric not in SCENARIO_ACCEPTANCE_METRICS:
            raise ValueError(f"unsupported BMRB acceptance metric: {metric!r}")
        if self.scenario_id is None:
            raise ValueError("scenario acceptance criteria require scenario_id")
        object.__setattr__(
            self,
            "scenario_id",
            _required_text("scenario_id", self.scenario_id),
        )
        if self.reducer not in SCENARIO_REDUCERS:
            raise ValueError(
                "scenario acceptance criteria require reducer minimum, mean, or maximum"
            )

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
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OperatingAcceptanceCriterion":
        return cls(
            criterion_id=_required_text("criterion_id", payload.get("criterion_id")),
            metric=_required_text("metric", payload.get("metric")),
            rationale=_required_text("rationale", payload.get("rationale")),
            lower_bound=(
                None if payload.get("lower_bound") is None else float(payload["lower_bound"])
            ),
            upper_bound=(
                None if payload.get("upper_bound") is None else float(payload["upper_bound"])
            ),
            scenario_id=(
                None if payload.get("scenario_id") is None else str(payload["scenario_id"])
            ),
            reducer=str(payload.get("reducer", "identity")),
        )


def _operating_policy_from_mapping(payload: Mapping[str, Any]) -> BMRBOperatingStudyPolicy:
    grid_payload = _required_mapping("evaluation_policy.grid", payload.get("grid"))
    seed_payload = _required_mapping(
        "evaluation_policy.seed_partition",
        payload.get("seed_partition"),
    )
    grid = OperatingCurveGrid(
        scenario_ids=tuple(str(value) for value in grid_payload.get("scenario_ids", ())),
        participant_counts=tuple(int(value) for value in grid_payload.get("participant_counts", ())),
        effect_scales=tuple(float(value) for value in grid_payload.get("effect_scales", ())),
        heterogeneity_scales=tuple(
            float(value) for value in grid_payload.get("heterogeneity_scales", ())
        ),
        measurement_noise_scales=tuple(
            float(value) for value in grid_payload.get("measurement_noise_scales", ())
        ),
    )
    seed_partition = SimulationSeedPartition(
        development_offset=int(seed_payload.get("development_offset", 0)),
        evaluation_offset=int(seed_payload.get("evaluation_offset", 0)),
        cell_stride=int(seed_payload.get("cell_stride", 0)),
        replicate_stride=int(seed_payload.get("replicate_stride", 0)),
        max_replicates_per_cell=int(seed_payload.get("max_replicates_per_cell", 0)),
    )
    policy = BMRBOperatingStudyPolicy(
        study_id=_required_text("evaluation_policy.study_id", payload.get("study_id")),
        source_sha=_required_text("evaluation_policy.source_sha", payload.get("source_sha")),
        partition=str(payload.get("partition", "")),
        grid=grid,
        replicates_per_cell=int(payload.get("replicates_per_cell", 0)),
        bootstrap_resamples=int(payload.get("bootstrap_resamples", 0)),
        primary_calibration_per_class=int(payload.get("primary_calibration_per_class", -1)),
        seed_partition=seed_partition,
    )
    if policy.to_mapping() != dict(payload):
        raise ValueError("evaluation operating policy is not canonical or has stale fingerprints")
    return policy


@dataclass(frozen=True)
class BMRBOperatingAcceptancePlan:
    """Scientific decisions that must be frozen before final evaluation is run."""

    study_id: str
    development_evidence_ref: str
    development_artifact_fingerprint: str
    development_policy_fingerprint: str
    evaluation_policy: BMRBOperatingStudyPolicy
    criteria: tuple[OperatingAcceptanceCriterion, ...]
    multiplicity_policy: str
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
            _sha256(
                "development_artifact_fingerprint",
                self.development_artifact_fingerprint,
            ),
        )
        object.__setattr__(
            self,
            "development_policy_fingerprint",
            _sha256("development_policy_fingerprint", self.development_policy_fingerprint),
        )
        object.__setattr__(
            self,
            "multiplicity_policy",
            _required_text("multiplicity_policy", self.multiplicity_policy),
        )
        object.__setattr__(
            self,
            "scientific_rationale",
            _required_text("scientific_rationale", self.scientific_rationale),
        )
        if self.evaluation_policy.partition != "evaluation":
            raise ValueError("a final BMRB acceptance plan must bind partition='evaluation'")
        if not self.criteria:
            raise ValueError("a final BMRB acceptance plan requires explicit acceptance criteria")
        criteria = tuple(sorted(self.criteria, key=lambda criterion: criterion.criterion_id))
        if len({criterion.criterion_id for criterion in criteria}) != len(criteria):
            raise ValueError("acceptance criterion_id values must be unique")
        object.__setattr__(self, "criteria", criteria)

        grid_scenarios = set(self.evaluation_policy.grid.scenario_ids)
        missing_scenarios = sorted(REQUIRED_SCENARIO_PASS_CRITERIA - grid_scenarios)
        if missing_scenarios:
            raise ValueError(
                "final evaluation grid omits required known-truth scenarios: "
                f"{missing_scenarios}"
            )
        for criterion in criteria:
            if criterion.scenario_id is not None and criterion.scenario_id not in grid_scenarios:
                raise ValueError(
                    f"criterion {criterion.criterion_id!r} targets a scenario outside the "
                    "frozen evaluation grid"
                )

        aggregate_metrics = {
            criterion.metric
            for criterion in criteria
            if criterion.metric in AGGREGATE_ACCEPTANCE_METRICS
        }
        missing_aggregate = sorted(REQUIRED_AGGREGATE_METRICS - aggregate_metrics)
        if missing_aggregate:
            raise ValueError(
                "final evaluation plan is missing required aggregate acceptance metrics: "
                f"{missing_aggregate}"
            )
        scenario_pass_criteria = {
            criterion.scenario_id
            for criterion in criteria
            if criterion.metric == "scenario.observed_pass_rate"
        }
        missing_scenario_criteria = sorted(
            REQUIRED_SCENARIO_PASS_CRITERIA - scenario_pass_criteria
        )
        if missing_scenario_criteria:
            raise ValueError(
                "final evaluation plan must explicitly bound pass behavior for core known-truth "
                f"scenarios: {missing_scenario_criteria}"
            )

    def decision_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "method": BMRB_EVALUATION_PLAN_METHOD,
            "study_id": self.study_id,
            "development_evidence_ref": self.development_evidence_ref,
            "development_artifact_fingerprint": self.development_artifact_fingerprint,
            "development_policy_fingerprint": self.development_policy_fingerprint,
            "evaluation_policy": self.evaluation_policy.to_mapping(),
            "criteria": [criterion.to_mapping() for criterion in self.criteria],
            "multiplicity_policy": self.multiplicity_policy,
            "scientific_rationale": self.scientific_rationale,
            "evaluation_executed": False,
        }

    @property
    def plan_fingerprint(self) -> str:
        return canonical_scientific_fingerprint(
            "quantumbci.bmrb-operating-acceptance-plan.v1",
            self.decision_payload(),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {**self.decision_payload(), "plan_fingerprint": self.plan_fingerprint}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BMRBOperatingAcceptancePlan":
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("BMRB acceptance plan schema_version must be 1")
        if payload.get("method") != BMRB_EVALUATION_PLAN_METHOD:
            raise ValueError("BMRB acceptance plan method mismatch")
        if payload.get("evaluation_executed") is not False:
            raise ValueError("a preregistration plan must precede final evaluation execution")
        raw_criteria = payload.get("criteria")
        if not isinstance(raw_criteria, list):
            raise ValueError("BMRB acceptance plan criteria must be a list")
        evaluation_policy = _operating_policy_from_mapping(
            _required_mapping("evaluation_policy", payload.get("evaluation_policy"))
        )
        plan = cls(
            study_id=_required_text("study_id", payload.get("study_id")),
            development_evidence_ref=_required_text(
                "development_evidence_ref",
                payload.get("development_evidence_ref"),
            ),
            development_artifact_fingerprint=_sha256(
                "development_artifact_fingerprint",
                payload.get("development_artifact_fingerprint"),
            ),
            development_policy_fingerprint=_sha256(
                "development_policy_fingerprint",
                payload.get("development_policy_fingerprint"),
            ),
            evaluation_policy=evaluation_policy,
            criteria=tuple(
                OperatingAcceptanceCriterion.from_mapping(
                    _required_mapping("criterion", item)
                )
                for item in raw_criteria
            ),
            multiplicity_policy=_required_text(
                "multiplicity_policy",
                payload.get("multiplicity_policy"),
            ),
            scientific_rationale=_required_text(
                "scientific_rationale",
                payload.get("scientific_rationale"),
            ),
        )
        if plan.to_mapping() != dict(payload):
            raise ValueError("BMRB acceptance plan fingerprint mismatch or noncanonical payload")
        return plan


@dataclass(frozen=True)
class BMRBEvaluationSeal:
    """Externally preregistered, tamper-evident authorization for a future evaluation."""

    plan: BMRBOperatingAcceptancePlan
    preregistration: PreregistrationEvidence

    def __post_init__(self) -> None:
        if not self.preregistration.matches_policy(self.plan.plan_fingerprint):
            raise ValueError(
                "external preregistration does not bind the exact BMRB acceptance plan"
            )

    def scientific_payload(self) -> dict[str, Any]:
        return {
            "schema_version": BMRB_EVALUATION_SEAL_SCHEMA,
            "artifact_role": BMRB_EVALUATION_SEAL_ROLE,
            "plan": self.plan.to_mapping(),
            "preregistration": self.preregistration.to_mapping(),
            "evaluation_executed": False,
            "claim_boundary": (
                "This seal records a preregistered synthetic benchmark evaluation plan. It does "
                "not execute the evaluation, validate biological truth, establish neural causal "
                "necessity, or authorize a physical-quantum interpretation."
            ),
        }

    @property
    def artifact_fingerprint(self) -> str:
        return canonical_scientific_fingerprint(
            "quantumbci.bmrb-operating-evaluation-seal.v1",
            self.scientific_payload(),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            **self.scientific_payload(),
            "artifact_fingerprint": self.artifact_fingerprint,
        }


def verify_bmrb_evaluation_seal_mapping(payload: Mapping[str, Any]) -> BMRBEvaluationSeal:
    """Verify a serialized evaluation seal without running the evaluation partition."""

    if int(payload.get("schema_version", 0)) != BMRB_EVALUATION_SEAL_SCHEMA:
        raise ValueError("BMRB evaluation seal schema_version mismatch")
    if payload.get("artifact_role") != BMRB_EVALUATION_SEAL_ROLE:
        raise ValueError("artifact is not a BMRB operating evaluation seal")
    if payload.get("evaluation_executed") is not False:
        raise ValueError("a BMRB evaluation seal must be created before evaluation execution")
    claimed_artifact = _sha256("artifact_fingerprint", payload.get("artifact_fingerprint"))
    core = {key: value for key, value in payload.items() if key != "artifact_fingerprint"}
    expected_artifact = canonical_scientific_fingerprint(
        "quantumbci.bmrb-operating-evaluation-seal.v1",
        core,
    )
    if claimed_artifact != expected_artifact:
        raise ValueError("BMRB evaluation seal artifact fingerprint mismatch")

    plan = BMRBOperatingAcceptancePlan.from_mapping(
        _required_mapping("plan", payload.get("plan"))
    )
    preregistration = PreregistrationEvidence.from_mapping(
        _required_mapping("preregistration", payload.get("preregistration"))
    )
    seal = BMRBEvaluationSeal(plan=plan, preregistration=preregistration)
    if seal.to_mapping() != dict(payload):
        raise ValueError("BMRB evaluation seal is noncanonical or internally inconsistent")
    return seal


def write_bmrb_evaluation_seal(seal: BMRBEvaluationSeal, output: str | Path) -> Path:
    """Write one canonical evaluation-seal artifact without executing evaluation seeds."""

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(seal.to_mapping()) + "\n", encoding="utf-8")
    return path


def load_bmrb_evaluation_seal(path: str | Path) -> dict[str, Any]:
    """Load and verify one serialized BMRB evaluation seal."""

    artifact_path = Path(path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("BMRB evaluation seal must contain a JSON object")
    verify_bmrb_evaluation_seal_mapping(payload)
    return payload
