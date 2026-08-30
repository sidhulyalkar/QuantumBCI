"""Machine-readable authority for outcome-dependent candidate search.

A closed candidate family is not enough if the rule that decides which candidate to inspect next
can change after evidence is seen. This module freezes a conservative adaptive search protocol on
top of the single-family BMRB multiplicity authority.

The search protocol controls discovery/inspection only. It never transfers confirmatory promotion
authority away from the one primary candidate frozen by ``BMRBMultiplicityPlan``. Complete
candidate evidence is still required when the multiplicity decision is applied, so adaptive early
stopping cannot silently erase failed or uninspected hypotheses from the confirmatory record.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .bmrb_multiplicity import BMRBMultiplicityPlan, apply_multiplicity_plan
from .bmrb_validation import BMRBValidationReplicate
from .preregistration import canonical_scientific_fingerprint

BMRB_ADAPTIVE_SEARCH_METHOD = "predeclared_outcome_routed_search_v1"
BMRB_ADAPTIVE_SEARCH_RESULT_ROLE = "bmrb_adaptive_search_transcript_v1"


def _required_text(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _positive_int(name: str, value: Any) -> int:
    if type(value) is bool:
        raise ValueError(f"{name} must be a positive integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if integer < 1 or integer != value:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _finite_float(name: str, value: Any) -> float:
    if type(value) is bool:
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


@dataclass(frozen=True)
class BMRBAdaptiveSearchPlan:
    """Frozen outcome-routed inspection policy over one closed multiplicity family."""

    plan_id: str
    multiplicity_plan: BMRBMultiplicityPlan
    max_evaluations: int
    routing_effect_cutoff: float
    above_cutoff_stride: int
    below_cutoff_stride: int
    scientific_rationale: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _required_text("plan_id", self.plan_id))
        object.__setattr__(
            self,
            "scientific_rationale",
            _required_text("scientific_rationale", self.scientific_rationale),
        )
        maximum = _positive_int("max_evaluations", self.max_evaluations)
        if maximum > len(self.multiplicity_plan.candidates):
            raise ValueError("max_evaluations must not exceed the frozen candidate count")
        object.__setattr__(self, "max_evaluations", maximum)
        object.__setattr__(
            self,
            "routing_effect_cutoff",
            _finite_float("routing_effect_cutoff", self.routing_effect_cutoff),
        )

        above = _positive_int("above_cutoff_stride", self.above_cutoff_stride)
        below = _positive_int("below_cutoff_stride", self.below_cutoff_stride)
        if above == below:
            raise ValueError(
                "adaptive v1 requires distinct above/below strides so routing is outcome-dependent"
            )
        object.__setattr__(self, "above_cutoff_stride", above)
        object.__setattr__(self, "below_cutoff_stride", below)

    def decision_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "method": BMRB_ADAPTIVE_SEARCH_METHOD,
            "plan_id": self.plan_id,
            "multiplicity_plan": self.multiplicity_plan.to_mapping(),
            "start_candidate_id": self.multiplicity_plan.primary_candidate_id,
            "max_evaluations": self.max_evaluations,
            "routing_metric": "reference_observed_effect",
            "routing_effect_cutoff": self.routing_effect_cutoff,
            "above_cutoff_stride": self.above_cutoff_stride,
            "below_cutoff_stride": self.below_cutoff_stride,
            "collision_rule": "first_unvisited_from_preferred_index_circular",
            "stop_rule": "first_scientific_survivor_or_max_evaluations",
            "promotion_rule": "multiplicity_primary_only",
            "scientific_rationale": self.scientific_rationale,
        }

    @property
    def plan_fingerprint(self) -> str:
        return canonical_scientific_fingerprint(
            "quantumbci.bmrb-adaptive-search-plan.v1",
            self.decision_payload(),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {**self.decision_payload(), "plan_fingerprint": self.plan_fingerprint}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BMRBAdaptiveSearchPlan":
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("BMRB adaptive search schema_version must be 1")
        if payload.get("method") != BMRB_ADAPTIVE_SEARCH_METHOD:
            raise ValueError("BMRB adaptive search method mismatch")
        if payload.get("routing_metric") != "reference_observed_effect":
            raise ValueError("BMRB adaptive search routing_metric mismatch")
        if payload.get("collision_rule") != "first_unvisited_from_preferred_index_circular":
            raise ValueError("BMRB adaptive search collision_rule mismatch")
        if payload.get("stop_rule") != "first_scientific_survivor_or_max_evaluations":
            raise ValueError("BMRB adaptive search stop_rule mismatch")
        if payload.get("promotion_rule") != "multiplicity_primary_only":
            raise ValueError("BMRB adaptive search promotion_rule mismatch")
        raw_multiplicity = payload.get("multiplicity_plan")
        if not isinstance(raw_multiplicity, Mapping):
            raise ValueError("multiplicity_plan must be an object")
        multiplicity_plan = BMRBMultiplicityPlan.from_mapping(raw_multiplicity)
        if payload.get("start_candidate_id") != multiplicity_plan.primary_candidate_id:
            raise ValueError("adaptive search must start from the frozen primary candidate")
        plan = cls(
            plan_id=_required_text("plan_id", payload.get("plan_id")),
            multiplicity_plan=multiplicity_plan,
            max_evaluations=_positive_int(
                "max_evaluations", payload.get("max_evaluations", 0)
            ),
            routing_effect_cutoff=_finite_float(
                "routing_effect_cutoff", payload.get("routing_effect_cutoff")
            ),
            above_cutoff_stride=_positive_int(
                "above_cutoff_stride", payload.get("above_cutoff_stride", 0)
            ),
            below_cutoff_stride=_positive_int(
                "below_cutoff_stride", payload.get("below_cutoff_stride", 0)
            ),
            scientific_rationale=_required_text(
                "scientific_rationale", payload.get("scientific_rationale")
            ),
        )
        supplied = _required_text("plan_fingerprint", payload.get("plan_fingerprint"))
        if supplied != plan.plan_fingerprint:
            raise ValueError("BMRB adaptive search plan fingerprint mismatch")
        if plan.to_mapping() != dict(payload):
            raise ValueError("BMRB adaptive search plan is noncanonical")
        return plan


@dataclass(frozen=True)
class BMRBAdaptiveSearchStep:
    evaluation_index: int
    candidate_id: str
    reference_observed_effect: float
    scientific_criteria_passed: bool
    route: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "evaluation_index": self.evaluation_index,
            "candidate_id": self.candidate_id,
            "reference_observed_effect": self.reference_observed_effect,
            "scientific_criteria_passed": self.scientific_criteria_passed,
            "route": self.route,
        }


@dataclass(frozen=True)
class BMRBAdaptiveSearchTranscript:
    plan: BMRBAdaptiveSearchPlan
    steps: tuple[BMRBAdaptiveSearchStep, ...]
    naive_adaptive_survivor: bool
    authorized_primary_promotion: bool
    exhaustive_any_survivor: bool

    def to_mapping(self) -> dict[str, Any]:
        inspected = tuple(step.candidate_id for step in self.steps)
        first_survivor = next(
            (step.candidate_id for step in self.steps if step.scientific_criteria_passed),
            None,
        )
        return {
            "schema_version": 1,
            "artifact_role": BMRB_ADAPTIVE_SEARCH_RESULT_ROLE,
            "plan_fingerprint": self.plan.plan_fingerprint,
            "evaluations_used": len(self.steps),
            "max_evaluations": self.plan.max_evaluations,
            "inspected_candidate_ids": list(inspected),
            "first_adaptive_survivor": first_survivor,
            "naive_adaptive_survivor": self.naive_adaptive_survivor,
            "authorized_primary_promotion": self.authorized_primary_promotion,
            "exhaustive_any_survivor": self.exhaustive_any_survivor,
            "stopped_on_survivor": bool(first_survivor is not None),
            "steps": [step.to_mapping() for step in self.steps],
            "interpretation": (
                "The adaptive inspection path can react to observed evidence, but it cannot transfer "
                "promotion authority away from the multiplicity plan's predeclared primary."
            ),
            "physical_quantum_promotion_eligible": False,
        }


def _next_unvisited_index(
    *,
    current_index: int,
    stride: int,
    visited: set[int],
    candidate_count: int,
) -> int | None:
    preferred = (current_index + stride) % candidate_count
    for offset in range(candidate_count):
        candidate = (preferred + offset) % candidate_count
        if candidate not in visited:
            return candidate
    return None


def run_adaptive_search(
    plan: BMRBAdaptiveSearchPlan,
    evidence: Mapping[str, BMRBValidationReplicate],
) -> BMRBAdaptiveSearchTranscript:
    """Execute one frozen adaptive inspection plan over complete closed-world evidence.

    ``evidence`` must contain the full frozen candidate family. This function simulates which
    candidates an adaptive analyst would inspect and when they would stop, while the multiplicity
    authority still receives complete evidence for the confirmatory promotion decision.
    """

    if any(type(candidate_id) is not str for candidate_id in evidence):
        raise ValueError("adaptive search evidence keys must be strings")
    expected = set(plan.multiplicity_plan.candidate_ids)
    actual = set(evidence)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "adaptive search evidence must exactly match the frozen candidate family; "
            f"missing={missing} extra={extra}"
        )
    for candidate_id, replicate in evidence.items():
        if not isinstance(replicate, BMRBValidationReplicate):
            raise ValueError(
                f"adaptive search evidence for {candidate_id!r} must be BMRBValidationReplicate"
            )

    candidate_ids = plan.multiplicity_plan.candidate_ids
    index_by_id = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
    current_index = index_by_id[plan.multiplicity_plan.primary_candidate_id]
    visited: set[int] = set()
    steps: list[BMRBAdaptiveSearchStep] = []

    for evaluation_index in range(plan.max_evaluations):
        if current_index in visited:
            raise RuntimeError("adaptive search attempted to revisit a candidate")
        visited.add(current_index)
        candidate_id = candidate_ids[current_index]
        replicate = evidence[candidate_id]
        observed_effect = _finite_float(
            f"evidence[{candidate_id!r}].reference_observed_effect",
            replicate.reference_observed_effect,
        )
        above = observed_effect >= plan.routing_effect_cutoff
        route = "above_cutoff" if above else "below_cutoff"
        steps.append(
            BMRBAdaptiveSearchStep(
                evaluation_index=evaluation_index,
                candidate_id=candidate_id,
                reference_observed_effect=observed_effect,
                scientific_criteria_passed=replicate.scientific_criteria_passed,
                route=route,
            )
        )
        if replicate.scientific_criteria_passed:
            break
        stride = plan.above_cutoff_stride if above else plan.below_cutoff_stride
        next_index = _next_unvisited_index(
            current_index=current_index,
            stride=stride,
            visited=visited,
            candidate_count=len(candidate_ids),
        )
        if next_index is None:
            break
        current_index = next_index

    scientific_results = {
        candidate_id: replicate.scientific_criteria_passed
        for candidate_id, replicate in evidence.items()
    }
    multiplicity_decision = apply_multiplicity_plan(
        plan.multiplicity_plan,
        scientific_results,
    )
    return BMRBAdaptiveSearchTranscript(
        plan=plan,
        steps=tuple(steps),
        naive_adaptive_survivor=any(step.scientific_criteria_passed for step in steps),
        authorized_primary_promotion=multiplicity_decision.authorized_any_promotion,
        exhaustive_any_survivor=multiplicity_decision.naive_any_survivor,
    )
