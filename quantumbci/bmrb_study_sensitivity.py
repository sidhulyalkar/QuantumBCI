"""Study-level heterogeneity and influence diagnostics for BMRB replication.

This v1 layer is deliberately sensitivity-only. It consumes a completed
``BMRBStudyReplicationDecision`` and quantifies cross-study directional agreement,
effect spread, leave-one-study-out influence, and replication-margin fragility.
It does not alter the already-qualified replication promotion decision.

Thresholds are explicit policy content rather than universal biological constants. A
future promotion-authoritative heterogeneity method requires its own method identifier,
preregistration contract, and known-truth operating-characteristic validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .bmrb_study_replication import BMRBStudyReplicationDecision
from .preregistration import PreregistrationEvidence, canonical_scientific_fingerprint

BMRB_STUDY_SENSITIVITY_METHOD = "study_effect_heterogeneity_leave_one_out_v1"
BMRB_STUDY_SENSITIVITY_RESULT_ROLE = "bmrb_study_sensitivity_assessment_v1"


def _required_text(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _fraction(name: str, value: Any) -> float:
    number = _finite(name, value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return number


def _nonnegative(name: str, value: Any) -> float:
    number = _finite(name, value)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


@dataclass(frozen=True)
class BMRBStudySensitivityPolicy:
    """Explicit thresholds for a non-promotion-authoritative sensitivity report."""

    policy_id: str
    min_direction_agreement_fraction: float
    max_effect_range: float
    max_leave_one_out_mean_shift: float
    scientific_rationale: str
    preregistration: PreregistrationEvidence | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _required_text("policy_id", self.policy_id))
        object.__setattr__(
            self,
            "scientific_rationale",
            _required_text("scientific_rationale", self.scientific_rationale),
        )
        object.__setattr__(
            self,
            "min_direction_agreement_fraction",
            _fraction(
                "min_direction_agreement_fraction", self.min_direction_agreement_fraction
            ),
        )
        object.__setattr__(
            self,
            "max_effect_range",
            _nonnegative("max_effect_range", self.max_effect_range),
        )
        object.__setattr__(
            self,
            "max_leave_one_out_mean_shift",
            _nonnegative(
                "max_leave_one_out_mean_shift", self.max_leave_one_out_mean_shift
            ),
        )

    def decision_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "method": BMRB_STUDY_SENSITIVITY_METHOD,
            "policy_id": self.policy_id,
            "min_direction_agreement_fraction": self.min_direction_agreement_fraction,
            "max_effect_range": self.max_effect_range,
            "max_leave_one_out_mean_shift": self.max_leave_one_out_mean_shift,
            "scientific_rationale": self.scientific_rationale,
            "promotion_authoritative": False,
        }

    @property
    def decision_fingerprint(self) -> str:
        return canonical_scientific_fingerprint(
            "quantumbci.bmrb-study-sensitivity-policy.v1", self.decision_payload()
        )

    @property
    def confirmatory_authority(self) -> bool:
        return bool(
            self.preregistration is not None
            and self.preregistration.matches_policy(self.decision_fingerprint)
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            **self.decision_payload(),
            "decision_fingerprint": self.decision_fingerprint,
            "preregistration": (
                None if self.preregistration is None else self.preregistration.to_mapping()
            ),
            "confirmatory_authority": self.confirmatory_authority,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BMRBStudySensitivityPolicy":
        if payload.get("method") not in (None, BMRB_STUDY_SENSITIVITY_METHOD):
            raise ValueError(f"unexpected study sensitivity method: {payload.get('method')!r}")
        if payload.get("promotion_authoritative") not in (None, False):
            raise ValueError("v1 study sensitivity is not promotion-authoritative")
        registration = payload.get("preregistration")
        if registration is not None and not isinstance(registration, Mapping):
            raise ValueError("preregistration must be an object or null")
        policy = cls(
            policy_id=_required_text("policy_id", payload.get("policy_id")),
            min_direction_agreement_fraction=_fraction(
                "min_direction_agreement_fraction",
                payload.get("min_direction_agreement_fraction"),
            ),
            max_effect_range=_nonnegative("max_effect_range", payload.get("max_effect_range")),
            max_leave_one_out_mean_shift=_nonnegative(
                "max_leave_one_out_mean_shift",
                payload.get("max_leave_one_out_mean_shift"),
            ),
            scientific_rationale=_required_text(
                "scientific_rationale", payload.get("scientific_rationale")
            ),
            preregistration=(
                None
                if registration is None
                else PreregistrationEvidence.from_mapping(registration)
            ),
        )
        supplied = payload.get("decision_fingerprint")
        if supplied is not None and str(supplied) != policy.decision_fingerprint:
            raise ValueError("decision_fingerprint does not match reconstructed policy")
        return policy


@dataclass(frozen=True)
class BMRBLeaveOneStudyOutPoint:
    removed_study_id: str
    remaining_study_count: int
    remaining_unweighted_effect_mean: float
    absolute_mean_shift: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "removed_study_id": self.removed_study_id,
            "remaining_study_count": self.remaining_study_count,
            "remaining_unweighted_effect_mean": self.remaining_unweighted_effect_mean,
            "absolute_mean_shift": self.absolute_mean_shift,
        }


@dataclass(frozen=True)
class BMRBStudySensitivityAssessment:
    policy: BMRBStudySensitivityPolicy
    replication: BMRBStudyReplicationDecision
    direction_agreement_fraction: float
    effect_range: float
    leave_one_study_out: tuple[BMRBLeaveOneStudyOutPoint, ...]
    max_leave_one_out_mean_shift: float
    most_influential_study_id: str
    successful_replication_margin: int
    single_successful_replication_removal_flips_claim: bool

    @property
    def heterogeneity_criteria_passed(self) -> bool:
        return bool(
            self.direction_agreement_fraction >= self.policy.min_direction_agreement_fraction
            and self.effect_range <= self.policy.max_effect_range
            and self.max_leave_one_out_mean_shift
            <= self.policy.max_leave_one_out_mean_shift
        )

    @property
    def sensitivity_warning(self) -> bool:
        return bool(
            not self.heterogeneity_criteria_passed
            or self.single_successful_replication_removal_flips_claim
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_role": BMRB_STUDY_SENSITIVITY_RESULT_ROLE,
            "method": BMRB_STUDY_SENSITIVITY_METHOD,
            "policy": self.policy.to_mapping(),
            "replication_policy_fingerprint": self.replication.policy.decision_fingerprint,
            "replication_criteria_passed": self.replication.replication_criteria_passed,
            "replication_broad_claim_promotion_eligible": (
                self.replication.broad_claim_promotion_eligible
            ),
            "direction_agreement_fraction": self.direction_agreement_fraction,
            "effect_range": self.effect_range,
            "leave_one_study_out": [item.to_mapping() for item in self.leave_one_study_out],
            "max_leave_one_out_mean_shift": self.max_leave_one_out_mean_shift,
            "most_influential_study_id": self.most_influential_study_id,
            "successful_replication_margin": self.successful_replication_margin,
            "single_successful_replication_removal_flips_claim": (
                self.single_successful_replication_removal_flips_claim
            ),
            "heterogeneity_criteria_passed": self.heterogeneity_criteria_passed,
            "sensitivity_warning": self.sensitivity_warning,
            "promotion_authoritative": False,
            "replication_promotion_decision_unchanged": True,
            "physical_quantum_promotion_eligible": False,
            "interpretation": (
                "This report quantifies study-level heterogeneity and influence. It does not "
                "change the qualified replication promotion decision."
            ),
        }


def assess_study_sensitivity(
    replication: BMRBStudyReplicationDecision,
    *,
    policy: BMRBStudySensitivityPolicy,
) -> BMRBStudySensitivityAssessment:
    """Quantify heterogeneity and single-study influence without changing promotion."""

    evidence = tuple(replication.evidence)
    if len(evidence) < 3:
        raise ValueError("study sensitivity v1 requires at least three independent studies")

    effects = [float(item.reference_effect) for item in evidence]
    full_mean = float(sum(effects) / len(effects))
    primary_sign = _sign(effects[0])
    direction_agreement = float(
        sum(_sign(effect) == primary_sign for effect in effects) / len(effects)
    )
    effect_range = float(max(effects) - min(effects))

    points: list[BMRBLeaveOneStudyOutPoint] = []
    for index, item in enumerate(evidence):
        remaining = effects[:index] + effects[index + 1 :]
        remaining_mean = float(sum(remaining) / len(remaining))
        points.append(
            BMRBLeaveOneStudyOutPoint(
                removed_study_id=item.study_id,
                remaining_study_count=len(remaining),
                remaining_unweighted_effect_mean=remaining_mean,
                absolute_mean_shift=abs(remaining_mean - full_mean),
            )
        )
    most_influential = max(points, key=lambda item: (item.absolute_mean_shift, item.removed_study_id))

    success_count = len(replication.successful_replication_studies)
    margin = success_count - replication.policy.min_successful_replications
    single_success_flip = bool(
        replication.replication_criteria_passed and success_count > 0 and margin == 0
    )

    return BMRBStudySensitivityAssessment(
        policy=policy,
        replication=replication,
        direction_agreement_fraction=direction_agreement,
        effect_range=effect_range,
        leave_one_study_out=tuple(points),
        max_leave_one_out_mean_shift=most_influential.absolute_mean_shift,
        most_influential_study_id=most_influential.removed_study_id,
        successful_replication_margin=margin,
        single_successful_replication_removal_flips_claim=single_success_flip,
    )
