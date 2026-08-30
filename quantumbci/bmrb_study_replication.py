"""Study-level replication authority for broad BMRB mechanism claims.

This layer consumes one finished confirmatory evidence object per independent study or
dataset. It never reopens participant rows and never treats participant count as the
number of replications. A large study can therefore improve its own within-study
precision without acquiring extra votes at the cross-study layer.

A failed broad-replication decision does not erase study-specific evidence. The result
keeps positive and failed studies visible so a mechanism may remain context-specific
without being promoted as broadly replicated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .confirmatory_representation import ConfirmatoryRepresentationResult
from .preregistration import PreregistrationEvidence, canonical_scientific_fingerprint

BMRB_STUDY_REPLICATION_METHOD = "primary_plus_predeclared_replications_equal_study_vote_v1"
BMRB_STUDY_REPLICATION_RESULT_ROLE = "bmrb_study_replication_decision_v1"
_STUDY_ROLES = frozenset({"primary", "replication"})


def _required_text(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _strict_bool(name: str, value: Any) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a JSON/Python boolean")
    return value


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


def _nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if number < 0 or number != value:
        raise ValueError(f"{name} must be a non-negative integer")
    return number


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


@dataclass(frozen=True)
class BMRBStudyReplicationSlot:
    """One predeclared independent study/dataset in the replication family."""

    study_id: str
    dataset_id: str
    role: str
    order: int
    rationale: str

    def __post_init__(self) -> None:
        for name in ("study_id", "dataset_id", "role", "rationale"):
            object.__setattr__(self, name, _required_text(name, getattr(self, name)))
        if self.role not in _STUDY_ROLES:
            raise ValueError(f"role must be one of {sorted(_STUDY_ROLES)}")
        object.__setattr__(self, "order", _nonnegative_int("order", self.order))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "dataset_id": self.dataset_id,
            "role": self.role,
            "order": self.order,
            "rationale": self.rationale,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BMRBStudyReplicationSlot":
        return cls(
            study_id=_required_text("study_id", payload.get("study_id")),
            dataset_id=_required_text("dataset_id", payload.get("dataset_id")),
            role=_required_text("role", payload.get("role")),
            order=_nonnegative_int("order", payload.get("order")),
            rationale=_required_text("rationale", payload.get("rationale")),
        )


@dataclass(frozen=True)
class BMRBStudyReplicationPolicy:
    """Predeclared authority for a broad cross-study mechanism claim.

    v1 is intentionally narrow: one primary study, one or more independent replication
    studies, equal study-level votes, and a declared minimum number of successful
    replications. Participant-count weighting is never promotion-authoritative.
    """

    policy_id: str
    mechanism_id: str
    studies: tuple[BMRBStudyReplicationSlot, ...]
    min_successful_replications: int
    scientific_rationale: str
    preregistration: PreregistrationEvidence | None = None

    def __post_init__(self) -> None:
        for name in ("policy_id", "mechanism_id", "scientific_rationale"):
            object.__setattr__(self, name, _required_text(name, getattr(self, name)))
        studies = tuple(self.studies)
        object.__setattr__(self, "studies", studies)
        if len(studies) < 2:
            raise ValueError("study replication requires one primary and at least one replication")
        study_ids = [item.study_id for item in studies]
        dataset_ids = [item.dataset_id for item in studies]
        if len(set(study_ids)) != len(study_ids):
            raise ValueError("study_id values must be unique")
        if len(set(dataset_ids)) != len(dataset_ids):
            raise ValueError("dataset_id values must be unique; one dataset cannot count twice")
        orders = sorted(item.order for item in studies)
        if orders != list(range(len(studies))):
            raise ValueError("study order must be contiguous from zero")
        primary = [item for item in studies if item.role == "primary"]
        if len(primary) != 1 or primary[0].order != 0:
            raise ValueError("v1 requires exactly one primary study at order zero")
        if any(item.role != "replication" for item in studies if item.order != 0):
            raise ValueError("all non-primary studies must have replication role")
        minimum = _positive_int("min_successful_replications", self.min_successful_replications)
        replication_count = len(studies) - 1
        if minimum > replication_count:
            raise ValueError("min_successful_replications exceeds frozen replication count")
        object.__setattr__(self, "min_successful_replications", minimum)

    @property
    def primary_study_id(self) -> str:
        return min(self.studies, key=lambda item: item.order).study_id

    @property
    def replication_study_ids(self) -> tuple[str, ...]:
        return tuple(
            item.study_id
            for item in sorted(self.studies, key=lambda item: item.order)
            if item.role == "replication"
        )

    def decision_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "method": BMRB_STUDY_REPLICATION_METHOD,
            "policy_id": self.policy_id,
            "mechanism_id": self.mechanism_id,
            "studies": [
                item.to_mapping() for item in sorted(self.studies, key=lambda item: item.order)
            ],
            "min_successful_replications": self.min_successful_replications,
            "primary_must_pass": True,
            "study_weighting": "one_independent_study_one_vote",
            "participant_weighting_role": "diagnostic_only",
            "scientific_rationale": self.scientific_rationale,
        }

    @property
    def decision_fingerprint(self) -> str:
        return canonical_scientific_fingerprint(
            "quantumbci.bmrb-study-replication-policy.v1", self.decision_payload()
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
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BMRBStudyReplicationPolicy":
        if payload.get("method") not in (None, BMRB_STUDY_REPLICATION_METHOD):
            raise ValueError(f"unexpected study replication method: {payload.get('method')!r}")
        raw_studies = payload.get("studies")
        if not isinstance(raw_studies, Sequence) or isinstance(raw_studies, (str, bytes)):
            raise ValueError("studies must be an array")
        registration = payload.get("preregistration")
        if registration is not None and not isinstance(registration, Mapping):
            raise ValueError("preregistration must be an object or null")
        policy = cls(
            policy_id=_required_text("policy_id", payload.get("policy_id")),
            mechanism_id=_required_text("mechanism_id", payload.get("mechanism_id")),
            studies=tuple(BMRBStudyReplicationSlot.from_mapping(item) for item in raw_studies),
            min_successful_replications=_positive_int(
                "min_successful_replications", payload.get("min_successful_replications")
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
class BMRBStudyEvidence:
    """One bounded study-level evidence object consumed by replication authority."""

    study_id: str
    dataset_id: str
    mechanism_id: str
    participant_count: int
    scientific_criteria_passed: bool
    confirmatory_authority: bool
    promotion_eligible: bool
    reference_effect: float
    reference_ci_lower: float
    reference_ci_upper: float
    source_fingerprint: str

    def __post_init__(self) -> None:
        for name in ("study_id", "dataset_id", "mechanism_id", "source_fingerprint"):
            object.__setattr__(self, name, _required_text(name, getattr(self, name)))
        object.__setattr__(
            self,
            "participant_count",
            _positive_int("participant_count", self.participant_count),
        )
        for name in (
            "scientific_criteria_passed",
            "confirmatory_authority",
            "promotion_eligible",
        ):
            object.__setattr__(self, name, _strict_bool(name, getattr(self, name)))
        for name in ("reference_effect", "reference_ci_lower", "reference_ci_upper"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.reference_ci_lower > self.reference_ci_upper:
            raise ValueError("reference confidence interval lower bound exceeds upper bound")
        expected_promotion = self.scientific_criteria_passed and self.confirmatory_authority
        if self.promotion_eligible != expected_promotion:
            raise ValueError(
                "study promotion_eligible must equal scientific PASS AND confirmatory authority"
            )

    @property
    def evidence_fingerprint(self) -> str:
        return canonical_scientific_fingerprint(
            "quantumbci.bmrb-study-evidence.v1", self.decision_payload()
        )

    def decision_payload(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "dataset_id": self.dataset_id,
            "mechanism_id": self.mechanism_id,
            "participant_count": self.participant_count,
            "scientific_criteria_passed": self.scientific_criteria_passed,
            "confirmatory_authority": self.confirmatory_authority,
            "promotion_eligible": self.promotion_eligible,
            "reference_effect": self.reference_effect,
            "reference_ci_lower": self.reference_ci_lower,
            "reference_ci_upper": self.reference_ci_upper,
            "source_fingerprint": self.source_fingerprint,
        }

    def to_mapping(self) -> dict[str, Any]:
        return {**self.decision_payload(), "evidence_fingerprint": self.evidence_fingerprint}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BMRBStudyEvidence":
        evidence = cls(
            study_id=_required_text("study_id", payload.get("study_id")),
            dataset_id=_required_text("dataset_id", payload.get("dataset_id")),
            mechanism_id=_required_text("mechanism_id", payload.get("mechanism_id")),
            participant_count=_positive_int("participant_count", payload.get("participant_count")),
            scientific_criteria_passed=_strict_bool(
                "scientific_criteria_passed", payload.get("scientific_criteria_passed")
            ),
            confirmatory_authority=_strict_bool(
                "confirmatory_authority", payload.get("confirmatory_authority")
            ),
            promotion_eligible=_strict_bool(
                "promotion_eligible", payload.get("promotion_eligible")
            ),
            reference_effect=_finite("reference_effect", payload.get("reference_effect")),
            reference_ci_lower=_finite(
                "reference_ci_lower", payload.get("reference_ci_lower")
            ),
            reference_ci_upper=_finite(
                "reference_ci_upper", payload.get("reference_ci_upper")
            ),
            source_fingerprint=_required_text(
                "source_fingerprint", payload.get("source_fingerprint")
            ),
        )
        supplied = payload.get("evidence_fingerprint")
        if supplied is not None and str(supplied) != evidence.evidence_fingerprint:
            raise ValueError("evidence_fingerprint does not match reconstructed evidence")
        return evidence

    @classmethod
    def from_confirmatory_result(
        cls,
        result: ConfirmatoryRepresentationResult,
        *,
        dataset_id: str,
    ) -> "BMRBStudyEvidence":
        reference_id = result.policy.reference_representation_id
        matches = [lane for lane in result.lanes if lane.representation_id == reference_id]
        if len(matches) != 1:
            raise ValueError("confirmatory result must contain exactly one reference lane")
        reference = matches[0].candidate
        return cls(
            study_id=result.study_id,
            dataset_id=dataset_id,
            mechanism_id=result.mechanism_id,
            participant_count=result.participant_count,
            scientific_criteria_passed=result.scientific_criteria_passed,
            confirmatory_authority=result.policy.confirmatory_authority,
            promotion_eligible=result.promotion_eligible,
            reference_effect=reference.observed_mean,
            reference_ci_lower=reference.bootstrap_ci_lower,
            reference_ci_upper=reference.bootstrap_ci_upper,
            source_fingerprint=result.source_fingerprint,
        )


@dataclass(frozen=True)
class BMRBStudyReplicationDecision:
    policy: BMRBStudyReplicationPolicy
    evidence: tuple[BMRBStudyEvidence, ...]
    primary_study_passed: bool
    successful_replication_studies: tuple[str, ...]
    failed_replication_studies: tuple[str, ...]
    positive_studies: tuple[str, ...]
    study_positive_fraction: float
    replication_positive_fraction: float
    participant_weighted_positive_fraction: float
    unweighted_study_effect_mean: float
    study_effect_min: float
    study_effect_max: float
    study_effect_range: float
    all_studies_confirmatory_authority: bool

    @property
    def replication_criteria_passed(self) -> bool:
        return bool(
            self.primary_study_passed
            and len(self.successful_replication_studies) >= self.policy.min_successful_replications
        )

    @property
    def context_specific_only(self) -> bool:
        return bool(self.positive_studies and not self.replication_criteria_passed)

    @property
    def broad_claim_authority(self) -> bool:
        return bool(self.policy.confirmatory_authority and self.all_studies_confirmatory_authority)

    @property
    def broad_claim_promotion_eligible(self) -> bool:
        return bool(self.replication_criteria_passed and self.broad_claim_authority)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_role": BMRB_STUDY_REPLICATION_RESULT_ROLE,
            "method": BMRB_STUDY_REPLICATION_METHOD,
            "policy": self.policy.to_mapping(),
            "evidence": [item.to_mapping() for item in self.evidence],
            "primary_study_passed": self.primary_study_passed,
            "successful_replication_studies": list(self.successful_replication_studies),
            "failed_replication_studies": list(self.failed_replication_studies),
            "positive_studies": list(self.positive_studies),
            "study_positive_fraction": self.study_positive_fraction,
            "replication_positive_fraction": self.replication_positive_fraction,
            "participant_weighted_positive_fraction": self.participant_weighted_positive_fraction,
            "participant_weighting_role": "diagnostic_only",
            "unweighted_study_effect_mean": self.unweighted_study_effect_mean,
            "study_effect_min": self.study_effect_min,
            "study_effect_max": self.study_effect_max,
            "study_effect_range": self.study_effect_range,
            "all_studies_confirmatory_authority": self.all_studies_confirmatory_authority,
            "replication_criteria_passed": self.replication_criteria_passed,
            "context_specific_only": self.context_specific_only,
            "broad_claim_authority": self.broad_claim_authority,
            "broad_claim_promotion_eligible": self.broad_claim_promotion_eligible,
            "physical_quantum_promotion_eligible": False,
            "interpretation": (
                "Replication authority counts independent studies, not participant rows. "
                "Participant-weighted summaries are diagnostic only; failed broad replication "
                "does not erase study-specific evidence."
            ),
        }


def evaluate_study_replication(
    policy: BMRBStudyReplicationPolicy,
    evidence: Sequence[BMRBStudyEvidence],
) -> BMRBStudyReplicationDecision:
    """Apply a complete frozen study family to one broad-replication decision."""

    materialized = tuple(evidence)
    by_study: dict[str, BMRBStudyEvidence] = {}
    for item in materialized:
        if item.study_id in by_study:
            raise ValueError(f"duplicate study evidence for {item.study_id!r}")
        by_study[item.study_id] = item
    frozen_ids = {item.study_id for item in policy.studies}
    supplied_ids = set(by_study)
    if supplied_ids != frozen_ids:
        missing = sorted(frozen_ids - supplied_ids)
        extra = sorted(supplied_ids - frozen_ids)
        raise ValueError(
            f"study evidence must match frozen family exactly; missing={missing} extra={extra}"
        )

    ordered: list[BMRBStudyEvidence] = []
    source_fingerprints: set[str] = set()
    for slot in sorted(policy.studies, key=lambda item: item.order):
        item = by_study[slot.study_id]
        if item.dataset_id != slot.dataset_id:
            raise ValueError(
                f"dataset identity mismatch for study {slot.study_id!r}: "
                f"expected={slot.dataset_id!r} observed={item.dataset_id!r}"
            )
        if item.mechanism_id != policy.mechanism_id:
            raise ValueError(
                f"mechanism mismatch for study {slot.study_id!r}: "
                f"expected={policy.mechanism_id!r} observed={item.mechanism_id!r}"
            )
        if item.source_fingerprint in source_fingerprints:
            raise ValueError("independent study slots cannot reuse the same source_fingerprint")
        source_fingerprints.add(item.source_fingerprint)
        ordered.append(item)

    primary = ordered[0]
    replications = ordered[1:]
    successful_replications = tuple(
        item.study_id for item in replications if item.scientific_criteria_passed
    )
    failed_replications = tuple(
        item.study_id for item in replications if not item.scientific_criteria_passed
    )
    positives = tuple(item.study_id for item in ordered if item.scientific_criteria_passed)
    total_participants = sum(item.participant_count for item in ordered)
    positive_participants = sum(
        item.participant_count for item in ordered if item.scientific_criteria_passed
    )
    effects = [item.reference_effect for item in ordered]

    return BMRBStudyReplicationDecision(
        policy=policy,
        evidence=tuple(ordered),
        primary_study_passed=primary.scientific_criteria_passed,
        successful_replication_studies=successful_replications,
        failed_replication_studies=failed_replications,
        positive_studies=positives,
        study_positive_fraction=float(len(positives) / len(ordered)),
        replication_positive_fraction=float(len(successful_replications) / len(replications)),
        participant_weighted_positive_fraction=float(positive_participants / total_participants),
        unweighted_study_effect_mean=float(sum(effects) / len(effects)),
        study_effect_min=float(min(effects)),
        study_effect_max=float(max(effects)),
        study_effect_range=float(max(effects) - min(effects)),
        all_studies_confirmatory_authority=all(
            item.confirmatory_authority for item in ordered
        ),
    )
