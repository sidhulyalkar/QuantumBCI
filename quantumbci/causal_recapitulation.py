"""Causal-necessity evidence for Brain Mechanism Recapitulation Benchmarking.

The causal tier is deliberately stricter than intervention faithfulness inside a
single model. QuantumBCI combines three surfaces:

1. preregistered intervention direction / dose response;
2. held-out mechanism faithfulness and ablation necessity;
3. matched-classical recovery after the candidate mechanism is ablated.

neuros-mechint artifacts remain the authority for their own intervention policies.
QuantumBCI consumes their versioned scientific payloads without requiring PyTorch at
runtime, then adds BMRB-specific participant replication and promotion discipline.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .recapitulation import (
    EvidenceGate,
    EvidenceTier,
    GateStatus,
    MechanismNecessityProfile,
)

NEUROS_MECHINT_DOSE_RESPONSE_SCHEMA = "neuros-mechint.dose-response-study.v1"
NEUROS_MECHINT_EVIDENCE_PACK_SCHEMA = "neuros-mechint.evidence-pack.v1"
CAUSAL_EVIDENCE_METHOD_ID = "participant_balanced_causal_necessity_v1"


def _required_text(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _finite(name: str, value: Any) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _fraction(name: str, value: Any, *, allow_above_one: bool = False) -> float:
    number = _finite(name, value)
    upper_ok = allow_above_one or number <= 1.0
    if number < 0.0 or not upper_ok:
        suffix = "[0, +inf)" if allow_above_one else "[0, 1]"
        raise ValueError(f"{name} must lie in {suffix}")
    return number


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fingerprint(domain: bytes, value: Any) -> str:
    return sha256(domain + _canonical_json(value).encode("utf-8")).hexdigest()


def _unwrap_scientific_result(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept a native scientific result or a neuros-mechint artifact envelope."""

    result = payload.get("result")
    if isinstance(result, Mapping):
        return result
    return payload


@dataclass(frozen=True)
class MatchedClassicalRecovery:
    """Recovery attempt after candidate-mechanism ablation.

    ``classical_recovery_fraction`` is the fraction of candidate ablation loss restored
    by the strongest matched classical alternative under the same evidence/information
    budget. Values above 1 are allowed and mean the control recovered more than the
    candidate's measured loss. Negative recovery is prohibited; worsening controls
    should be recorded as zero recovery plus metadata in the producing study.
    """

    classical_model_id: str
    classical_recovery_fraction: float
    information_set_id: str
    source_fingerprint: str

    def __post_init__(self) -> None:
        _required_text("classical_model_id", self.classical_model_id)
        _required_text("information_set_id", self.information_set_id)
        _required_text("source_fingerprint", self.source_fingerprint)
        object.__setattr__(
            self,
            "classical_recovery_fraction",
            _fraction(
                "classical_recovery_fraction",
                self.classical_recovery_fraction,
                allow_above_one=True,
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "classical_model_id": self.classical_model_id,
            "classical_recovery_fraction": float(self.classical_recovery_fraction),
            "information_set_id": self.information_set_id,
            "source_fingerprint": self.source_fingerprint,
        }


@dataclass(frozen=True)
class CausalCaseEvidence:
    participant_id: str
    occasion_id: str
    case_id: str
    mechanism_id: str
    intervention_id: str
    dose_response_passed: bool
    oriented_endpoint_effect: float
    mean_monotonic_fraction: float
    faithfulness_passed: bool
    sufficiency_fraction: float
    necessity_fraction: float
    joint_random_percentile: float
    matched_recovery: MatchedClassicalRecovery
    dose_source_fingerprint: str
    faithfulness_source_fingerprint: str
    source_schemas: tuple[str, str]

    def __post_init__(self) -> None:
        for name in (
            "participant_id",
            "occasion_id",
            "case_id",
            "mechanism_id",
            "intervention_id",
            "dose_source_fingerprint",
            "faithfulness_source_fingerprint",
        ):
            _required_text(name, getattr(self, name))
        object.__setattr__(
            self, "oriented_endpoint_effect", _finite("oriented_endpoint_effect", self.oriented_endpoint_effect)
        )
        object.__setattr__(
            self,
            "mean_monotonic_fraction",
            _fraction("mean_monotonic_fraction", self.mean_monotonic_fraction),
        )
        object.__setattr__(
            self, "sufficiency_fraction", _finite("sufficiency_fraction", self.sufficiency_fraction)
        )
        object.__setattr__(
            self, "necessity_fraction", _finite("necessity_fraction", self.necessity_fraction)
        )
        object.__setattr__(
            self,
            "joint_random_percentile",
            _fraction("joint_random_percentile", self.joint_random_percentile),
        )
        if len(self.source_schemas) != 2 or not all(str(item).strip() for item in self.source_schemas):
            raise ValueError("source_schemas must contain dose-response and faithfulness schemas")

    @property
    def direction_matched(self) -> bool:
        return self.oriented_endpoint_effect > 0.0

    def to_mapping(self) -> dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "occasion_id": self.occasion_id,
            "case_id": self.case_id,
            "mechanism_id": self.mechanism_id,
            "intervention_id": self.intervention_id,
            "dose_response_passed": bool(self.dose_response_passed),
            "direction_matched": bool(self.direction_matched),
            "oriented_endpoint_effect": float(self.oriented_endpoint_effect),
            "mean_monotonic_fraction": float(self.mean_monotonic_fraction),
            "faithfulness_passed": bool(self.faithfulness_passed),
            "sufficiency_fraction": float(self.sufficiency_fraction),
            "necessity_fraction": float(self.necessity_fraction),
            "joint_random_percentile": float(self.joint_random_percentile),
            "matched_recovery": self.matched_recovery.to_mapping(),
            "dose_source_fingerprint": self.dose_source_fingerprint,
            "faithfulness_source_fingerprint": self.faithfulness_source_fingerprint,
            "source_schemas": list(self.source_schemas),
        }


@dataclass(frozen=True)
class CausalNecessityPolicy:
    """Explicit BMRB causal promotion policy.

    ``preregistered`` is intentionally part of the scientific contract. A policy can
    evaluate evidence when false, but it cannot authorize causal promotion.
    """

    policy_id: str
    preregistered: bool
    min_participants: int = 3
    min_direction_match_fraction: float = 0.80
    min_dose_response_pass_fraction: float = 0.80
    min_faithfulness_pass_fraction: float = 0.80
    min_mean_necessity_fraction: float = 0.50
    min_mean_joint_random_percentile: float = 0.95
    max_mean_classical_recovery_fraction: float = 0.25

    def __post_init__(self) -> None:
        _required_text("policy_id", self.policy_id)
        if self.min_participants < 2:
            raise ValueError("min_participants must be at least 2")
        for name in (
            "min_direction_match_fraction",
            "min_dose_response_pass_fraction",
            "min_faithfulness_pass_fraction",
            "min_mean_joint_random_percentile",
        ):
            object.__setattr__(self, name, _fraction(name, getattr(self, name)))
        object.__setattr__(
            self,
            "min_mean_necessity_fraction",
            _finite("min_mean_necessity_fraction", self.min_mean_necessity_fraction),
        )
        object.__setattr__(
            self,
            "max_mean_classical_recovery_fraction",
            _fraction(
                "max_mean_classical_recovery_fraction",
                self.max_mean_classical_recovery_fraction,
                allow_above_one=True,
            ),
        )

    @property
    def decision_rule(self) -> str:
        return (
            f"participants>={self.min_participants}; "
            f"direction_match>={self.min_direction_match_fraction:.3f}; "
            f"dose_pass>={self.min_dose_response_pass_fraction:.3f}; "
            f"faithfulness_pass>={self.min_faithfulness_pass_fraction:.3f}; "
            f"mean_necessity>={self.min_mean_necessity_fraction:.3f}; "
            f"mean_joint_random_percentile>={self.min_mean_joint_random_percentile:.3f}; "
            f"mean_classical_recovery<={self.max_mean_classical_recovery_fraction:.3f}"
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "preregistered": bool(self.preregistered),
            "min_participants": int(self.min_participants),
            "min_direction_match_fraction": float(self.min_direction_match_fraction),
            "min_dose_response_pass_fraction": float(self.min_dose_response_pass_fraction),
            "min_faithfulness_pass_fraction": float(self.min_faithfulness_pass_fraction),
            "min_mean_necessity_fraction": float(self.min_mean_necessity_fraction),
            "min_mean_joint_random_percentile": float(self.min_mean_joint_random_percentile),
            "max_mean_classical_recovery_fraction": float(
                self.max_mean_classical_recovery_fraction
            ),
            "decision_rule": self.decision_rule,
        }


@dataclass(frozen=True)
class ParticipantCausalSummary:
    participant_id: str
    case_count: int
    direction_match_fraction: float
    dose_response_pass_fraction: float
    faithfulness_pass_fraction: float
    mean_oriented_endpoint_effect: float
    mean_monotonic_fraction: float
    mean_sufficiency_fraction: float
    mean_necessity_fraction: float
    mean_joint_random_percentile: float
    mean_classical_recovery_fraction: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "case_count": int(self.case_count),
            "direction_match_fraction": float(self.direction_match_fraction),
            "dose_response_pass_fraction": float(self.dose_response_pass_fraction),
            "faithfulness_pass_fraction": float(self.faithfulness_pass_fraction),
            "mean_oriented_endpoint_effect": float(self.mean_oriented_endpoint_effect),
            "mean_monotonic_fraction": float(self.mean_monotonic_fraction),
            "mean_sufficiency_fraction": float(self.mean_sufficiency_fraction),
            "mean_necessity_fraction": float(self.mean_necessity_fraction),
            "mean_joint_random_percentile": float(self.mean_joint_random_percentile),
            "mean_classical_recovery_fraction": float(self.mean_classical_recovery_fraction),
        }


@dataclass(frozen=True)
class CausalNecessityResult:
    mechanism_id: str
    intervention_id: str
    policy: CausalNecessityPolicy
    participants: tuple[ParticipantCausalSummary, ...]
    cases: tuple[CausalCaseEvidence, ...]
    direction_match_fraction: float
    dose_response_pass_fraction: float
    faithfulness_pass_fraction: float
    mean_oriented_endpoint_effect: float
    mean_necessity_fraction: float
    mean_joint_random_percentile: float
    mean_classical_recovery_fraction: float
    scientific_criteria_passed: bool
    reasons: tuple[str, ...]
    source_fingerprint: str

    @property
    def promotion_eligible(self) -> bool:
        return bool(self.policy.preregistered and self.scientific_criteria_passed)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_role": "causal_necessity_evidence",
            "method_id": CAUSAL_EVIDENCE_METHOD_ID,
            "mechanism_id": self.mechanism_id,
            "intervention_id": self.intervention_id,
            "policy": self.policy.to_mapping(),
            "participant_count": len(self.participants),
            "case_count": len(self.cases),
            "direction_match_fraction": float(self.direction_match_fraction),
            "dose_response_pass_fraction": float(self.dose_response_pass_fraction),
            "faithfulness_pass_fraction": float(self.faithfulness_pass_fraction),
            "mean_oriented_endpoint_effect": float(self.mean_oriented_endpoint_effect),
            "mean_necessity_fraction": float(self.mean_necessity_fraction),
            "mean_joint_random_percentile": float(self.mean_joint_random_percentile),
            "mean_classical_recovery_fraction": float(self.mean_classical_recovery_fraction),
            "scientific_criteria_passed": bool(self.scientific_criteria_passed),
            "promotion_eligible": bool(self.promotion_eligible),
            "reasons": list(self.reasons),
            "source_fingerprint": self.source_fingerprint,
            "participants": [item.to_mapping() for item in self.participants],
            "cases": [item.to_mapping() for item in self.cases],
            "physical_quantum_promotion_eligible": False,
            "interpretation": (
                "Causal necessity requires intervention direction/dose response, held-out "
                "faithfulness/ablation evidence, and failure of the strongest matched classical "
                "recovery control. Passing scientific criteria is distinct from promotion "
                "eligibility: the policy must also have been preregistered."
            ),
        }


def causal_case_from_neuros_mechint(
    *,
    participant_id: str,
    occasion_id: str,
    case_id: str,
    mechanism_id: str,
    dose_response: Mapping[str, Any],
    faithfulness: Mapping[str, Any],
    matched_recovery: MatchedClassicalRecovery,
) -> CausalCaseEvidence:
    """Adapt versioned neuros-mechint scientific results into one BMRB case."""

    dose = _unwrap_scientific_result(dose_response)
    dose_schema = str(dose.get("schema_version", ""))
    if dose_schema != NEUROS_MECHINT_DOSE_RESPONSE_SCHEMA:
        raise ValueError(
            "unsupported neuros-mechint dose-response schema: "
            f"{dose_schema!r}"
        )
    dose_spec = dose.get("spec")
    if not isinstance(dose_spec, Mapping):
        raise ValueError("dose-response result is missing spec")
    intervention_id = _required_text("intervention_id", dose_spec.get("intervention_id"))
    expected_direction = int(dose_spec.get("expected_direction", 0))
    if expected_direction not in {-1, 1}:
        raise ValueError("dose-response expected_direction must be -1 or 1")
    endpoint_effect = _finite("endpoint_effect", dose.get("endpoint_effect"))
    # neuros-mechint serializes endpoint_effect already oriented by expected_direction.
    monotonic = _fraction("mean_monotonic_fraction", dose.get("mean_monotonic_fraction"))
    dose_fp = _required_text(
        "dose study_fingerprint", dose.get("study_fingerprint", _fingerprint(b"qbc.dose\0", dose))
    )

    pack = _unwrap_scientific_result(faithfulness)
    pack_schema = str(pack.get("schema_version", ""))
    if pack_schema != NEUROS_MECHINT_EVIDENCE_PACK_SCHEMA:
        raise ValueError(
            "unsupported neuros-mechint evidence-pack schema: "
            f"{pack_schema!r}"
        )
    validation = pack.get("validation_aggregate")
    promotion = pack.get("promotion")
    if not isinstance(validation, Mapping) or not isinstance(promotion, Mapping):
        raise ValueError("evidence pack is missing validation_aggregate or promotion")
    joint_random = validation.get("mean_joint_random_percentile")
    if joint_random is None:
        raise ValueError("evidence pack validation joint random percentile is unavailable")
    pack_fp = _required_text(
        "faithfulness study_fingerprint",
        pack.get("study_fingerprint", _fingerprint(b"qbc.faithfulness\0", pack)),
    )

    return CausalCaseEvidence(
        participant_id=_required_text("participant_id", participant_id),
        occasion_id=_required_text("occasion_id", occasion_id),
        case_id=_required_text("case_id", case_id),
        mechanism_id=_required_text("mechanism_id", mechanism_id),
        intervention_id=intervention_id,
        dose_response_passed=bool(dose.get("passed", False)),
        oriented_endpoint_effect=endpoint_effect,
        mean_monotonic_fraction=monotonic,
        faithfulness_passed=bool(promotion.get("passed", False)),
        sufficiency_fraction=_finite(
            "mean_sufficiency", validation.get("mean_sufficiency")
        ),
        necessity_fraction=_finite(
            "mean_necessity", validation.get("mean_necessity")
        ),
        joint_random_percentile=_fraction(
            "mean_joint_random_percentile", joint_random
        ),
        matched_recovery=matched_recovery,
        dose_source_fingerprint=dose_fp,
        faithfulness_source_fingerprint=pack_fp,
        source_schemas=(dose_schema, pack_schema),
    )


def _participant_summary(
    participant_id: str,
    cases: Sequence[CausalCaseEvidence],
) -> ParticipantCausalSummary:
    if not cases:
        raise ValueError("participant causal summary requires cases")
    values = lambda fn: np.asarray([fn(case) for case in cases], dtype=float)
    return ParticipantCausalSummary(
        participant_id=participant_id,
        case_count=len(cases),
        direction_match_fraction=float(np.mean(values(lambda case: case.direction_matched))),
        dose_response_pass_fraction=float(np.mean(values(lambda case: case.dose_response_passed))),
        faithfulness_pass_fraction=float(np.mean(values(lambda case: case.faithfulness_passed))),
        mean_oriented_endpoint_effect=float(np.mean(values(lambda case: case.oriented_endpoint_effect))),
        mean_monotonic_fraction=float(np.mean(values(lambda case: case.mean_monotonic_fraction))),
        mean_sufficiency_fraction=float(np.mean(values(lambda case: case.sufficiency_fraction))),
        mean_necessity_fraction=float(np.mean(values(lambda case: case.necessity_fraction))),
        mean_joint_random_percentile=float(np.mean(values(lambda case: case.joint_random_percentile))),
        mean_classical_recovery_fraction=float(
            np.mean(values(lambda case: case.matched_recovery.classical_recovery_fraction))
        ),
    )


def evaluate_causal_necessity(
    cases: Iterable[CausalCaseEvidence],
    *,
    policy: CausalNecessityPolicy,
) -> CausalNecessityResult:
    materialized = tuple(cases)
    if not materialized:
        raise ValueError("causal necessity requires case evidence")
    case_keys: set[tuple[str, str]] = set()
    for case in materialized:
        key = (case.participant_id, case.occasion_id)
        if key in case_keys:
            raise ValueError(f"duplicate participant/occasion causal case: {key}")
        case_keys.add(key)
    mechanism_ids = {case.mechanism_id for case in materialized}
    intervention_ids = {case.intervention_id for case in materialized}
    if len(mechanism_ids) != 1:
        raise ValueError("causal necessity cases must share one mechanism_id")
    if len(intervention_ids) != 1:
        raise ValueError("causal necessity cases must share one intervention_id")

    participant_ids = tuple(sorted({case.participant_id for case in materialized}))
    participants = tuple(
        _participant_summary(
            participant,
            [case for case in materialized if case.participant_id == participant],
        )
        for participant in participant_ids
    )

    def mean(field: str) -> float:
        return float(np.mean([float(getattr(item, field)) for item in participants]))

    direction = mean("direction_match_fraction")
    dose_pass = mean("dose_response_pass_fraction")
    faith_pass = mean("faithfulness_pass_fraction")
    endpoint = mean("mean_oriented_endpoint_effect")
    necessity = mean("mean_necessity_fraction")
    random_percentile = mean("mean_joint_random_percentile")
    recovery = mean("mean_classical_recovery_fraction")

    reasons: list[str] = []
    if len(participants) < policy.min_participants:
        reasons.append(
            f"independent participants {len(participants)} < {policy.min_participants}"
        )
    if direction < policy.min_direction_match_fraction:
        reasons.append(
            f"direction match {direction:.3f} < {policy.min_direction_match_fraction:.3f}"
        )
    if dose_pass < policy.min_dose_response_pass_fraction:
        reasons.append(
            f"dose-response pass fraction {dose_pass:.3f} < "
            f"{policy.min_dose_response_pass_fraction:.3f}"
        )
    if faith_pass < policy.min_faithfulness_pass_fraction:
        reasons.append(
            f"faithfulness pass fraction {faith_pass:.3f} < "
            f"{policy.min_faithfulness_pass_fraction:.3f}"
        )
    if necessity < policy.min_mean_necessity_fraction:
        reasons.append(
            f"mean necessity {necessity:.3f} < {policy.min_mean_necessity_fraction:.3f}"
        )
    if random_percentile < policy.min_mean_joint_random_percentile:
        reasons.append(
            f"mean joint random percentile {random_percentile:.3f} < "
            f"{policy.min_mean_joint_random_percentile:.3f}"
        )
    if recovery > policy.max_mean_classical_recovery_fraction:
        reasons.append(
            f"mean matched-classical recovery {recovery:.3f} > "
            f"{policy.max_mean_classical_recovery_fraction:.3f}"
        )

    source_payload = {
        "policy": policy.to_mapping(),
        "cases": [
            case.to_mapping()
            for case in sorted(
                materialized,
                key=lambda item: (item.participant_id, item.occasion_id, item.case_id),
            )
        ],
    }
    return CausalNecessityResult(
        mechanism_id=next(iter(mechanism_ids)),
        intervention_id=next(iter(intervention_ids)),
        policy=policy,
        participants=participants,
        cases=materialized,
        direction_match_fraction=direction,
        dose_response_pass_fraction=dose_pass,
        faithfulness_pass_fraction=faith_pass,
        mean_oriented_endpoint_effect=endpoint,
        mean_necessity_fraction=necessity,
        mean_joint_random_percentile=random_percentile,
        mean_classical_recovery_fraction=recovery,
        scientific_criteria_passed=not reasons,
        reasons=tuple(reasons),
        source_fingerprint=_fingerprint(b"quantumbci.causal-necessity.v1\0", source_payload),
    )


def attach_causal_evidence(
    profile: MechanismNecessityProfile,
    result: CausalNecessityResult,
) -> MechanismNecessityProfile:
    """Return a new BMRB profile with a causally evaluated tier.

    Causal evidence can FAIL even when earlier tiers are unresolved because it is an
    independent falsifier. It can PASS only when the policy was preregistered and the
    upstream profile already has a contiguous PASS ceiling through repeated-case
    evidence. Otherwise successful causal evidence remains CHARACTERIZED.
    """

    if profile.mechanism_id != result.mechanism_id:
        raise ValueError("causal result mechanism_id does not match BMRB profile")
    upstream_ceiling = profile.promotion_ceiling
    upstream_ready = (
        upstream_ceiling is not None and upstream_ceiling >= EvidenceTier.REPEATED_CASE
    )
    if not result.scientific_criteria_passed:
        status = GateStatus.FAIL
        summary = (
            "Causal-necessity evidence failed one or more preregistered/evaluated criteria: "
            + "; ".join(result.reasons)
        )
        threshold = None
    elif result.promotion_eligible and upstream_ready:
        status = GateStatus.PASS
        summary = (
            "Preregistered intervention, faithfulness and matched-classical recovery criteria "
            "passed across independent participants after all upstream BMRB tiers passed."
        )
        threshold = result.policy.decision_rule
    else:
        status = GateStatus.CHARACTERIZED
        blockers = []
        if not result.policy.preregistered:
            blockers.append("causal policy was not preregistered")
        if not upstream_ready:
            blockers.append("upstream BMRB promotion has not reached repeated_case")
        summary = (
            "Causal criteria were measured and satisfied, but promotion is blocked because "
            + " and ".join(blockers)
            + "."
        )
        threshold = None

    causal_gate = EvidenceGate(
        id="causal_intervention_and_ablation",
        tier=EvidenceTier.CAUSAL_MECHANISTIC,
        status=status,
        summary=summary,
        evidence_ref=result.source_fingerprint,
        metric="participant_balanced_causal_necessity",
        value=float(result.mean_necessity_fraction),
        threshold=threshold,
    )
    gates = []
    replaced = False
    for gate in profile.gates:
        if gate.tier == EvidenceTier.CAUSAL_MECHANISTIC:
            gates.append(causal_gate)
            replaced = True
        else:
            gates.append(gate)
    if not replaced:
        gates.append(causal_gate)
    return MechanismNecessityProfile(
        mechanism_id=profile.mechanism_id,
        claim_class=profile.claim_class,
        signature=profile.signature,
        gates=tuple(gates),
        metadata={
            **dict(profile.metadata or {}),
            "causal_source_fingerprint": result.source_fingerprint,
            "causal_policy_id": result.policy.policy_id,
        },
    )
