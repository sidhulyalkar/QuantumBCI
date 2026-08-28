"""Brain Mechanism Recapitulation Benchmark evidence contracts.

BMRB treats a neural signature, rather than "the brain", as the unit of scientific
claim. Evidence coverage and scientific promotion are intentionally separate:
collecting a stability or reliability artifact does not imply that a preregistered
gate was passed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Iterable, Mapping

from .claims import ClaimClass


class EvidenceTier(IntEnum):
    DESCRIPTIVE = 0
    PREDICTIVE = 1
    ADVERSARY_SURVIVING = 2
    SOURCE_STABILITY = 3
    REPEATED_CASE = 4
    CAUSAL_MECHANISTIC = 5
    PHYSICAL_QUANTUM = 6


class GateStatus(str, Enum):
    NOT_RUN = "not_run"
    CHARACTERIZED = "characterized"
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


def _required_text(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


@dataclass(frozen=True)
class RecapitulationSignature:
    """A preregistrable neural computation/signature to be recapitulated."""

    id: str
    title: str
    domain: str
    target: str
    inference_unit: str
    primary_metric: str
    favorable_direction: str
    required_controls: tuple[str, ...]
    description: str

    def __post_init__(self) -> None:
        for name in (
            "id",
            "title",
            "domain",
            "target",
            "inference_unit",
            "primary_metric",
            "favorable_direction",
            "description",
        ):
            _required_text(f"RecapitulationSignature.{name}", getattr(self, name))
        if not self.required_controls:
            raise ValueError("RecapitulationSignature.required_controls must not be empty")
        if any(not str(item).strip() for item in self.required_controls):
            raise ValueError("RecapitulationSignature.required_controls contains an empty control")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "domain": self.domain,
            "target": self.target,
            "inference_unit": self.inference_unit,
            "primary_metric": self.primary_metric,
            "favorable_direction": self.favorable_direction,
            "required_controls": list(self.required_controls),
            "description": self.description,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RecapitulationSignature":
        controls = payload.get("required_controls")
        if not isinstance(controls, (list, tuple)):
            raise ValueError("signature.required_controls must be a list")
        return cls(
            id=_required_text("signature.id", payload.get("id")),
            title=_required_text("signature.title", payload.get("title")),
            domain=_required_text("signature.domain", payload.get("domain")),
            target=_required_text("signature.target", payload.get("target")),
            inference_unit=_required_text(
                "signature.inference_unit", payload.get("inference_unit")
            ),
            primary_metric=_required_text(
                "signature.primary_metric", payload.get("primary_metric")
            ),
            favorable_direction=_required_text(
                "signature.favorable_direction", payload.get("favorable_direction")
            ),
            required_controls=tuple(str(item) for item in controls),
            description=_required_text("signature.description", payload.get("description")),
        )


@dataclass(frozen=True)
class EvidenceGate:
    """One stage of the mechanism-necessity evidence ladder."""

    id: str
    tier: EvidenceTier
    status: GateStatus
    summary: str
    evidence_ref: str | None = None
    metric: str | None = None
    value: float | None = None
    threshold: str | None = None

    def __post_init__(self) -> None:
        _required_text("EvidenceGate.id", self.id)
        _required_text("EvidenceGate.summary", self.summary)
        if self.value is not None and self.status == GateStatus.NOT_RUN:
            raise ValueError("NOT_RUN evidence gates cannot carry a numeric value")
        if self.status == GateStatus.PASS and self.threshold is None:
            raise ValueError(
                "PASS requires an explicit preregistered threshold/decision rule; "
                "use CHARACTERIZED when evidence exists without a gate"
            )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tier": self.tier.name.lower(),
            "tier_index": int(self.tier),
            "status": self.status.value,
            "summary": self.summary,
            "evidence_ref": self.evidence_ref,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EvidenceGate":
        tier_value = payload.get("tier_index")
        if tier_value is not None:
            tier = EvidenceTier(int(tier_value))
        else:
            raw_tier = _required_text("gate.tier", payload.get("tier")).upper()
            try:
                tier = EvidenceTier[raw_tier]
            except KeyError as exc:
                raise ValueError(f"unknown BMRB evidence tier: {raw_tier!r}") from exc
        try:
            status = GateStatus(_required_text("gate.status", payload.get("status")))
        except ValueError as exc:
            raise ValueError(f"unknown BMRB gate status: {payload.get('status')!r}") from exc
        value = payload.get("value")
        return cls(
            id=_required_text("gate.id", payload.get("id")),
            tier=tier,
            status=status,
            summary=_required_text("gate.summary", payload.get("summary")),
            evidence_ref=(
                None if payload.get("evidence_ref") is None else str(payload.get("evidence_ref"))
            ),
            metric=None if payload.get("metric") is None else str(payload.get("metric")),
            value=None if value is None else float(value),
            threshold=(
                None if payload.get("threshold") is None else str(payload.get("threshold"))
            ),
        )


def validate_monotonic_promotion(gates: Iterable[EvidenceGate]) -> None:
    """Raise when a later tier passes despite an earlier non-pass gate."""

    ordered = sorted(tuple(gates), key=lambda gate: int(gate.tier))
    blocked = False
    for gate in ordered:
        if blocked and gate.status == GateStatus.PASS:
            raise ValueError(
                f"gate {gate.id!r} passes after an earlier tier blocked promotion"
            )
        if gate.status != GateStatus.PASS:
            blocked = True


@dataclass(frozen=True)
class MechanismNecessityProfile:
    """Mechanism x neural-signature evidence profile."""

    mechanism_id: str
    claim_class: ClaimClass
    signature: RecapitulationSignature
    gates: tuple[EvidenceGate, ...]
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _required_text("mechanism_id", self.mechanism_id)
        if not self.gates:
            raise ValueError("MechanismNecessityProfile.gates must not be empty")
        ids = [gate.id for gate in self.gates]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence gate ids must be unique")
        tiers = [gate.tier for gate in self.gates]
        if len(tiers) != len(set(tiers)):
            raise ValueError("BMRB v1 allows exactly one evidence gate per tier")
        if EvidenceTier.DESCRIPTIVE not in tiers:
            raise ValueError("BMRB profiles must include a descriptive evidence gate")
        validate_monotonic_promotion(self.gates)
        physical = next(
            (gate for gate in self.gates if gate.tier == EvidenceTier.PHYSICAL_QUANTUM),
            None,
        )
        if (
            physical is not None
            and self.claim_class != ClaimClass.PHYSICAL_QUANTUM
            and physical.status not in {GateStatus.NOT_APPLICABLE, GateStatus.NOT_RUN}
        ):
            raise ValueError(
                "non-physical claim classes cannot pass or characterize physical-quantum evidence"
            )

    @property
    def ordered_gates(self) -> tuple[EvidenceGate, ...]:
        return tuple(sorted(self.gates, key=lambda gate: int(gate.tier)))

    @property
    def evidence_coverage_tier(self) -> EvidenceTier:
        available = [
            gate.tier
            for gate in self.gates
            if gate.status in {GateStatus.CHARACTERIZED, GateStatus.PASS, GateStatus.FAIL}
        ]
        if not available:
            return EvidenceTier.DESCRIPTIVE
        return max(available)

    @property
    def promotion_ceiling(self) -> EvidenceTier | None:
        by_tier = {gate.tier: gate for gate in self.gates}
        ceiling: EvidenceTier | None = None
        for tier in EvidenceTier:
            gate = by_tier.get(tier)
            if gate is None or gate.status != GateStatus.PASS:
                break
            ceiling = tier
        return ceiling

    @property
    def first_failing_gate(self) -> str | None:
        for gate in self.ordered_gates:
            if gate.status == GateStatus.FAIL:
                return gate.id
        return None

    @property
    def unresolved_gate(self) -> str | None:
        for gate in self.ordered_gates:
            if gate.status in {GateStatus.NOT_RUN, GateStatus.CHARACTERIZED}:
                return gate.id
        return None

    def to_mapping(self) -> dict[str, Any]:
        ceiling = self.promotion_ceiling
        return {
            "schema_version": 1,
            "artifact_role": "mechanism_necessity_profile",
            "mechanism_id": self.mechanism_id,
            "claim_class": self.claim_class.value,
            "signature": self.signature.to_mapping(),
            "gates": [gate.to_mapping() for gate in self.ordered_gates],
            "evidence_coverage_tier": self.evidence_coverage_tier.name.lower(),
            "promotion_ceiling": None if ceiling is None else ceiling.name.lower(),
            "first_failing_gate": self.first_failing_gate,
            "unresolved_gate": self.unresolved_gate,
            "metadata": dict(self.metadata or {}),
            "necessity_claim_permitted": bool(
                ceiling is not None and ceiling >= EvidenceTier.CAUSAL_MECHANISTIC
            ),
            "interpretation": (
                "Evidence coverage records how far the study has measured. Promotion ceiling "
                "records only contiguous preregistered PASS decisions. A characterized stability "
                "or reliability surface is therefore visible without being silently promoted."
            ),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MechanismNecessityProfile":
        role = payload.get("artifact_role")
        if role is not None and role != "mechanism_necessity_profile":
            raise ValueError(f"unexpected profile artifact_role: {role!r}")
        signature = payload.get("signature")
        gates = payload.get("gates")
        metadata = payload.get("metadata", {})
        if not isinstance(signature, Mapping):
            raise ValueError("mechanism profile is missing signature mapping")
        if not isinstance(gates, list) or not gates:
            raise ValueError("mechanism profile is missing gates list")
        if not isinstance(metadata, Mapping):
            raise ValueError("mechanism profile metadata must be an object")
        try:
            claim_class = ClaimClass(_required_text("claim_class", payload.get("claim_class")))
        except ValueError as exc:
            raise ValueError(f"unknown QuantumBCI claim_class: {payload.get('claim_class')!r}") from exc
        profile = cls(
            mechanism_id=_required_text("mechanism_id", payload.get("mechanism_id")),
            claim_class=claim_class,
            signature=RecapitulationSignature.from_mapping(signature),
            gates=tuple(
                EvidenceGate.from_mapping(gate)
                for gate in gates
                if isinstance(gate, Mapping)
            ),
            metadata=dict(metadata),
        )
        if len(profile.gates) != len(gates):
            raise ValueError("every mechanism profile gate must be an object")
        # Derived fields are not authority, but when present they must agree with the
        # reconstructed contract so stale or hand-edited reports fail closed.
        expected = profile.to_mapping()
        for key in (
            "evidence_coverage_tier",
            "promotion_ceiling",
            "first_failing_gate",
            "unresolved_gate",
            "necessity_claim_permitted",
        ):
            if key in payload and payload.get(key) != expected.get(key):
                raise ValueError(f"mechanism profile derived field mismatch: {key}")
        return profile


def bmrb_dynamics_signature() -> RecapitulationSignature:
    return RecapitulationSignature(
        id="BMRB_DYNAMICS_V1",
        title="Held-out neural latent dynamics recapitulation",
        domain="temporal_dynamics",
        target=(
            "one-step and autonomous-rollout structure of frozen neural latent trajectories, "
            "including uncertainty and regime-sensitive alternatives"
        ),
        inference_unit="participant",
        primary_metric="held_out_dynamics_error_and_predictive_density",
        favorable_direction="preregistered_per_comparison",
        required_controls=(
            "unconstrained_affine_generator",
            "observed_state_var",
            "probabilistic_state_space",
            "switching_state",
            "flexible_nonlinear",
        ),
        description=(
            "Tests whether a candidate mechanism is required to reproduce declared held-out "
            "neural dynamics after matched classical predictive adversaries, source stability, "
            "repeated-case reliability, and eventually causal intervention/ablation evidence."
        ),
    )


def gate_map(profile: MechanismNecessityProfile) -> dict[str, EvidenceGate]:
    return {gate.id: gate for gate in profile.gates}


def mechanism_profile_from_mapping(payload: Mapping[str, Any]) -> MechanismNecessityProfile:
    """Strict convenience parser for serialized BMRB mechanism profiles."""

    return MechanismNecessityProfile.from_mapping(payload)
