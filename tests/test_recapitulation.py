from __future__ import annotations

import pytest

from quantumbci.claims import ClaimClass
from quantumbci.recapitulation import (
    EvidenceGate,
    EvidenceTier,
    GateStatus,
    MechanismNecessityProfile,
    bmrb_dynamics_signature,
    validate_monotonic_promotion,
)


def test_characterized_evidence_does_not_silently_promote() -> None:
    profile = MechanismNecessityProfile(
        mechanism_id="lindblad_latent_dynamics",
        claim_class=ClaimClass.QUANTUM_INSPIRED,
        signature=bmrb_dynamics_signature(),
        gates=(
            EvidenceGate(
                id="descriptive",
                tier=EvidenceTier.DESCRIPTIVE,
                status=GateStatus.PASS,
                summary="qualified evidence exists",
                threshold="all declared case artifacts pass execution",
            ),
            EvidenceGate(
                id="predictive",
                tier=EvidenceTier.PREDICTIVE,
                status=GateStatus.CHARACTERIZED,
                summary="predictive evidence measured without a preregistered pooled threshold",
            ),
            EvidenceGate(
                id="stability",
                tier=EvidenceTier.SOURCE_STABILITY,
                status=GateStatus.CHARACTERIZED,
                summary="source stability measured without a universal gate",
            ),
            EvidenceGate(
                id="physical",
                tier=EvidenceTier.PHYSICAL_QUANTUM,
                status=GateStatus.NOT_APPLICABLE,
                summary="not a physical-quantum claim",
            ),
        ),
    )

    assert profile.evidence_coverage_tier == EvidenceTier.SOURCE_STABILITY
    assert profile.promotion_ceiling == EvidenceTier.DESCRIPTIVE
    assert profile.unresolved_gate == "predictive"
    assert profile.to_mapping()["necessity_claim_permitted"] is False


def test_pass_requires_explicit_decision_rule() -> None:
    with pytest.raises(ValueError, match="PASS requires"):
        EvidenceGate(
            id="predictive",
            tier=EvidenceTier.PREDICTIVE,
            status=GateStatus.PASS,
            summary="looks good",
        )


def _nonmonotonic_gates() -> tuple[EvidenceGate, ...]:
    return (
        EvidenceGate(
            id="descriptive",
            tier=EvidenceTier.DESCRIPTIVE,
            status=GateStatus.PASS,
            summary="qualified evidence exists",
            threshold="qualified artifact present",
        ),
        EvidenceGate(
            id="predictive",
            tier=EvidenceTier.PREDICTIVE,
            status=GateStatus.CHARACTERIZED,
            summary="measured but no gate",
        ),
        EvidenceGate(
            id="adversary",
            tier=EvidenceTier.ADVERSARY_SURVIVING,
            status=GateStatus.PASS,
            summary="would otherwise pass",
            threshold="predeclared comparison margin",
        ),
    )


def test_later_pass_cannot_jump_over_unresolved_tier() -> None:
    with pytest.raises(ValueError, match="passes after an earlier tier blocked"):
        validate_monotonic_promotion(_nonmonotonic_gates())


def test_profile_constructor_enforces_monotonic_promotion() -> None:
    with pytest.raises(ValueError, match="passes after an earlier tier blocked"):
        MechanismNecessityProfile(
            mechanism_id="lindblad_latent_dynamics",
            claim_class=ClaimClass.QUANTUM_INSPIRED,
            signature=bmrb_dynamics_signature(),
            gates=_nonmonotonic_gates(),
        )


def test_quantum_inspired_profile_cannot_claim_physical_quantum_evidence() -> None:
    with pytest.raises(ValueError, match="non-physical claim classes"):
        MechanismNecessityProfile(
            mechanism_id="lindblad_latent_dynamics",
            claim_class=ClaimClass.QUANTUM_INSPIRED,
            signature=bmrb_dynamics_signature(),
            gates=(
                EvidenceGate(
                    id="descriptive",
                    tier=EvidenceTier.DESCRIPTIVE,
                    status=GateStatus.PASS,
                    summary="qualified evidence exists",
                    threshold="qualified artifact present",
                ),
                EvidenceGate(
                    id="physical",
                    tier=EvidenceTier.PHYSICAL_QUANTUM,
                    status=GateStatus.CHARACTERIZED,
                    summary="invalid physical characterization",
                ),
            ),
        )


def test_causal_necessity_requires_contiguous_passes() -> None:
    gates = []
    for tier in (
        EvidenceTier.DESCRIPTIVE,
        EvidenceTier.PREDICTIVE,
        EvidenceTier.ADVERSARY_SURVIVING,
        EvidenceTier.SOURCE_STABILITY,
        EvidenceTier.REPEATED_CASE,
        EvidenceTier.CAUSAL_MECHANISTIC,
    ):
        gates.append(
            EvidenceGate(
                id=tier.name.lower(),
                tier=tier,
                status=GateStatus.PASS,
                summary=f"{tier.name.lower()} preregistered gate passed",
                threshold=f"declared-{tier.name.lower()}-criterion",
            )
        )
    gates.append(
        EvidenceGate(
            id="physical",
            tier=EvidenceTier.PHYSICAL_QUANTUM,
            status=GateStatus.NOT_APPLICABLE,
            summary="quantum-inspired claim ceiling",
        )
    )
    validate_monotonic_promotion(gates[:-1])
    profile = MechanismNecessityProfile(
        mechanism_id="lindblad_latent_dynamics",
        claim_class=ClaimClass.QUANTUM_INSPIRED,
        signature=bmrb_dynamics_signature(),
        gates=tuple(gates),
    )
    assert profile.promotion_ceiling == EvidenceTier.CAUSAL_MECHANISTIC
    assert profile.to_mapping()["necessity_claim_permitted"] is True
