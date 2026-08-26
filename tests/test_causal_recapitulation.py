from __future__ import annotations

import pytest

from quantumbci.causal_recapitulation import (
    CausalCaseEvidence,
    CausalNecessityPolicy,
    MatchedClassicalRecovery,
    attach_causal_evidence,
    evaluate_causal_necessity,
)
from quantumbci.claims import ClaimClass
from quantumbci.recapitulation import (
    EvidenceGate,
    EvidenceTier,
    GateStatus,
    MechanismNecessityProfile,
    bmrb_dynamics_signature,
)


def _recovery(participant: str, value: float = 0.10) -> MatchedClassicalRecovery:
    return MatchedClassicalRecovery(
        classical_model_id="matched_nonlinear_control",
        classical_recovery_fraction=value,
        information_set_id="same-evidence-budget-v1",
        source_fingerprint=f"recovery-{participant}-{value}",
    )


def _case(
    participant: str,
    occasion: str = "s1",
    *,
    recovery: float = 0.10,
    direction: float = 0.6,
    dose_passed: bool = True,
    faithfulness_passed: bool = True,
    necessity: float = 0.75,
    random_percentile: float = 0.99,
) -> CausalCaseEvidence:
    return CausalCaseEvidence(
        participant_id=participant,
        occasion_id=occasion,
        case_id=f"{participant}-{occasion}",
        mechanism_id="lindblad_latent_dynamics",
        intervention_id="erase_candidate_information",
        dose_response_passed=dose_passed,
        oriented_endpoint_effect=direction,
        mean_monotonic_fraction=0.95,
        faithfulness_passed=faithfulness_passed,
        sufficiency_fraction=0.88,
        necessity_fraction=necessity,
        joint_random_percentile=random_percentile,
        matched_recovery=_recovery(participant, recovery),
        dose_source_fingerprint=f"dose-{participant}-{occasion}",
        faithfulness_source_fingerprint=f"faith-{participant}-{occasion}",
        source_schemas=(
            "neuros-mechint.dose-response-study.v1",
            "neuros-mechint.evidence-pack.v1",
        ),
    )


def _policy(*, preregistered: bool = True) -> CausalNecessityPolicy:
    return CausalNecessityPolicy(
        policy_id="bmrb-causal-fixture-v1",
        preregistered=preregistered,
        min_participants=3,
        min_direction_match_fraction=0.80,
        min_dose_response_pass_fraction=0.80,
        min_faithfulness_pass_fraction=0.80,
        min_mean_necessity_fraction=0.50,
        min_mean_joint_random_percentile=0.95,
        max_mean_classical_recovery_fraction=0.25,
    )


def _profile(*, upstream_passed: bool) -> MechanismNecessityProfile:
    gates = []
    for tier, gate_id in (
        (EvidenceTier.DESCRIPTIVE, "descriptive_contract"),
        (EvidenceTier.PREDICTIVE, "predictive_sufficiency"),
        (EvidenceTier.ADVERSARY_SURVIVING, "matched_classical_adversaries"),
        (EvidenceTier.SOURCE_STABILITY, "source_resampling_stability"),
        (EvidenceTier.REPEATED_CASE, "repeated_case_reliability"),
    ):
        if tier is EvidenceTier.DESCRIPTIVE or upstream_passed:
            gates.append(
                EvidenceGate(
                    id=gate_id,
                    tier=tier,
                    status=GateStatus.PASS,
                    summary=f"{gate_id} passed fixture criterion",
                    threshold=f"fixture-{gate_id}-criterion",
                )
            )
        else:
            gates.append(
                EvidenceGate(
                    id=gate_id,
                    tier=tier,
                    status=GateStatus.CHARACTERIZED,
                    summary=f"{gate_id} measured without promotion criterion",
                )
            )
    gates.extend(
        (
            EvidenceGate(
                id="causal_intervention_and_ablation",
                tier=EvidenceTier.CAUSAL_MECHANISTIC,
                status=GateStatus.NOT_RUN,
                summary="causal evidence not yet attached",
            ),
            EvidenceGate(
                id="physical_quantum_witness",
                tier=EvidenceTier.PHYSICAL_QUANTUM,
                status=GateStatus.NOT_APPLICABLE,
                summary="quantum-inspired mechanism",
            ),
        )
    )
    return MechanismNecessityProfile(
        mechanism_id="lindblad_latent_dynamics",
        claim_class=ClaimClass.QUANTUM_INSPIRED,
        signature=bmrb_dynamics_signature(),
        gates=tuple(gates),
    )


def test_preregistered_causal_evidence_can_promote_only_after_upstream_passes() -> None:
    result = evaluate_causal_necessity(
        [_case("p1"), _case("p2"), _case("p3")],
        policy=_policy(preregistered=True),
    )
    assert result.scientific_criteria_passed is True
    assert result.promotion_eligible is True

    profile = attach_causal_evidence(_profile(upstream_passed=True), result)
    causal = next(
        gate for gate in profile.gates if gate.tier is EvidenceTier.CAUSAL_MECHANISTIC
    )
    assert causal.status is GateStatus.PASS
    assert causal.threshold == result.policy.decision_rule
    assert profile.promotion_ceiling is EvidenceTier.CAUSAL_MECHANISTIC
    assert profile.to_mapping()["necessity_claim_permitted"] is True


def test_successful_retrospective_policy_remains_characterized_not_passed() -> None:
    result = evaluate_causal_necessity(
        [_case("p1"), _case("p2"), _case("p3")],
        policy=_policy(preregistered=False),
    )
    assert result.scientific_criteria_passed is True
    assert result.promotion_eligible is False

    profile = attach_causal_evidence(_profile(upstream_passed=True), result)
    causal = next(
        gate for gate in profile.gates if gate.tier is EvidenceTier.CAUSAL_MECHANISTIC
    )
    assert causal.status is GateStatus.CHARACTERIZED
    assert profile.promotion_ceiling is EvidenceTier.REPEATED_CASE
    assert profile.to_mapping()["necessity_claim_permitted"] is False


def test_causal_pass_cannot_jump_over_unresolved_upstream_tiers() -> None:
    result = evaluate_causal_necessity(
        [_case("p1"), _case("p2"), _case("p3")],
        policy=_policy(preregistered=True),
    )
    profile = attach_causal_evidence(_profile(upstream_passed=False), result)
    causal = next(
        gate for gate in profile.gates if gate.tier is EvidenceTier.CAUSAL_MECHANISTIC
    )
    assert causal.status is GateStatus.CHARACTERIZED
    assert profile.promotion_ceiling is EvidenceTier.DESCRIPTIVE
    assert "upstream BMRB promotion" in causal.summary


def test_matched_classical_recovery_is_an_independent_causal_falsifier() -> None:
    result = evaluate_causal_necessity(
        [
            _case("p1", recovery=0.70),
            _case("p2", recovery=0.65),
            _case("p3", recovery=0.60),
        ],
        policy=_policy(preregistered=True),
    )
    assert result.scientific_criteria_passed is False
    assert any("matched-classical recovery" in reason for reason in result.reasons)

    profile = attach_causal_evidence(_profile(upstream_passed=False), result)
    causal = next(
        gate for gate in profile.gates if gate.tier is EvidenceTier.CAUSAL_MECHANISTIC
    )
    assert causal.status is GateStatus.FAIL
    assert profile.first_failing_gate == "causal_intervention_and_ablation"


def test_participant_balancing_prevents_many_sessions_from_overweighting_one_person() -> None:
    cases = [
        _case("p1", "s1", recovery=0.0, necessity=1.0),
        _case("p1", "s2", recovery=0.0, necessity=1.0),
        _case("p1", "s3", recovery=0.0, necessity=1.0),
        _case("p2", "s1", recovery=0.3, necessity=0.4),
        _case("p3", "s1", recovery=0.3, necessity=0.4),
    ]
    result = evaluate_causal_necessity(cases, policy=_policy(preregistered=True))
    # Participant means are averaged equally: (1.0 + 0.4 + 0.4) / 3 = 0.6.
    assert result.mean_necessity_fraction == pytest.approx(0.6)
    # Likewise recovery is (0.0 + 0.3 + 0.3) / 3 = 0.2, not case-weighted 0.12.
    assert result.mean_classical_recovery_fraction == pytest.approx(0.2)


def test_duplicate_participant_occasion_fails_closed() -> None:
    with pytest.raises(ValueError, match="duplicate participant/occasion"):
        evaluate_causal_necessity(
            [_case("p1", "s1"), _case("p1", "s1"), _case("p2"), _case("p3")],
            policy=_policy(),
        )
