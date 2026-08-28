from __future__ import annotations

import pytest

from quantumbci.confirmatory_representation import (
    ConfirmatoryRepresentationObservation,
    ConfirmatoryRepresentationPolicy,
    build_confirmatory_representation_profile,
    evaluate_confirmatory_representation,
)
from quantumbci.preregistration import PreregistrationEvidence
from quantumbci.recapitulation import EvidenceTier, GateStatus, gate_map


def _policy(
    *,
    primary_budget: int = 10,
    with_registration: bool = True,
    min_candidate: float = 0.05,
    min_ablation: float = 0.05,
) -> ConfirmatoryRepresentationPolicy:
    provisional = ConfirmatoryRepresentationPolicy(
        policy_id="confirmatory-v2",
        reference_representation_id="raw",
        primary_calibration_per_class=primary_budget,
        primary_classical_control="normalized_covariance",
        min_participants=3,
        min_representations=2,
        min_representation_families=2,
        min_candidate_advantage=min_candidate,
        min_ablation_necessity=min_ablation,
        min_reference_positive_fraction=2 / 3,
        min_all_lane_positive_fraction=2 / 3,
        min_direction_match_fraction=2 / 3,
        min_ablation_direction_match_fraction=2 / 3,
        min_information_novel_representation_fraction=1.0,
        sample_size_rationale="Three synthetic participants qualify the software contract only.",
        inference_seed=23,
        bootstrap_resamples=500,
    )
    if not with_registration:
        return provisional
    evidence = PreregistrationEvidence(
        registration_uri="https://osf.io/example/register/confirmatory-v2",
        registered_at="2026-08-01T00:00:00Z",
        registration_document_sha256="a" * 64,
        registered_policy_sha256=provisional.decision_fingerprint,
        registry="OSF",
    )
    return ConfirmatoryRepresentationPolicy(
        **{
            **provisional.__dict__,
            "preregistration": evidence,
        }
    )


def _observation(
    lane: str,
    family: str,
    participant: str,
    *,
    budget: int,
    advantage: float,
    ablation: float,
    novel: bool = True,
) -> ConfirmatoryRepresentationObservation:
    candidate = 0.8
    model = None if family == "raw_neural" else lane.upper()
    return ConfirmatoryRepresentationObservation(
        participant_id=participant,
        occasion_id="ses-5",
        case_id=f"{participant}-ses-5",
        calibration_per_class=budget,
        representation_id=lane,
        representation_family=family,
        authority_fingerprint=f"authority-{participant}-ses-5-{budget}",
        representation_sha256=(lane[0] * 64),
        source_fingerprint=(participant[-1] * 64),
        candidate_metric=candidate,
        primary_control_metric=candidate - advantage,
        ablated_metric=candidate - ablation,
        information_novel=novel,
        model_id=model,
        model_revision=None if model is None else "rev-1",
    )


def _budget_reversal_cases(*, novel: bool = True) -> list[ConfirmatoryRepresentationObservation]:
    rows: list[ConfirmatoryRepresentationObservation] = []
    for participant in ("p1", "p2", "p3"):
        for lane, family in (("raw", "raw_neural"), ("labram", "foundation_model")):
            rows.append(
                _observation(
                    lane,
                    family,
                    participant,
                    budget=0,
                    advantage=-0.20,
                    ablation=-0.10,
                    novel=novel,
                )
            )
            rows.append(
                _observation(
                    lane,
                    family,
                    participant,
                    budget=10,
                    advantage=0.20,
                    ablation=0.15,
                    novel=novel,
                )
            )
    return rows


def test_primary_budget_is_not_averaged_with_secondary_frontier() -> None:
    result = evaluate_confirmatory_representation(
        _budget_reversal_cases(),
        study_id="budget-reversal",
        mechanism_id="fixture",
        policy=_policy(primary_budget=10),
    )
    assert result.available_calibration_budgets == (0, 10)
    assert result.scientific_criteria_passed is True
    assert result.promotion_eligible is True
    assert all(lane.candidate.observed_mean == pytest.approx(0.20) for lane in result.lanes)
    frontier = {point.calibration_per_class: point for point in result.calibration_frontier}
    assert frontier[0].lane_mean_candidate_advantage["raw"] == pytest.approx(-0.20)
    assert frontier[10].lane_mean_candidate_advantage["raw"] == pytest.approx(0.20)

    failed = evaluate_confirmatory_representation(
        _budget_reversal_cases(),
        study_id="budget-reversal",
        mechanism_id="fixture",
        policy=_policy(primary_budget=0),
    )
    assert failed.scientific_criteria_passed is False
    assert failed.promotion_eligible is False


def test_external_registration_must_match_exact_policy_fingerprint() -> None:
    policy = _policy(with_registration=True)
    assert policy.confirmatory_authority is True
    changed = ConfirmatoryRepresentationPolicy(
        **{
            **policy.__dict__,
            "min_candidate_advantage": policy.min_candidate_advantage + 0.01,
        }
    )
    assert changed.confirmatory_authority is False
    assert changed.decision_fingerprint != policy.decision_fingerprint


def test_retrospective_result_is_characterized_not_confirmatory_pass() -> None:
    result = evaluate_confirmatory_representation(
        _budget_reversal_cases(),
        study_id="retrospective",
        mechanism_id="fixture",
        policy=_policy(with_registration=False),
    )
    assert result.scientific_criteria_passed is True
    assert result.promotion_eligible is False
    profile = build_confirmatory_representation_profile(result)
    gates = gate_map(profile)
    assert gates["paired_representation_authority"].status == GateStatus.PASS
    assert gates["confirmatory_primary_effect"].status == GateStatus.CHARACTERIZED
    assert gates["matched_representation_adversaries"].status == GateStatus.CHARACTERIZED
    assert profile.promotion_ceiling == EvidenceTier.DESCRIPTIVE


def test_classical_equivalence_fails_adversary_without_erasing_predictive_effect() -> None:
    result = evaluate_confirmatory_representation(
        _budget_reversal_cases(novel=False),
        study_id="equivalence-null",
        mechanism_id="density_second_moment",
        policy=_policy(),
    )
    assert result.promotion_eligible is False
    profile = build_confirmatory_representation_profile(result)
    gates = gate_map(profile)
    assert gates["confirmatory_primary_effect"].status == GateStatus.PASS
    assert gates["matched_representation_adversaries"].status == GateStatus.FAIL
    assert gates["cross_representation_stability"].status == GateStatus.CHARACTERIZED
    assert profile.first_failing_gate == "matched_representation_adversaries"
    assert profile.promotion_ceiling == EvidenceTier.PREDICTIVE


def test_primary_control_is_explicit_and_not_selected_from_evaluation() -> None:
    policy = _policy()
    assert policy.primary_classical_control == "normalized_covariance"
    payload = policy.to_mapping()
    assert payload["primary_classical_control"] == "normalized_covariance"
    assert payload["decision_fingerprint"] == policy.decision_fingerprint
