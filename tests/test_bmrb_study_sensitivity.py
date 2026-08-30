from __future__ import annotations

import math

import pytest

from quantumbci.bmrb_study_replication import (
    BMRBStudyEvidence,
    BMRBStudyReplicationPolicy,
    BMRBStudyReplicationSlot,
    evaluate_study_replication,
)
from quantumbci.bmrb_study_sensitivity import (
    BMRBStudySensitivityPolicy,
    assess_study_sensitivity,
)
from quantumbci.bmrb_study_sensitivity_stress import run_bmrb_study_sensitivity_stress
from quantumbci.preregistration import PreregistrationEvidence


def _replication_policy() -> BMRBStudyReplicationPolicy:
    provisional = BMRBStudyReplicationPolicy(
        policy_id="sensitivity-test-replication",
        mechanism_id="mechanism-a",
        studies=(
            BMRBStudyReplicationSlot("primary", "d0", "primary", 0, "primary"),
            BMRBStudyReplicationSlot("r1", "d1", "replication", 1, "replication one"),
            BMRBStudyReplicationSlot("r2", "d2", "replication", 2, "replication two"),
            BMRBStudyReplicationSlot("r3", "d3", "replication", 3, "replication three"),
        ),
        min_successful_replications=2,
        scientific_rationale="Synthetic four-study sensitivity fixture.",
    )
    registration = PreregistrationEvidence(
        registration_uri="https://osf.io/example/register/sensitivity-test-replication",
        registered_at="2026-08-30T00:00:00Z",
        registration_document_sha256="b" * 64,
        registered_policy_sha256=provisional.decision_fingerprint,
        registry="OSF",
    )
    return BMRBStudyReplicationPolicy(
        **{**provisional.__dict__, "preregistration": registration}
    )


def _sensitivity_policy(*, registered: bool = True) -> BMRBStudySensitivityPolicy:
    provisional = BMRBStudySensitivityPolicy(
        policy_id="sensitivity-test-v1",
        min_direction_agreement_fraction=0.75,
        max_effect_range=0.08,
        max_leave_one_out_mean_shift=0.04,
        scientific_rationale="Synthetic thresholds for software semantics only.",
    )
    if not registered:
        return provisional
    registration = PreregistrationEvidence(
        registration_uri="https://osf.io/example/register/sensitivity-test-v1",
        registered_at="2026-08-30T00:00:00Z",
        registration_document_sha256="c" * 64,
        registered_policy_sha256=provisional.decision_fingerprint,
        registry="OSF",
    )
    return BMRBStudySensitivityPolicy(
        **{**provisional.__dict__, "preregistration": registration}
    )


def _evidence(
    study_id: str,
    dataset_id: str,
    *,
    effect: float,
    passed: bool,
    source_char: str,
) -> BMRBStudyEvidence:
    return BMRBStudyEvidence(
        study_id=study_id,
        dataset_id=dataset_id,
        mechanism_id="mechanism-a",
        participant_count=20,
        scientific_criteria_passed=passed,
        confirmatory_authority=True,
        promotion_eligible=passed,
        reference_effect=effect,
        reference_ci_lower=effect - 0.02,
        reference_ci_upper=effect + 0.02,
        source_fingerprint=source_char * 64,
    )


def _decision(*, fragile: bool):
    last_effect = -0.10 if fragile else 0.09
    last_pass = not fragile
    return evaluate_study_replication(
        _replication_policy(),
        (
            _evidence("primary", "d0", effect=0.12, passed=True, source_char="1"),
            _evidence("r1", "d1", effect=0.11, passed=True, source_char="2"),
            _evidence("r2", "d2", effect=0.10, passed=True, source_char="3"),
            _evidence("r3", "d3", effect=last_effect, passed=last_pass, source_char="4"),
        ),
    )


def test_fragile_and_robust_passes_are_distinguished_without_changing_promotion() -> None:
    fragile = assess_study_sensitivity(_decision(fragile=True), policy=_sensitivity_policy())
    robust = assess_study_sensitivity(_decision(fragile=False), policy=_sensitivity_policy())

    assert fragile.replication.broad_claim_promotion_eligible is True
    assert robust.replication.broad_claim_promotion_eligible is True
    assert fragile.successful_replication_margin == 0
    assert fragile.single_successful_replication_removal_flips_claim is True
    assert fragile.heterogeneity_criteria_passed is False
    assert fragile.sensitivity_warning is True
    assert robust.successful_replication_margin == 1
    assert robust.single_successful_replication_removal_flips_claim is False
    assert robust.heterogeneity_criteria_passed is True
    assert robust.sensitivity_warning is False
    assert fragile.effect_range > robust.effect_range
    assert fragile.max_leave_one_out_mean_shift > robust.max_leave_one_out_mean_shift
    assert fragile.to_mapping()["replication_promotion_decision_unchanged"] is True


def test_leave_one_out_reports_every_study_once_and_identifies_conflicting_study() -> None:
    result = assess_study_sensitivity(_decision(fragile=True), policy=_sensitivity_policy())
    assert {point.removed_study_id for point in result.leave_one_study_out} == {
        "primary",
        "r1",
        "r2",
        "r3",
    }
    assert all(point.remaining_study_count == 3 for point in result.leave_one_study_out)
    assert result.most_influential_study_id == "r3"
    assert result.max_leave_one_out_mean_shift > 0.05


def test_direction_agreement_is_study_weighted_not_participant_weighted() -> None:
    decision = evaluate_study_replication(
        _replication_policy(),
        (
            BMRBStudyEvidence(
                study_id="primary",
                dataset_id="d0",
                mechanism_id="mechanism-a",
                participant_count=1000,
                scientific_criteria_passed=True,
                confirmatory_authority=True,
                promotion_eligible=True,
                reference_effect=0.12,
                reference_ci_lower=0.10,
                reference_ci_upper=0.14,
                source_fingerprint="5" * 64,
            ),
            _evidence("r1", "d1", effect=0.11, passed=True, source_char="6"),
            _evidence("r2", "d2", effect=0.10, passed=True, source_char="7"),
            _evidence("r3", "d3", effect=-0.10, passed=False, source_char="8"),
        ),
    )
    result = assess_study_sensitivity(decision, policy=_sensitivity_policy())
    assert result.direction_agreement_fraction == 0.75


def test_sensitivity_requires_three_independent_studies() -> None:
    policy = BMRBStudyReplicationPolicy(
        policy_id="two-study",
        mechanism_id="mechanism-a",
        studies=(
            BMRBStudyReplicationSlot("primary", "d0", "primary", 0, "primary"),
            BMRBStudyReplicationSlot("r1", "d1", "replication", 1, "replication"),
        ),
        min_successful_replications=1,
        scientific_rationale="Two-study fixture.",
    )
    decision = evaluate_study_replication(
        policy,
        (
            _evidence("primary", "d0", effect=0.12, passed=True, source_char="9"),
            _evidence("r1", "d1", effect=0.11, passed=True, source_char="a"),
        ),
    )
    with pytest.raises(ValueError, match="at least three independent studies"):
        assess_study_sensitivity(decision, policy=_sensitivity_policy())


def test_policy_round_trip_and_preregistration_binding() -> None:
    policy = _sensitivity_policy()
    restored = BMRBStudySensitivityPolicy.from_mapping(policy.to_mapping())
    assert restored.decision_fingerprint == policy.decision_fingerprint
    assert restored.confirmatory_authority is True

    changed = BMRBStudySensitivityPolicy(
        **{**policy.__dict__, "max_effect_range": 0.2}
    )
    assert changed.decision_fingerprint != policy.decision_fingerprint
    assert changed.confirmatory_authority is False

    payload = policy.to_mapping()
    payload["promotion_authoritative"] = True
    with pytest.raises(ValueError, match="not promotion-authoritative"):
        BMRBStudySensitivityPolicy.from_mapping(payload)


def test_policy_fails_closed_on_nonfinite_or_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="\[0, 1\]"):
        BMRBStudySensitivityPolicy(
            policy_id="bad",
            min_direction_agreement_fraction=1.1,
            max_effect_range=0.1,
            max_leave_one_out_mean_shift=0.1,
            scientific_rationale="invalid",
        )
    with pytest.raises(ValueError, match="must be finite"):
        BMRBStudySensitivityPolicy(
            policy_id="bad",
            min_direction_agreement_fraction=0.8,
            max_effect_range=math.nan,
            max_leave_one_out_mean_shift=0.1,
            scientific_rationale="invalid",
        )
    with pytest.raises(ValueError, match="non-negative"):
        BMRBStudySensitivityPolicy(
            policy_id="bad",
            min_direction_agreement_fraction=0.8,
            max_effect_range=0.1,
            max_leave_one_out_mean_shift=-0.1,
            scientific_rationale="invalid",
        )


def test_preregistered_diagnostic_does_not_become_promotion_authority() -> None:
    result = assess_study_sensitivity(_decision(fragile=False), policy=_sensitivity_policy())
    mapping = result.to_mapping()
    assert result.policy.confirmatory_authority is True
    assert mapping["promotion_authoritative"] is False
    assert mapping["replication_broad_claim_promotion_eligible"] is True
    assert mapping["physical_quantum_promotion_eligible"] is False


def test_installed_sensitivity_stress_separates_fragile_from_redundant_pass() -> None:
    result = run_bmrb_study_sensitivity_stress()
    assert result["both_replication_decisions_pass"] is True
    assert result["both_replication_promotions_remain_eligible"] is True
    assert result["fragile_has_zero_success_margin"] is True
    assert result["fragile_single_replication_removal_flips"] is True
    assert result["fragile_sensitivity_warning"] is True
    assert result["robust_has_positive_success_margin"] is True
    assert result["robust_single_replication_removal_does_not_flip"] is True
    assert result["robust_sensitivity_passes"] is True
    assert result["robust_has_no_sensitivity_warning"] is True
    assert result["promotion_decision_unchanged_by_sensitivity"] is True
