from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

import quantumbci.kumar2024_e001_decision_authority as authority
from quantumbci.kumar2024_e001_decision_authority import (
    E001_CLASSICAL_CONTROLS,
    KUMAR2024_AUTHORITY_CAPSULE_FINGERPRINT,
    KUMAR2024_COHORT_AUTHORITY_FINGERPRINT,
    KUMAR2024_RAW_DATASET_FINGERPRINT,
    Kumar2024E001BootstrapAuthority,
    Kumar2024E001DecisionPlan,
    Kumar2024E001PreregistrationSeal,
    Kumar2024E001PrimaryCriterion,
)
from quantumbci.preregistration import PreregistrationEvidence

CAPSULE = Path("evidence/kumar2024-authority-freeze-v1/authority-capsule.json")

# These values are software-test fixtures only. They are not scientific recommendations
# or preregistered production thresholds.
FIXTURE_MINIMUM_EFFECT = 0.01
FIXTURE_PRIMARY_BUDGET = 10
FIXTURE_BOOTSTRAP_RESAMPLES = 500
FIXTURE_BOOTSTRAP_SEED = 20260902


def _criterion(
    *,
    minimum_effect: float = FIXTURE_MINIMUM_EFFECT,
    calibration_per_class: int = FIXTURE_PRIMARY_BUDGET,
) -> Kumar2024E001PrimaryCriterion:
    return Kumar2024E001PrimaryCriterion(
        calibration_per_class=calibration_per_class,
        minimum_effect=minimum_effect,
        rationale="software fixture only; production value requires external scientific justification",
    )


def _plan() -> Kumar2024E001DecisionPlan:
    return Kumar2024E001DecisionPlan.from_verified_authority_capsule(
        CAPSULE,
        primary_criterion=_criterion(),
        bootstrap=Kumar2024E001BootstrapAuthority(
            n_resamples=FIXTURE_BOOTSTRAP_RESAMPLES,
            seed=FIXTURE_BOOTSTRAP_SEED,
        ),
        rationale=(
            "software fixture plan for validating authority serialization; "
            "not a production preregistration"
        ),
    )


def _plan_fingerprint(payload: dict) -> str:
    value = deepcopy(payload)
    value.pop("plan_fingerprint", None)
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(authority.KUMAR2024_E001_DOMAIN + encoded).hexdigest()


def _seal_fingerprint(payload: dict) -> str:
    value = deepcopy(payload)
    value.pop("artifact_fingerprint", None)
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(authority.KUMAR2024_E001_SEAL_DOMAIN + encoded).hexdigest()


def _preregistration(plan: Kumar2024E001DecisionPlan) -> PreregistrationEvidence:
    return PreregistrationEvidence(
        registration_uri="https://example.invalid/software-fixture-registration",
        registered_at="2026-09-02T12:00:00Z",
        registration_document_sha256="a" * 64,
        registered_policy_sha256=plan.fingerprint,
        registry="software-fixture",
    )


def test_verified_capsule_builds_exact_outcome_blind_decision_authority() -> None:
    plan = _plan()
    payload = plan.to_mapping()
    assert payload["authority"]["capsule_fingerprint"] == (
        KUMAR2024_AUTHORITY_CAPSULE_FINGERPRINT
    )
    assert payload["authority"]["cohort_authority_fingerprint"] == (
        KUMAR2024_COHORT_AUTHORITY_FINGERPRINT
    )
    assert payload["authority"]["raw_dataset_fingerprint"] == (
        KUMAR2024_RAW_DATASET_FINGERPRINT
    )
    assert payload["authority"]["subjects"] == list(range(1, 19))
    assert len(payload["authority"]["case_authorities"]) == 18
    assert payload["decision_semantics"]["evaluation_executed"] is False
    assert payload["decision_semantics"]["confirmatory_outcomes_observed"] is False
    assert (
        payload["decision_semantics"]["information_novelty_promotion_eligible"]
        is False
    )
    assert payload["decision_semantics"]["physical_quantum_promotion_eligible"] is False


def test_plan_round_trip_is_canonical_and_control_family_matches_production() -> None:
    plan = _plan()
    restored = Kumar2024E001DecisionPlan.from_mapping(plan.to_mapping())
    assert restored.to_mapping() == plan.to_mapping()
    controls = restored.control_authority.to_mapping()
    assert tuple(controls["classical_controls"]) == E001_CLASSICAL_CONTROLS
    assert controls["candidate"] == "density"
    assert controls["exact_equivalence_control"] == "normalized_covariance"
    assert controls["ablation"] == "offdiagonal_ablation"
    assert controls["strongest_classical_control_promotion_authoritative"] is False


def test_primary_threshold_is_explicit_and_cohort_supported() -> None:
    with pytest.raises(ValueError, match="minimum_effect must be finite"):
        Kumar2024E001PrimaryCriterion.from_mapping(
            {
                "estimand": authority.E001_PRIMARY_ESTIMAND,
                "control": "offdiagonal_ablation",
                "calibration_per_class": 10,
                "statistic": authority.E001_PRIMARY_STATISTIC,
                "comparison": "greater_than_or_equal",
                "rationale": "missing threshold must fail",
            }
        )
    with pytest.raises(ValueError, match="cohort-wide supported minimum"):
        Kumar2024E001DecisionPlan.from_verified_authority_capsule(
            CAPSULE,
            primary_criterion=_criterion(calibration_per_class=15),
            bootstrap=Kumar2024E001BootstrapAuthority(n_resamples=500, seed=1),
            rationale="software fixture",
        )


def test_case_authority_tampering_is_rejected_even_with_fresh_plan_fingerprint() -> None:
    payload = deepcopy(_plan().to_mapping())
    payload["authority"]["case_authorities"][0]["authority_fingerprint"] = "0" * 16
    payload["plan_fingerprint"] = _plan_fingerprint(payload)
    with pytest.raises(ValueError, match="do not reproduce the frozen cohort fingerprint"):
        Kumar2024E001DecisionPlan.from_mapping(payload)


def test_control_family_cannot_drift_even_after_refingerprinting() -> None:
    payload = deepcopy(_plan().to_mapping())
    payload["control_authority"]["classical_controls"][-1] = "invented_control"
    payload["plan_fingerprint"] = _plan_fingerprint(payload)
    with pytest.raises(ValueError, match="classical_controls"):
        Kumar2024E001DecisionPlan.from_mapping(payload)


def test_information_novelty_hard_ceiling_cannot_be_promoted() -> None:
    payload = deepcopy(_plan().to_mapping())
    payload["decision_semantics"]["information_novelty_promotion_eligible"] = True
    payload["plan_fingerprint"] = _plan_fingerprint(payload)
    with pytest.raises(ValueError, match="decision semantics drifted"):
        Kumar2024E001DecisionPlan.from_mapping(payload)


def test_participant_inference_missing_evidence_and_subgroups_are_fixed() -> None:
    payload = deepcopy(_plan().to_mapping())
    payload["bootstrap"]["inference_unit"] = "session"
    payload["plan_fingerprint"] = _plan_fingerprint(payload)
    with pytest.raises(ValueError, match="participant/subject"):
        Kumar2024E001DecisionPlan.from_mapping(payload)

    payload = deepcopy(_plan().to_mapping())
    payload["evidence_handling"]["invalid_evidence_is_scientific_null"] = True
    payload["plan_fingerprint"] = _plan_fingerprint(payload)
    with pytest.raises(ValueError, match="scientific null"):
        Kumar2024E001DecisionPlan.from_mapping(payload)

    payload = deepcopy(_plan().to_mapping())
    payload["subgroup_authority"]["promotion_authoritative"] = True
    payload["plan_fingerprint"] = _plan_fingerprint(payload)
    with pytest.raises(ValueError, match="promotion_authoritative"):
        Kumar2024E001DecisionPlan.from_mapping(payload)


def test_external_preregistration_must_bind_exact_plan() -> None:
    plan = _plan()
    seal = Kumar2024E001PreregistrationSeal(
        plan=plan,
        preregistration=_preregistration(plan),
    )
    restored = Kumar2024E001PreregistrationSeal.from_mapping(seal.to_mapping())
    assert restored.to_mapping() == seal.to_mapping()
    with pytest.raises(ValueError, match="does not bind"):
        Kumar2024E001PreregistrationSeal(
            plan=plan,
            preregistration=PreregistrationEvidence(
                registration_uri="https://example.invalid/wrong-policy",
                registered_at="2026-09-02T12:00:00Z",
                registration_document_sha256="b" * 64,
                registered_policy_sha256="c" * 64,
                registry="software-fixture",
            ),
        )


def test_seal_rejects_claim_escalation_even_after_outer_refingerprinting() -> None:
    plan = _plan()
    payload = Kumar2024E001PreregistrationSeal(
        plan=plan,
        preregistration=_preregistration(plan),
    ).to_mapping()
    payload["physical_quantum_promotion_eligible"] = True
    payload["artifact_fingerprint"] = _seal_fingerprint(payload)
    with pytest.raises(ValueError, match="physical_quantum_promotion_eligible"):
        Kumar2024E001PreregistrationSeal.from_mapping(payload)


def test_missing_registration_uri_is_rejected_strictly() -> None:
    plan = _plan()
    payload = Kumar2024E001PreregistrationSeal(
        plan=plan,
        preregistration=_preregistration(plan),
    ).to_mapping()
    payload["preregistration"]["registration_uri"] = None
    payload["artifact_fingerprint"] = _seal_fingerprint(payload)
    with pytest.raises(ValueError, match="registration_uri must not be null"):
        Kumar2024E001PreregistrationSeal.from_mapping(payload)
