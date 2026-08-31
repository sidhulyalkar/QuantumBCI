"""Known-truth software fixtures for study-level heterogeneity and influence diagnostics.

The fixtures contrast a fragile broad replication PASS with a redundant, directionally
consistent PASS. They validate diagnostic semantics only. They do not validate
biological truth, a universal heterogeneity threshold, or a physical-quantum mechanism.
"""

from __future__ import annotations

from typing import Any

from .bmrb_study_replication import (
    BMRBStudyEvidence,
    BMRBStudyReplicationPolicy,
    BMRBStudyReplicationSlot,
    evaluate_study_replication,
)
from .bmrb_study_sensitivity import (
    BMRBStudySensitivityPolicy,
    assess_study_sensitivity,
)
from .preregistration import PreregistrationEvidence

BMRB_STUDY_SENSITIVITY_STRESS_BENCHMARK = "BMRB_STUDY_HETEROGENEITY_STRESS_V1"


def _replication_policy() -> BMRBStudyReplicationPolicy:
    provisional = BMRBStudyReplicationPolicy(
        policy_id="heterogeneity-replication-v1",
        mechanism_id="synthetic-mechanism",
        studies=(
            BMRBStudyReplicationSlot("primary", "dataset-0", "primary", 0, "Frozen primary."),
            BMRBStudyReplicationSlot("rep-1", "dataset-1", "replication", 1, "Replication one."),
            BMRBStudyReplicationSlot("rep-2", "dataset-2", "replication", 2, "Replication two."),
            BMRBStudyReplicationSlot("rep-3", "dataset-3", "replication", 3, "Replication three."),
        ),
        min_successful_replications=2,
        scientific_rationale="Primary plus at least two of three independent replications.",
    )
    registration = PreregistrationEvidence(
        registration_uri="https://osf.io/example/register/heterogeneity-replication-v1",
        registered_at="2026-08-30T00:00:00Z",
        registration_document_sha256="8" * 64,
        registered_policy_sha256=provisional.decision_fingerprint,
        registry="synthetic-software-fixture",
    )
    return BMRBStudyReplicationPolicy(
        **{**provisional.__dict__, "preregistration": registration}
    )


def _sensitivity_policy() -> BMRBStudySensitivityPolicy:
    provisional = BMRBStudySensitivityPolicy(
        policy_id="heterogeneity-sensitivity-v1",
        min_direction_agreement_fraction=0.75,
        max_effect_range=0.08,
        max_leave_one_out_mean_shift=0.04,
        scientific_rationale=(
            "Synthetic thresholds discriminate a deliberately conflicting study from a "
            "compact same-direction fixture; they are not universal biological cutoffs."
        ),
    )
    registration = PreregistrationEvidence(
        registration_uri="https://osf.io/example/register/heterogeneity-sensitivity-v1",
        registered_at="2026-08-30T00:00:00Z",
        registration_document_sha256="9" * 64,
        registered_policy_sha256=provisional.decision_fingerprint,
        registry="synthetic-software-fixture",
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
        mechanism_id="synthetic-mechanism",
        participant_count=30,
        scientific_criteria_passed=passed,
        confirmatory_authority=True,
        promotion_eligible=passed,
        reference_effect=effect,
        reference_ci_lower=effect - 0.02,
        reference_ci_upper=effect + 0.02,
        source_fingerprint=source_char * 64,
    )


def _fragile_replication():
    return evaluate_study_replication(
        _replication_policy(),
        (
            _evidence("primary", "dataset-0", effect=0.12, passed=True, source_char="1"),
            _evidence("rep-1", "dataset-1", effect=0.11, passed=True, source_char="2"),
            _evidence("rep-2", "dataset-2", effect=0.10, passed=True, source_char="3"),
            _evidence("rep-3", "dataset-3", effect=-0.10, passed=False, source_char="4"),
        ),
    )


def _robust_replication():
    return evaluate_study_replication(
        _replication_policy(),
        (
            _evidence("primary", "dataset-0", effect=0.12, passed=True, source_char="5"),
            _evidence("rep-1", "dataset-1", effect=0.11, passed=True, source_char="6"),
            _evidence("rep-2", "dataset-2", effect=0.10, passed=True, source_char="7"),
            _evidence("rep-3", "dataset-3", effect=0.09, passed=True, source_char="a"),
        ),
    )


def run_bmrb_study_sensitivity_stress() -> dict[str, Any]:
    """Contrast zero-margin heterogeneous PASS with redundant compact PASS."""

    policy = _sensitivity_policy()
    fragile = assess_study_sensitivity(_fragile_replication(), policy=policy)
    robust = assess_study_sensitivity(_robust_replication(), policy=policy)
    return {
        "benchmark": BMRB_STUDY_SENSITIVITY_STRESS_BENCHMARK,
        "policy": policy.to_mapping(),
        "fragile": fragile.to_mapping(),
        "robust": robust.to_mapping(),
        "both_replication_decisions_pass": (
            fragile.replication.replication_criteria_passed
            and robust.replication.replication_criteria_passed
        ),
        "both_replication_promotions_remain_eligible": (
            fragile.replication.broad_claim_promotion_eligible
            and robust.replication.broad_claim_promotion_eligible
        ),
        "fragile_has_zero_success_margin": fragile.successful_replication_margin == 0,
        "fragile_single_replication_removal_flips": (
            fragile.single_successful_replication_removal_flips_claim
        ),
        "fragile_sensitivity_warning": fragile.sensitivity_warning,
        "robust_has_positive_success_margin": robust.successful_replication_margin > 0,
        "robust_single_replication_removal_does_not_flip": (
            not robust.single_successful_replication_removal_flips_claim
        ),
        "robust_sensitivity_passes": robust.heterogeneity_criteria_passed,
        "robust_has_no_sensitivity_warning": not robust.sensitivity_warning,
        "fragile_effect_range_exceeds_robust": fragile.effect_range > robust.effect_range,
        "fragile_influence_exceeds_robust": (
            fragile.max_leave_one_out_mean_shift > robust.max_leave_one_out_mean_shift
        ),
        "promotion_decision_unchanged_by_sensitivity": (
            fragile.to_mapping()["replication_promotion_decision_unchanged"]
            and robust.to_mapping()["replication_promotion_decision_unchanged"]
        ),
        "claim_boundary": (
            "This synthetic sensitivity stress does not validate biological truth or a "
            "physical-quantum mechanism."
        ),
    }
