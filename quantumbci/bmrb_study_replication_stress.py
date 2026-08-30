"""Known-truth software stress for study-level BMRB replication authority.

The stress demonstrates a specific hierarchy failure: participant-count weighting can
reverse a descriptive headline when study sizes are swapped, even though the scientific
pattern is unchanged. The promotion-authoritative decision therefore counts independent
studies, not participant rows.

This synthetic software stress does not validate biological truth, a universal
replication threshold, or any physical-quantum mechanism claim.
"""

from __future__ import annotations

from typing import Any

from .bmrb_study_replication import (
    BMRBStudyEvidence,
    BMRBStudyReplicationPolicy,
    BMRBStudyReplicationSlot,
    evaluate_study_replication,
)
from .preregistration import PreregistrationEvidence

BMRB_STUDY_REPLICATION_STRESS_BENCHMARK = "BMRB_STUDY_REPLICATION_IMBALANCE_STRESS_V1"


def _registered_policy() -> BMRBStudyReplicationPolicy:
    provisional = BMRBStudyReplicationPolicy(
        policy_id="study-replication-stress-v1",
        mechanism_id="synthetic-mechanism",
        studies=(
            BMRBStudyReplicationSlot(
                study_id="primary-study",
                dataset_id="dataset-primary",
                role="primary",
                order=0,
                rationale="Frozen primary software fixture.",
            ),
            BMRBStudyReplicationSlot(
                study_id="replication-study",
                dataset_id="dataset-replication",
                role="replication",
                order=1,
                rationale="Independent replication software fixture.",
            ),
        ),
        min_successful_replications=1,
        scientific_rationale=(
            "The stress requires the primary and one independent replication to pass. "
            "Participant imbalance is diagnostic only."
        ),
    )
    registration = PreregistrationEvidence(
        registration_uri="https://osf.io/example/register/study-replication-stress-v1",
        registered_at="2026-08-30T00:00:00Z",
        registration_document_sha256="b" * 64,
        registered_policy_sha256=provisional.decision_fingerprint,
        registry="synthetic-software-fixture",
    )
    return BMRBStudyReplicationPolicy(
        **{
            **provisional.__dict__,
            "preregistration": registration,
        }
    )


def _evidence(
    *,
    study_id: str,
    dataset_id: str,
    participant_count: int,
    passed: bool,
    source_char: str,
) -> BMRBStudyEvidence:
    effect = 0.12 if passed else 0.01
    return BMRBStudyEvidence(
        study_id=study_id,
        dataset_id=dataset_id,
        mechanism_id="synthetic-mechanism",
        participant_count=participant_count,
        scientific_criteria_passed=passed,
        confirmatory_authority=True,
        promotion_eligible=passed,
        reference_effect=effect,
        reference_ci_lower=effect - 0.02,
        reference_ci_upper=effect + 0.02,
        source_fingerprint=source_char * 64,
    )


def _primary_pass_replication_fail(
    *,
    primary_participants: int,
    replication_participants: int,
):
    policy = _registered_policy()
    return evaluate_study_replication(
        policy,
        (
            _evidence(
                study_id="primary-study",
                dataset_id="dataset-primary",
                participant_count=primary_participants,
                passed=True,
                source_char="1",
            ),
            _evidence(
                study_id="replication-study",
                dataset_id="dataset-replication",
                participant_count=replication_participants,
                passed=False,
                source_char="2",
            ),
        ),
    )


def run_bmrb_study_replication_stress(
    *,
    large_study_participants: int = 500,
    small_study_participants: int = 20,
) -> dict[str, Any]:
    """Swap study sizes while keeping the scientific PASS/FAIL pattern unchanged."""

    large_primary = _primary_pass_replication_fail(
        primary_participants=large_study_participants,
        replication_participants=small_study_participants,
    )
    small_primary = _primary_pass_replication_fail(
        primary_participants=small_study_participants,
        replication_participants=large_study_participants,
    )
    policy = _registered_policy()
    positive_control = evaluate_study_replication(
        policy,
        (
            _evidence(
                study_id="primary-study",
                dataset_id="dataset-primary",
                participant_count=small_study_participants,
                passed=True,
                source_char="3",
            ),
            _evidence(
                study_id="replication-study",
                dataset_id="dataset-replication",
                participant_count=large_study_participants,
                passed=True,
                source_char="4",
            ),
        ),
    )

    first_weighted = large_primary.participant_weighted_positive_fraction
    second_weighted = small_primary.participant_weighted_positive_fraction
    return {
        "benchmark": BMRB_STUDY_REPLICATION_STRESS_BENCHMARK,
        "policy": policy.to_mapping(),
        "large_primary": large_primary.to_mapping(),
        "small_primary": small_primary.to_mapping(),
        "official_decision_invariant_to_participant_swap": (
            large_primary.replication_criteria_passed
            == small_primary.replication_criteria_passed
            == False
        ),
        "participant_weighted_positive_fraction_large_primary": first_weighted,
        "participant_weighted_positive_fraction_small_primary": second_weighted,
        "participant_weighting_reverses_majority_headline": (
            first_weighted > 0.5 and second_weighted < 0.5
        ),
        "study_positive_fraction_invariant": (
            large_primary.study_positive_fraction == small_primary.study_positive_fraction == 0.5
        ),
        "context_specific_primary_preserved": (
            large_primary.context_specific_only
            and small_primary.context_specific_only
            and large_primary.positive_studies == ("primary-study",)
            and small_primary.positive_studies == ("primary-study",)
        ),
        "positive_control_broad_replication_passed": positive_control.replication_criteria_passed,
        "positive_control_broad_claim_promotion_eligible": (
            positive_control.broad_claim_promotion_eligible
        ),
        "claim_boundary": (
            "This synthetic software stress does not validate biological truth or a "
            "physical-quantum mechanism."
        ),
    }
