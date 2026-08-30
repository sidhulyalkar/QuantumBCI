from __future__ import annotations

import math

import pytest

from quantumbci.bmrb_study_replication import (
    BMRBStudyEvidence,
    BMRBStudyReplicationPolicy,
    BMRBStudyReplicationSlot,
    evaluate_study_replication,
)
from quantumbci.bmrb_study_replication_stress import run_bmrb_study_replication_stress
from quantumbci.confirmatory_representation import (
    ConfirmatoryRepresentationObservation,
    ConfirmatoryRepresentationPolicy,
    evaluate_confirmatory_representation,
)
from quantumbci.preregistration import PreregistrationEvidence


def _policy(*, registered: bool = True) -> BMRBStudyReplicationPolicy:
    provisional = BMRBStudyReplicationPolicy(
        policy_id="replication-v1",
        mechanism_id="mechanism-a",
        studies=(
            BMRBStudyReplicationSlot(
                study_id="study-primary",
                dataset_id="dataset-a",
                role="primary",
                order=0,
                rationale="Frozen primary study.",
            ),
            BMRBStudyReplicationSlot(
                study_id="study-replication",
                dataset_id="dataset-b",
                role="replication",
                order=1,
                rationale="Frozen independent replication study.",
            ),
        ),
        min_successful_replications=1,
        scientific_rationale="Broad promotion requires the primary and replication to pass.",
    )
    if not registered:
        return provisional
    registration = PreregistrationEvidence(
        registration_uri="https://osf.io/example/register/replication-v1",
        registered_at="2026-08-30T00:00:00Z",
        registration_document_sha256="a" * 64,
        registered_policy_sha256=provisional.decision_fingerprint,
        registry="OSF",
    )
    return BMRBStudyReplicationPolicy(
        **{
            **provisional.__dict__,
            "preregistration": registration,
        }
    )


def _evidence(
    study_id: str,
    dataset_id: str,
    *,
    participants: int,
    passed: bool,
    source_char: str,
    authority: bool = True,
) -> BMRBStudyEvidence:
    effect = 0.12 if passed else 0.01
    return BMRBStudyEvidence(
        study_id=study_id,
        dataset_id=dataset_id,
        mechanism_id="mechanism-a",
        participant_count=participants,
        scientific_criteria_passed=passed,
        confirmatory_authority=authority,
        promotion_eligible=passed and authority,
        reference_effect=effect,
        reference_ci_lower=effect - 0.02,
        reference_ci_upper=effect + 0.02,
        source_fingerprint=source_char * 64,
    )


def test_equal_study_votes_ignore_participant_imbalance() -> None:
    first = evaluate_study_replication(
        _policy(),
        (
            _evidence(
                "study-primary",
                "dataset-a",
                participants=500,
                passed=True,
                source_char="1",
            ),
            _evidence(
                "study-replication",
                "dataset-b",
                participants=20,
                passed=False,
                source_char="2",
            ),
        ),
    )
    swapped = evaluate_study_replication(
        _policy(),
        (
            _evidence(
                "study-primary",
                "dataset-a",
                participants=20,
                passed=True,
                source_char="3",
            ),
            _evidence(
                "study-replication",
                "dataset-b",
                participants=500,
                passed=False,
                source_char="4",
            ),
        ),
    )
    assert first.replication_criteria_passed is False
    assert swapped.replication_criteria_passed is False
    assert first.study_positive_fraction == swapped.study_positive_fraction == 0.5
    assert first.participant_weighted_positive_fraction > 0.9
    assert swapped.participant_weighted_positive_fraction < 0.1
    assert first.context_specific_only is True
    assert swapped.context_specific_only is True
    assert first.positive_studies == swapped.positive_studies == ("study-primary",)


def test_broad_claim_requires_primary_replication_and_authority() -> None:
    evidence = (
        _evidence(
            "study-primary",
            "dataset-a",
            participants=20,
            passed=True,
            source_char="5",
        ),
        _evidence(
            "study-replication",
            "dataset-b",
            participants=500,
            passed=True,
            source_char="6",
        ),
    )
    qualified = evaluate_study_replication(_policy(), evidence)
    assert qualified.replication_criteria_passed is True
    assert qualified.broad_claim_authority is True
    assert qualified.broad_claim_promotion_eligible is True

    retrospective = evaluate_study_replication(_policy(registered=False), evidence)
    assert retrospective.replication_criteria_passed is True
    assert retrospective.broad_claim_authority is False
    assert retrospective.broad_claim_promotion_eligible is False

    unauthorized_study = evaluate_study_replication(
        _policy(),
        (
            evidence[0],
            _evidence(
                "study-replication",
                "dataset-b",
                participants=500,
                passed=True,
                source_char="7",
                authority=False,
            ),
        ),
    )
    assert unauthorized_study.replication_criteria_passed is True
    assert unauthorized_study.all_studies_confirmatory_authority is False
    assert unauthorized_study.broad_claim_promotion_eligible is False


def test_primary_failure_cannot_be_rescued_by_large_replication() -> None:
    result = evaluate_study_replication(
        _policy(),
        (
            _evidence(
                "study-primary",
                "dataset-a",
                participants=20,
                passed=False,
                source_char="8",
            ),
            _evidence(
                "study-replication",
                "dataset-b",
                participants=500,
                passed=True,
                source_char="9",
            ),
        ),
    )
    assert result.successful_replication_studies == ("study-replication",)
    assert result.primary_study_passed is False
    assert result.replication_criteria_passed is False
    assert result.context_specific_only is True
    assert result.positive_studies == ("study-replication",)


def test_complete_frozen_study_family_is_required() -> None:
    primary = _evidence(
        "study-primary", "dataset-a", participants=20, passed=True, source_char="a"
    )
    replication = _evidence(
        "study-replication", "dataset-b", participants=20, passed=True, source_char="b"
    )
    with pytest.raises(ValueError, match="match frozen family exactly"):
        evaluate_study_replication(_policy(), (primary,))
    extra = BMRBStudyEvidence(
        study_id="posthoc-study",
        dataset_id="dataset-c",
        mechanism_id="mechanism-a",
        participant_count=20,
        scientific_criteria_passed=True,
        confirmatory_authority=True,
        promotion_eligible=True,
        reference_effect=0.12,
        reference_ci_lower=0.10,
        reference_ci_upper=0.14,
        source_fingerprint="c" * 64,
    )
    with pytest.raises(ValueError, match="match frozen family exactly"):
        evaluate_study_replication(_policy(), (primary, replication, extra))


def test_dataset_and_source_reuse_cannot_manufacture_replication() -> None:
    with pytest.raises(ValueError, match="dataset_id values must be unique"):
        BMRBStudyReplicationPolicy(
            policy_id="bad",
            mechanism_id="mechanism-a",
            studies=(
                BMRBStudyReplicationSlot("p", "same", "primary", 0, "primary"),
                BMRBStudyReplicationSlot("r", "same", "replication", 1, "replication"),
            ),
            min_successful_replications=1,
            scientific_rationale="Invalid duplicate dataset fixture.",
        )

    shared_source = "d" * 64
    with pytest.raises(ValueError, match="cannot reuse the same source_fingerprint"):
        evaluate_study_replication(
            _policy(),
            (
                BMRBStudyEvidence(
                    study_id="study-primary",
                    dataset_id="dataset-a",
                    mechanism_id="mechanism-a",
                    participant_count=20,
                    scientific_criteria_passed=True,
                    confirmatory_authority=True,
                    promotion_eligible=True,
                    reference_effect=0.12,
                    reference_ci_lower=0.10,
                    reference_ci_upper=0.14,
                    source_fingerprint=shared_source,
                ),
                BMRBStudyEvidence(
                    study_id="study-replication",
                    dataset_id="dataset-b",
                    mechanism_id="mechanism-a",
                    participant_count=20,
                    scientific_criteria_passed=True,
                    confirmatory_authority=True,
                    promotion_eligible=True,
                    reference_effect=0.11,
                    reference_ci_lower=0.09,
                    reference_ci_upper=0.13,
                    source_fingerprint=shared_source,
                ),
            ),
        )


def test_policy_round_trip_and_tampering_change_authority() -> None:
    policy = _policy()
    restored = BMRBStudyReplicationPolicy.from_mapping(policy.to_mapping())
    assert restored.decision_fingerprint == policy.decision_fingerprint
    assert restored.confirmatory_authority is True

    payload = policy.to_mapping()
    payload["min_successful_replications"] = 2
    with pytest.raises(ValueError):
        BMRBStudyReplicationPolicy.from_mapping(payload)

    changed = BMRBStudyReplicationPolicy(
        **{
            **policy.__dict__,
            "scientific_rationale": "Changed after registration.",
        }
    )
    assert changed.decision_fingerprint != policy.decision_fingerprint
    assert changed.confirmatory_authority is False


def test_evidence_fails_closed_on_invalid_values_and_authority_mismatch() -> None:
    with pytest.raises(ValueError, match="boolean"):
        BMRBStudyEvidence(
            study_id="study-primary",
            dataset_id="dataset-a",
            mechanism_id="mechanism-a",
            participant_count=20,
            scientific_criteria_passed=1,
            confirmatory_authority=True,
            promotion_eligible=True,
            reference_effect=0.12,
            reference_ci_lower=0.10,
            reference_ci_upper=0.14,
            source_fingerprint="e" * 64,
        )
    with pytest.raises(ValueError, match="must be finite"):
        _evidence(
            "study-primary", "dataset-a", participants=20, passed=True, source_char="f"
        ).__class__(
            study_id="study-primary",
            dataset_id="dataset-a",
            mechanism_id="mechanism-a",
            participant_count=20,
            scientific_criteria_passed=True,
            confirmatory_authority=True,
            promotion_eligible=True,
            reference_effect=math.nan,
            reference_ci_lower=0.10,
            reference_ci_upper=0.14,
            source_fingerprint="f" * 64,
        )
    with pytest.raises(ValueError, match="must equal scientific PASS"):
        BMRBStudyEvidence(
            study_id="study-primary",
            dataset_id="dataset-a",
            mechanism_id="mechanism-a",
            participant_count=20,
            scientific_criteria_passed=False,
            confirmatory_authority=True,
            promotion_eligible=True,
            reference_effect=0.01,
            reference_ci_lower=-0.01,
            reference_ci_upper=0.03,
            source_fingerprint="0" * 64,
        )


def _confirmatory_policy() -> ConfirmatoryRepresentationPolicy:
    provisional = ConfirmatoryRepresentationPolicy(
        policy_id="confirmatory-adapter-v2",
        reference_representation_id="raw",
        primary_calibration_per_class=10,
        primary_classical_control="normalized_covariance",
        min_participants=3,
        min_representations=2,
        min_representation_families=2,
        min_candidate_advantage=0.05,
        min_ablation_necessity=0.05,
        min_reference_positive_fraction=2 / 3,
        min_all_lane_positive_fraction=2 / 3,
        min_direction_match_fraction=2 / 3,
        min_ablation_direction_match_fraction=2 / 3,
        min_information_novel_representation_fraction=1.0,
        sample_size_rationale="Three synthetic participants qualify the adapter contract only.",
        inference_seed=29,
        bootstrap_resamples=200,
    )
    registration = PreregistrationEvidence(
        registration_uri="https://osf.io/example/register/confirmatory-adapter-v2",
        registered_at="2026-08-30T00:00:00Z",
        registration_document_sha256="1" * 64,
        registered_policy_sha256=provisional.decision_fingerprint,
        registry="OSF",
    )
    return ConfirmatoryRepresentationPolicy(
        **{
            **provisional.__dict__,
            "preregistration": registration,
        }
    )


def _confirmatory_rows() -> list[ConfirmatoryRepresentationObservation]:
    rows: list[ConfirmatoryRepresentationObservation] = []
    for index, participant in enumerate(("p1", "p2", "p3"), start=1):
        for lane, family, digest in (
            ("raw", "raw_neural", "2"),
            ("labram", "foundation_model", "3"),
        ):
            candidate = 0.80
            rows.append(
                ConfirmatoryRepresentationObservation(
                    participant_id=participant,
                    occasion_id="ses-1",
                    case_id=f"{participant}-ses-1",
                    calibration_per_class=10,
                    representation_id=lane,
                    representation_family=family,
                    authority_fingerprint=f"authority-{participant}",
                    representation_sha256=digest * 64,
                    source_fingerprint=str(index) * 64,
                    candidate_metric=candidate,
                    primary_control_metric=candidate - 0.20,
                    ablated_metric=candidate - 0.15,
                    information_novel=True,
                    model_id=None if lane == "raw" else "LABRAM",
                    model_revision=None if lane == "raw" else "rev-1",
                )
            )
    return rows


def test_real_confirmatory_result_adapts_to_one_study_vote() -> None:
    result = evaluate_confirmatory_representation(
        _confirmatory_rows(),
        study_id="study-primary",
        mechanism_id="mechanism-a",
        policy=_confirmatory_policy(),
    )
    evidence = BMRBStudyEvidence.from_confirmatory_result(result, dataset_id="dataset-a")
    assert evidence.study_id == "study-primary"
    assert evidence.dataset_id == "dataset-a"
    assert evidence.participant_count == 3
    assert evidence.scientific_criteria_passed is True
    assert evidence.confirmatory_authority is True
    assert evidence.promotion_eligible is True
    assert evidence.reference_effect == pytest.approx(0.20)
    assert evidence.source_fingerprint == result.source_fingerprint


def test_installed_imbalance_stress_exposes_pseudoreplication_trap() -> None:
    result = run_bmrb_study_replication_stress()
    assert result["official_decision_invariant_to_participant_swap"] is True
    assert result["participant_weighting_reverses_majority_headline"] is True
    assert result["study_positive_fraction_invariant"] is True
    assert result["context_specific_primary_preserved"] is True
    assert result["positive_control_broad_replication_passed"] is True
    assert result["positive_control_broad_claim_promotion_eligible"] is True
