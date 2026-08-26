from __future__ import annotations

import pytest

from quantumbci.causal_recapitulation import (
    CausalCaseEvidence,
    CausalNecessityPolicy,
    MatchedClassicalRecovery,
    evaluate_causal_necessity,
)


def _case(participant: str, information_set: str) -> CausalCaseEvidence:
    return CausalCaseEvidence(
        participant_id=participant,
        occasion_id="s1",
        case_id=f"{participant}-s1",
        mechanism_id="lindblad_latent_dynamics",
        intervention_id="erase_candidate_information",
        dose_response_passed=True,
        oriented_endpoint_effect=0.5,
        mean_monotonic_fraction=1.0,
        faithfulness_passed=True,
        sufficiency_fraction=0.9,
        necessity_fraction=0.8,
        joint_random_percentile=0.99,
        matched_recovery=MatchedClassicalRecovery(
            classical_model_id="matched-control",
            classical_recovery_fraction=0.1,
            information_set_id=information_set,
            source_fingerprint=f"recovery-{participant}",
        ),
        dose_source_fingerprint=f"dose-{participant}",
        faithfulness_source_fingerprint=f"faith-{participant}",
        source_schemas=(
            "neuros-mechint.dose-response-study.v1",
            "neuros-mechint.evidence-pack.v1",
        ),
    )


def test_policy_parser_does_not_coerce_string_false_to_true() -> None:
    with pytest.raises(TypeError, match="JSON boolean"):
        CausalNecessityPolicy.from_mapping(
            {"policy_id": "strict-bool", "preregistered": "false"}
        )


def test_policy_constructor_requires_real_boolean() -> None:
    with pytest.raises(TypeError, match="JSON boolean"):
        CausalNecessityPolicy(policy_id="strict-bool", preregistered=1)  # type: ignore[arg-type]


def test_python_evaluator_rejects_mixed_information_set_authority() -> None:
    policy = CausalNecessityPolicy(policy_id="strict-info", preregistered=True)
    with pytest.raises(ValueError, match="information_set_id"):
        evaluate_causal_necessity(
            [
                _case("p1", "budget-a"),
                _case("p2", "budget-a"),
                _case("p3", "budget-b"),
            ],
            policy=policy,
        )


def test_case_booleans_are_not_truthiness_coerced() -> None:
    with pytest.raises(TypeError, match="dose_response_passed"):
        CausalCaseEvidence(
            participant_id="p1",
            occasion_id="s1",
            case_id="p1-s1",
            mechanism_id="lindblad_latent_dynamics",
            intervention_id="erase_candidate_information",
            dose_response_passed=1,  # type: ignore[arg-type]
            oriented_endpoint_effect=0.5,
            mean_monotonic_fraction=1.0,
            faithfulness_passed=True,
            sufficiency_fraction=0.9,
            necessity_fraction=0.8,
            joint_random_percentile=0.99,
            matched_recovery=MatchedClassicalRecovery(
                classical_model_id="control",
                classical_recovery_fraction=0.1,
                information_set_id="budget-a",
                source_fingerprint="recovery-p1",
            ),
            dose_source_fingerprint="dose-p1",
            faithfulness_source_fingerprint="faith-p1",
            source_schemas=(
                "neuros-mechint.dose-response-study.v1",
                "neuros-mechint.evidence-pack.v1",
            ),
        )
