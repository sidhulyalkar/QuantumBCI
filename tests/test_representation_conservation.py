from __future__ import annotations

import pytest

from quantumbci.recapitulation import EvidenceTier, GateStatus, gate_map
from quantumbci.representation_conservation import (
    RepresentationConservationPolicy,
    RepresentationEffectCase,
    build_representation_necessity_profile,
    evaluate_representation_conservation,
)


def _case(
    lane: str,
    family: str,
    participant: str,
    *,
    occasion: str = "ses-1",
    case_suffix: str = "a",
    budget: int = 0,
    advantage: float = 0.20,
    ablation: float = 0.30,
    novel: bool = True,
    authority: str | None = None,
    model_id: str | None = None,
) -> RepresentationEffectCase:
    candidate = 0.80
    return RepresentationEffectCase(
        participant_id=participant,
        occasion_id=occasion,
        case_id=f"{participant}-{case_suffix}",
        calibration_per_class=budget,
        representation_id=lane,
        representation_family=family,
        source_representation_id=f"source-{lane}",
        model_id=model_id,
        model_revision=None if model_id is None else "rev-1",
        mechanism_id="cross_feature_second_moment",
        authority_fingerprint=authority or f"authority-{participant}-{case_suffix}-{budget}",
        representation_sha256=(lane[0] * 64),
        source_fingerprint=(participant[-1] * 64),
        candidate_metric=candidate,
        strongest_control_metric=candidate - advantage,
        ablated_metric=candidate - ablation,
        higher_is_better=True,
        information_novel=novel,
    )


def _policy(*, preregistered: bool = True) -> RepresentationConservationPolicy:
    return RepresentationConservationPolicy(
        policy_id="rep-policy-v1",
        preregistered=preregistered,
        reference_representation_id="raw",
        min_participants=3,
        min_representations=3,
        min_representation_families=2,
        min_reference_positive_fraction=0.80,
        min_all_lane_positive_fraction=0.80,
        min_all_lane_ablation_positive_fraction=0.80,
        min_direction_match_fraction=0.80,
        min_ablation_direction_match_fraction=0.80,
        min_information_novel_representation_fraction=1.0,
    )


def _three_lane_cases(*, novel: bool = True) -> list[RepresentationEffectCase]:
    rows: list[RepresentationEffectCase] = []
    for participant, delta in (("p1", 0.18), ("p2", 0.22), ("p3", 0.26)):
        authority = f"authority-{participant}"
        rows.extend(
            [
                _case("raw", "raw_neural", participant, advantage=delta, ablation=0.28, novel=novel, authority=authority),
                _case("labram", "foundation_model", participant, advantage=delta * 0.9, ablation=0.24, novel=novel, authority=authority, model_id="LaBraM"),
                _case("eegpt", "foundation_model", participant, advantage=delta * 1.1, ablation=0.31, novel=novel, authority=authority, model_id="EEGPT"),
            ]
        )
    return rows


def test_preregistered_cross_representation_conservation_can_reach_repeated_case() -> None:
    result = evaluate_representation_conservation(_three_lane_cases(), policy=_policy())
    assert result.conservation_criteria_passed is True
    assert result.adversary_survival_passed is True
    assert result.promotion_eligible is True
    assert result.participant_count == 3
    assert result.representation_count == 3
    assert result.representation_family_count == 2
    assert result.direction_match_fraction == 1.0
    assert result.ablation_direction_match_fraction == 1.0
    assert result.information_novel_representation_fraction == 1.0
    assert result.pairwise_reference_correlations["labram"] == pytest.approx(1.0)
    assert result.pairwise_reference_correlations["eegpt"] == pytest.approx(1.0)

    profile = build_representation_necessity_profile(result)
    gates = gate_map(profile)
    assert gates["paired_representation_authority"].status == GateStatus.PASS
    assert gates["held_out_representation_effect"].status == GateStatus.PASS
    assert gates["matched_representation_adversaries"].status == GateStatus.PASS
    assert gates["cross_representation_stability"].status == GateStatus.PASS
    assert gates["cross_representation_replication"].status == GateStatus.PASS
    assert gates["causal_intervention_and_ablation"].status == GateStatus.NOT_RUN
    assert profile.promotion_ceiling == EvidenceTier.REPEATED_CASE
    assert profile.to_mapping()["necessity_claim_permitted"] is False


def test_equivalent_classical_lanes_preserve_conservation_but_fail_adversary() -> None:
    result = evaluate_representation_conservation(_three_lane_cases(novel=False), policy=_policy())
    assert result.conservation_criteria_passed is True
    assert result.adversary_survival_passed is False
    assert result.promotion_eligible is False

    profile = build_representation_necessity_profile(result)
    gates = gate_map(profile)
    assert gates["held_out_representation_effect"].status == GateStatus.PASS
    assert gates["matched_representation_adversaries"].status == GateStatus.FAIL
    assert gates["cross_representation_stability"].status == GateStatus.CHARACTERIZED
    assert gates["cross_representation_replication"].status == GateStatus.CHARACTERIZED
    assert profile.first_failing_gate == "matched_representation_adversaries"
    assert profile.promotion_ceiling == EvidenceTier.PREDICTIVE


def test_retrospective_policy_characterizes_without_promotion() -> None:
    result = evaluate_representation_conservation(
        _three_lane_cases(), policy=_policy(preregistered=False)
    )
    assert result.conservation_criteria_passed is True
    assert result.adversary_survival_passed is True
    assert result.promotion_eligible is False

    profile = build_representation_necessity_profile(result)
    gates = gate_map(profile)
    assert gates["held_out_representation_effect"].status == GateStatus.CHARACTERIZED
    assert gates["matched_representation_adversaries"].status == GateStatus.CHARACTERIZED
    assert gates["cross_representation_stability"].status == GateStatus.CHARACTERIZED
    assert gates["cross_representation_replication"].status == GateStatus.CHARACTERIZED
    assert profile.promotion_ceiling == EvidenceTier.DESCRIPTIVE


def test_authority_mismatch_across_lanes_fails_closed() -> None:
    cases = _three_lane_cases()
    broken = list(cases)
    original = broken[1]
    broken[1] = RepresentationEffectCase(
        **{
            **original.__dict__,
            "authority_fingerprint": "different-authority",
        }
    )
    with pytest.raises(ValueError, match="authority fingerprint mismatch"):
        evaluate_representation_conservation(broken, policy=_policy())


def test_missing_exact_pair_fails_closed_instead_of_shrinking_intersection() -> None:
    cases = _three_lane_cases()
    incomplete = [case for case in cases if not (case.representation_id == "eegpt" and case.participant_id == "p3")]
    with pytest.raises(ValueError, match="exactly paired"):
        evaluate_representation_conservation(incomplete, policy=_policy())


def test_participant_balancing_prevents_extra_cases_from_overweighting_one_person() -> None:
    cases: list[RepresentationEffectCase] = []
    for lane, family, model in (
        ("raw", "raw_neural", None),
        ("labram", "foundation_model", "LaBraM"),
        ("eegpt", "foundation_model", "EEGPT"),
    ):
        for suffix in ("a", "b", "c"):
            cases.append(
                _case(
                    lane,
                    family,
                    "p1",
                    case_suffix=suffix,
                    advantage=0.0,
                    ablation=0.2,
                    novel=True,
                    authority=f"authority-p1-{suffix}",
                    model_id=model,
                )
            )
        for participant in ("p2", "p3"):
            cases.append(
                _case(
                    lane,
                    family,
                    participant,
                    advantage=1.0,
                    ablation=0.2,
                    novel=True,
                    authority=f"authority-{participant}-a",
                    model_id=model,
                )
            )
    result = evaluate_representation_conservation(cases, policy=_policy())
    raw = next(lane for lane in result.lanes if lane.representation_id == "raw")
    assert raw.mean_candidate_advantage == pytest.approx(2.0 / 3.0)
    assert raw.mean_candidate_advantage != pytest.approx(0.4)


def test_policy_rejects_truthy_string_preregistration() -> None:
    with pytest.raises(ValueError, match="boolean"):
        RepresentationConservationPolicy.from_mapping(
            {
                "policy_id": "bad",
                "preregistered": "false",
                "reference_representation_id": "raw",
            }
        )
