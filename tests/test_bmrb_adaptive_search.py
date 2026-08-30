from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from quantumbci.bmrb_adaptive_search import (
    BMRBAdaptiveSearchPlan,
    run_adaptive_search,
)
from quantumbci.bmrb_adaptive_search_stress import (
    BMRB_ADAPTIVE_SEARCH_STRESS_BENCHMARK,
    default_adaptive_search_plan,
    run_bmrb_adaptive_search_stress,
)
from quantumbci.bmrb_multiplicity import winner_picking_demo_plan
from quantumbci.bmrb_validation import BMRBValidationReplicate


def _replicate(*, effect: float, passed: bool) -> BMRBValidationReplicate:
    return BMRBValidationReplicate(
        scenario_id="manual-adaptive-fixture",
        replicate=0,
        scientific_criteria_passed=passed,
        effect_criteria_passed=passed,
        adversary_survival_passed=True,
        conservation_criteria_passed=True,
        coverage_criteria_passed=True,
        reference_observed_effect=effect,
        reference_effect_ci_lower=effect - 0.01,
        reference_effect_ci_upper=effect + 0.01,
        reference_effect_bias=0.0,
        reference_ci_covers_truth=True,
        expected_failure_localized=not passed,
    )


def _manual_plan() -> BMRBAdaptiveSearchPlan:
    return BMRBAdaptiveSearchPlan(
        plan_id="manual-outcome-routing",
        multiplicity_plan=winner_picking_demo_plan(exploratory_candidates=4),
        max_evaluations=5,
        routing_effect_cutoff=0.05,
        above_cutoff_stride=1,
        below_cutoff_stride=2,
        scientific_rationale=(
            "CI fixture for deterministic outcome-routed candidate inspection."
        ),
    )


def test_outcome_routing_changes_path_but_cannot_transfer_promotion() -> None:
    plan = _manual_plan()
    candidate_ids = plan.multiplicity_plan.candidate_ids
    evidence = {
        candidate_ids[0]: _replicate(effect=0.04, passed=False),
        candidate_ids[1]: _replicate(effect=0.03, passed=False),
        candidate_ids[2]: _replicate(effect=0.06, passed=False),
        candidate_ids[3]: _replicate(effect=0.07, passed=True),
        candidate_ids[4]: _replicate(effect=0.02, passed=False),
    }

    transcript = run_adaptive_search(plan, evidence)
    payload = transcript.to_mapping()

    assert payload["inspected_candidate_ids"] == [
        candidate_ids[0],
        candidate_ids[2],
        candidate_ids[3],
    ]
    assert [step["route"] for step in payload["steps"]] == [
        "below_cutoff",
        "above_cutoff",
        "above_cutoff",
    ]
    assert transcript.naive_adaptive_survivor is True
    assert transcript.exhaustive_any_survivor is True
    assert transcript.authorized_primary_promotion is False
    assert payload["first_adaptive_survivor"] == candidate_ids[3]
    assert payload["physical_quantum_promotion_eligible"] is False


def test_primary_pass_promotes_and_stops_immediately() -> None:
    plan = _manual_plan()
    evidence = {
        candidate_id: _replicate(effect=0.02, passed=False)
        for candidate_id in plan.multiplicity_plan.candidate_ids
    }
    evidence[plan.multiplicity_plan.primary_candidate_id] = _replicate(
        effect=0.08,
        passed=True,
    )

    transcript = run_adaptive_search(plan, evidence)
    assert len(transcript.steps) == 1
    assert transcript.naive_adaptive_survivor is True
    assert transcript.authorized_primary_promotion is True


def test_uninspected_nonprimary_result_cannot_change_primary_authority() -> None:
    multiplicity = winner_picking_demo_plan(exploratory_candidates=2)
    plan = BMRBAdaptiveSearchPlan(
        plan_id="one-inspection-authority-fixture",
        multiplicity_plan=multiplicity,
        max_evaluations=1,
        routing_effect_cutoff=0.05,
        above_cutoff_stride=1,
        below_cutoff_stride=2,
        scientific_rationale=(
            "Prove that uninspected non-primary outcomes cannot alter primary promotion."
        ),
    )
    baseline = {
        candidate_id: _replicate(effect=0.02, passed=False)
        for candidate_id in multiplicity.candidate_ids
    }
    altered = dict(baseline)
    altered[multiplicity.candidate_ids[2]] = _replicate(effect=0.20, passed=True)

    baseline_transcript = run_adaptive_search(plan, baseline)
    altered_transcript = run_adaptive_search(plan, altered)

    assert baseline_transcript.to_mapping()["inspected_candidate_ids"] == [
        multiplicity.primary_candidate_id
    ]
    assert altered_transcript.to_mapping()["inspected_candidate_ids"] == [
        multiplicity.primary_candidate_id
    ]
    assert baseline_transcript.authorized_primary_promotion is False
    assert altered_transcript.authorized_primary_promotion is False
    assert baseline_transcript.naive_adaptive_survivor is False
    assert altered_transcript.naive_adaptive_survivor is False
    assert baseline_transcript.exhaustive_any_survivor is False
    assert altered_transcript.exhaustive_any_survivor is True


def test_adaptive_search_requires_complete_closed_world_evidence() -> None:
    plan = _manual_plan()
    evidence = {
        candidate_id: _replicate(effect=0.02, passed=False)
        for candidate_id in plan.multiplicity_plan.candidate_ids
    }
    evidence.pop(plan.multiplicity_plan.candidate_ids[-1])
    with pytest.raises(ValueError, match="exactly match the frozen candidate family"):
        run_adaptive_search(plan, evidence)

    complete = {
        candidate_id: _replicate(effect=0.02, passed=False)
        for candidate_id in plan.multiplicity_plan.candidate_ids
    }
    complete["post-hoc-candidate"] = _replicate(effect=0.20, passed=True)
    with pytest.raises(ValueError, match="extra=.*post-hoc-candidate"):
        run_adaptive_search(plan, complete)


def test_nonfinite_routing_evidence_fails_closed() -> None:
    plan = _manual_plan()
    evidence = {
        candidate_id: _replicate(effect=0.02, passed=False)
        for candidate_id in plan.multiplicity_plan.candidate_ids
    }
    evidence[plan.multiplicity_plan.primary_candidate_id] = _replicate(
        effect=float("nan"),
        passed=False,
    )
    with pytest.raises(ValueError, match="finite number"):
        run_adaptive_search(plan, evidence)


def test_plan_fingerprint_binds_routing_and_stopping_authority() -> None:
    plan = _manual_plan()
    changed_cutoff = replace(plan, routing_effect_cutoff=0.06)
    changed_stride = replace(plan, below_cutoff_stride=3)
    changed_budget = replace(plan, max_evaluations=4)

    assert changed_cutoff.plan_fingerprint != plan.plan_fingerprint
    assert changed_stride.plan_fingerprint != plan.plan_fingerprint
    assert changed_budget.plan_fingerprint != plan.plan_fingerprint
    assert BMRBAdaptiveSearchPlan.from_mapping(plan.to_mapping()) == plan

    tampered = deepcopy(plan.to_mapping())
    tampered["routing_effect_cutoff"] = 0.06
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        BMRBAdaptiveSearchPlan.from_mapping(tampered)


def test_invalid_adaptive_authority_fails_closed() -> None:
    multiplicity = winner_picking_demo_plan(exploratory_candidates=2)
    with pytest.raises(ValueError, match="distinct"):
        BMRBAdaptiveSearchPlan(
            plan_id="nonadaptive",
            multiplicity_plan=multiplicity,
            max_evaluations=3,
            routing_effect_cutoff=0.05,
            above_cutoff_stride=1,
            below_cutoff_stride=1,
            scientific_rationale="Must actually depend on the observed routing metric.",
        )
    with pytest.raises(ValueError, match="must not exceed"):
        BMRBAdaptiveSearchPlan(
            plan_id="too-large-budget",
            multiplicity_plan=multiplicity,
            max_evaluations=4,
            routing_effect_cutoff=0.05,
            above_cutoff_stride=1,
            below_cutoff_stride=2,
            scientific_rationale="Budget cannot exceed the frozen universe.",
        )
    with pytest.raises(ValueError, match="finite number"):
        BMRBAdaptiveSearchPlan(
            plan_id="nonfinite-cutoff",
            multiplicity_plan=multiplicity,
            max_evaluations=3,
            routing_effect_cutoff=float("nan"),
            above_cutoff_stride=1,
            below_cutoff_stride=2,
            scientific_rationale="Routing authority must reject nonfinite cutoffs.",
        )


def test_known_null_adaptive_stress_reproduces_search_amplification_without_authority_transfer() -> None:
    result = run_bmrb_adaptive_search_stress(
        family_replicates=8,
        candidate_count=20,
        participants=4,
        bootstrap_resamples=100,
        seed=5901,
    )

    assert result["benchmark"] == BMRB_ADAPTIVE_SEARCH_STRESS_BENCHMARK
    assert result["scenario"]["reference_effect"] < result["scenario"][
        "validation_effect_threshold"
    ]
    assert result["adaptive_matches_exhaustive_with_full_budget"] is True
    assert result["adaptive_any_survivor_rate"] == result[
        "exhaustive_any_survivor_rate"
    ]
    assert result["adaptive_any_survivor_rate"] > result[
        "authorized_primary_promotion_rate"
    ]
    assert result["adaptive_winner_picking_amplification"] > 0.0
    assert result["nonprimary_adaptive_survivor_rate"] > 0.0
    assert result["primary_authority_never_transferred"] is True
    assert result["adaptive_plan"] == default_adaptive_search_plan(
        candidate_count=20
    ).to_mapping()


def test_adaptive_claim_boundary_remains_nonbiological() -> None:
    result = run_bmrb_adaptive_search_stress(
        family_replicates=1,
        candidate_count=2,
        participants=4,
        bootstrap_resamples=100,
        seed=5901,
    )
    assert "does not validate biological truth" in result["claim_boundary"]
    assert "physical-quantum" in result["claim_boundary"]
