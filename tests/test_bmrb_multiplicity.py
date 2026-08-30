from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from quantumbci.bmrb_multiplicity import (
    BMRBMultiplicityCandidate,
    BMRBMultiplicityPlan,
    apply_multiplicity_plan,
    winner_picking_demo_plan,
)


def _single_family_plan() -> BMRBMultiplicityPlan:
    return BMRBMultiplicityPlan(
        plan_id="single-family-fixture",
        family_order=("candidate-search",),
        candidates=(
            BMRBMultiplicityCandidate(
                candidate_id="candidate-primary",
                family_id="candidate-search",
                role="primary",
                order=0,
                rationale="Primary candidate fixed before final evidence.",
            ),
            BMRBMultiplicityCandidate(
                candidate_id="candidate-secondary",
                family_id="candidate-search",
                role="secondary",
                order=1,
                rationale="Secondary candidate is reported without v1 promotion authority.",
            ),
            BMRBMultiplicityCandidate(
                candidate_id="candidate-exploratory-a",
                family_id="candidate-search",
                role="exploratory",
                order=2,
                rationale="Exploratory candidate remains visible but cannot be promoted post hoc.",
            ),
            BMRBMultiplicityCandidate(
                candidate_id="candidate-exploratory-b",
                family_id="candidate-search",
                role="exploratory",
                order=3,
                rationale="Second exploratory candidate remains report-only in v1.",
            ),
            BMRBMultiplicityCandidate(
                candidate_id="candidate-exploratory-c",
                family_id="candidate-search",
                role="exploratory",
                order=4,
                rationale="Third exploratory candidate supports candidate-set fingerprint tests.",
            ),
        ),
        scientific_rationale="CI fixture for one closed and ordered candidate family.",
    )


def test_winner_picking_cannot_transfer_promotion_authority() -> None:
    plan = winner_picking_demo_plan(exploratory_candidates=19)
    results = {candidate_id: False for candidate_id in plan.candidate_ids}
    results["mechanism-exploratory-07"] = True
    results["mechanism-exploratory-14"] = True

    decision = apply_multiplicity_plan(plan, results)

    assert decision.naive_any_survivor is True
    assert decision.authorized_any_promotion is False
    assert decision.suppressed_nonprimary_survivors == (
        "mechanism-exploratory-07",
        "mechanism-exploratory-14",
    )
    payload = decision.to_mapping()
    assert payload["primary_candidate_id"] == "mechanism-primary"
    assert payload["physical_quantum_promotion_eligible"] is False


def test_predeclared_primary_can_promote_only_after_scientific_pass() -> None:
    plan = winner_picking_demo_plan(exploratory_candidates=3)
    failed = {candidate_id: False for candidate_id in plan.candidate_ids}
    passed = dict(failed)
    passed["mechanism-primary"] = True

    assert apply_multiplicity_plan(plan, failed).authorized_any_promotion is False
    promoted = apply_multiplicity_plan(plan, passed)
    assert promoted.authorized_any_promotion is True
    primary = next(item for item in promoted.candidates if item.candidate_id == "mechanism-primary")
    assert primary.promotion_authority is True
    assert primary.promotion_eligible is True


def test_secondary_and_exploratory_survivors_remain_reportable_not_promotable() -> None:
    plan = _single_family_plan()
    results = {candidate_id: False for candidate_id in plan.candidate_ids}
    results["candidate-secondary"] = True
    results["candidate-exploratory-a"] = True
    results["candidate-exploratory-c"] = True

    decision = apply_multiplicity_plan(plan, results)
    assert decision.naive_any_survivor is True
    assert decision.authorized_any_promotion is False
    assert decision.suppressed_nonprimary_survivors == (
        "candidate-secondary",
        "candidate-exploratory-a",
        "candidate-exploratory-c",
    )


def test_results_must_exactly_match_frozen_candidate_family() -> None:
    plan = winner_picking_demo_plan(exploratory_candidates=2)
    complete = {candidate_id: False for candidate_id in plan.candidate_ids}

    missing = dict(complete)
    missing.pop("mechanism-exploratory-02")
    with pytest.raises(ValueError, match="missing=.*mechanism-exploratory-02"):
        apply_multiplicity_plan(plan, missing)

    extra = dict(complete)
    extra["post-hoc-winner"] = True
    with pytest.raises(ValueError, match="extra=.*post-hoc-winner"):
        apply_multiplicity_plan(plan, extra)

    non_boolean = dict(complete)
    non_boolean["mechanism-primary"] = 1
    with pytest.raises(ValueError, match="boolean"):
        apply_multiplicity_plan(plan, non_boolean)

    with pytest.raises(ValueError, match="candidate-id strings"):
        apply_multiplicity_plan(plan, {**complete, 7: False})


def test_v1_requires_exactly_one_primary_at_order_zero() -> None:
    plan = _single_family_plan()
    candidates = list(plan.candidates)
    candidates[0] = replace(candidates[0], role="secondary")
    with pytest.raises(ValueError, match="exactly one predeclared primary"):
        BMRBMultiplicityPlan(
            plan_id="missing-primary",
            family_order=plan.family_order,
            candidates=tuple(candidates),
            scientific_rationale=plan.scientific_rationale,
        )

    candidates = list(plan.candidates)
    candidates[0] = replace(candidates[0], order=1)
    candidates[1] = replace(candidates[1], order=0)
    with pytest.raises(ValueError, match="primary candidate must occupy order zero"):
        BMRBMultiplicityPlan(
            plan_id="primary-not-first",
            family_order=plan.family_order,
            candidates=tuple(candidates),
            scientific_rationale=plan.scientific_rationale,
        )


def test_v1_rejects_multi_family_laundering_of_search_opportunities() -> None:
    plan = _single_family_plan()
    second_family_primary = BMRBMultiplicityCandidate(
        candidate_id="laundered-primary",
        family_id="second-family",
        role="primary",
        order=0,
        rationale="Would create another promotion-authoritative search opportunity if allowed.",
    )
    with pytest.raises(ValueError, match="exactly one closed candidate family"):
        BMRBMultiplicityPlan(
            plan_id="multi-family-laundering",
            family_order=("candidate-search", "second-family"),
            candidates=(*plan.candidates, second_family_primary),
            scientific_rationale="This must fail closed in v1.",
        )


def test_candidate_order_and_family_membership_are_closed_world() -> None:
    plan = _single_family_plan()
    candidates = list(plan.candidates)
    candidates[1] = replace(candidates[1], order=7)
    with pytest.raises(ValueError, match="contiguous from zero"):
        BMRBMultiplicityPlan(
            plan_id="order-gap",
            family_order=plan.family_order,
            candidates=tuple(candidates),
            scientific_rationale=plan.scientific_rationale,
        )

    candidates = list(plan.candidates)
    candidates[-1] = replace(candidates[-1], family_id="undeclared-family")
    with pytest.raises(ValueError, match="single frozen family"):
        BMRBMultiplicityPlan(
            plan_id="unregistered-family",
            family_order=plan.family_order,
            candidates=tuple(candidates),
            scientific_rationale=plan.scientific_rationale,
        )


def test_plan_fingerprint_binds_roles_order_and_complete_candidate_set() -> None:
    plan = _single_family_plan()
    changed_role = BMRBMultiplicityPlan(
        plan_id=plan.plan_id,
        family_order=plan.family_order,
        candidates=tuple(
            replace(candidate, role="exploratory")
            if candidate.candidate_id == "candidate-secondary"
            else candidate
            for candidate in plan.candidates
        ),
        scientific_rationale=plan.scientific_rationale,
    )
    changed_order = BMRBMultiplicityPlan(
        plan_id=plan.plan_id,
        family_order=plan.family_order,
        candidates=tuple(
            replace(candidate, order=2)
            if candidate.candidate_id == "candidate-secondary"
            else replace(candidate, order=1)
            if candidate.candidate_id == "candidate-exploratory-a"
            else candidate
            for candidate in plan.candidates
        ),
        scientific_rationale=plan.scientific_rationale,
    )
    reduced = BMRBMultiplicityPlan(
        plan_id=plan.plan_id,
        family_order=plan.family_order,
        candidates=tuple(
            candidate
            for candidate in plan.candidates
            if candidate.candidate_id != "candidate-exploratory-c"
        ),
        scientific_rationale=plan.scientific_rationale,
    )

    assert changed_role.plan_fingerprint != plan.plan_fingerprint
    assert changed_order.plan_fingerprint != plan.plan_fingerprint
    assert reduced.plan_fingerprint != plan.plan_fingerprint


def test_plan_round_trip_rejects_nested_tampering() -> None:
    plan = _single_family_plan()
    payload = plan.to_mapping()
    assert BMRBMultiplicityPlan.from_mapping(payload) == plan

    tampered = deepcopy(payload)
    tampered["candidates"][1]["role"] = "exploratory"
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        BMRBMultiplicityPlan.from_mapping(tampered)


def test_candidate_order_rejects_boolean_values() -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        BMRBMultiplicityCandidate(
            candidate_id="bad-order",
            family_id="candidate-search",
            role="primary",
            order=True,
            rationale="Boolean order values must not pass integer validation.",
        )


def test_fixture_requires_an_actual_search_family() -> None:
    with pytest.raises(ValueError, match="positive"):
        winner_picking_demo_plan(exploratory_candidates=0)
    with pytest.raises(ValueError, match="positive"):
        winner_picking_demo_plan(exploratory_candidates=True)
