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


def _two_family_plan() -> BMRBMultiplicityPlan:
    return BMRBMultiplicityPlan(
        plan_id="two-family-fixture",
        family_order=("mechanism", "representation"),
        candidates=(
            BMRBMultiplicityCandidate(
                candidate_id="mechanism-primary",
                family_id="mechanism",
                role="primary",
                order=0,
                rationale="Primary mechanism fixed before final evidence.",
            ),
            BMRBMultiplicityCandidate(
                candidate_id="mechanism-secondary",
                family_id="mechanism",
                role="secondary",
                order=1,
                rationale="Secondary confirmatory mechanism is reported without v1 promotion authority.",
            ),
            BMRBMultiplicityCandidate(
                candidate_id="mechanism-exploratory",
                family_id="mechanism",
                role="exploratory",
                order=2,
                rationale="Exploratory mechanism remains visible but cannot be promoted post hoc.",
            ),
            BMRBMultiplicityCandidate(
                candidate_id="representation-primary",
                family_id="representation",
                role="primary",
                order=0,
                rationale="Primary representation hypothesis fixed before final evidence.",
            ),
            BMRBMultiplicityCandidate(
                candidate_id="representation-exploratory",
                family_id="representation",
                role="exploratory",
                order=1,
                rationale="Exploratory representation hypothesis is report-only in v1.",
            ),
        ),
        scientific_rationale="CI fixture for grouped and ordered candidate-family authority.",
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
    assert decision.to_mapping()["physical_quantum_promotion_eligible"] is False


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
    plan = _two_family_plan()
    results = {candidate_id: False for candidate_id in plan.candidate_ids}
    results["mechanism-secondary"] = True
    results["mechanism-exploratory"] = True
    results["representation-exploratory"] = True

    decision = apply_multiplicity_plan(plan, results)
    assert decision.naive_any_survivor is True
    assert decision.authorized_any_promotion is False
    assert decision.suppressed_nonprimary_survivors == (
        "mechanism-secondary",
        "mechanism-exploratory",
        "representation-exploratory",
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


def test_each_family_requires_exactly_one_primary_at_order_zero() -> None:
    plan = _two_family_plan()
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


def test_candidate_order_and_family_order_are_closed_world() -> None:
    plan = _two_family_plan()
    candidates = list(plan.candidates)
    candidates[1] = replace(candidates[1], order=4)
    with pytest.raises(ValueError, match="contiguous from zero"):
        BMRBMultiplicityPlan(
            plan_id="order-gap",
            family_order=plan.family_order,
            candidates=tuple(candidates),
            scientific_rationale=plan.scientific_rationale,
        )

    with pytest.raises(ValueError, match="exactly cover"):
        BMRBMultiplicityPlan(
            plan_id="unregistered-family",
            family_order=("mechanism",),
            candidates=plan.candidates,
            scientific_rationale=plan.scientific_rationale,
        )


def test_plan_fingerprint_binds_roles_order_and_complete_candidate_set() -> None:
    plan = _two_family_plan()
    changed_role = BMRBMultiplicityPlan(
        plan_id=plan.plan_id,
        family_order=plan.family_order,
        candidates=tuple(
            replace(candidate, role="exploratory")
            if candidate.candidate_id == "mechanism-secondary"
            else candidate
            for candidate in plan.candidates
        ),
        scientific_rationale=plan.scientific_rationale,
    )
    changed_order = BMRBMultiplicityPlan(
        plan_id=plan.plan_id,
        family_order=tuple(reversed(plan.family_order)),
        candidates=plan.candidates,
        scientific_rationale=plan.scientific_rationale,
    )
    reduced = BMRBMultiplicityPlan(
        plan_id=plan.plan_id,
        family_order=plan.family_order,
        candidates=tuple(
            candidate
            for candidate in plan.candidates
            if candidate.candidate_id != "mechanism-exploratory"
        ),
        scientific_rationale=plan.scientific_rationale,
    )

    assert changed_role.plan_fingerprint != plan.plan_fingerprint
    assert changed_order.plan_fingerprint != plan.plan_fingerprint
    assert reduced.plan_fingerprint != plan.plan_fingerprint


def test_plan_round_trip_rejects_nested_tampering() -> None:
    plan = _two_family_plan()
    payload = plan.to_mapping()
    assert BMRBMultiplicityPlan.from_mapping(payload) == plan

    tampered = deepcopy(payload)
    tampered["candidates"][1]["role"] = "exploratory"
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        BMRBMultiplicityPlan.from_mapping(tampered)


def test_fixture_requires_an_actual_search_family() -> None:
    with pytest.raises(ValueError, match="positive"):
        winner_picking_demo_plan(exploratory_candidates=0)
