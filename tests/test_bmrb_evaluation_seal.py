from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

import quantumbci.bmrb_evaluation_seal as seal_module
from quantumbci.bmrb_evaluation_seal import (
    BMRBEvaluationSeal,
    BMRBOperatingAcceptancePlan,
    OperatingAcceptanceCriterion,
    load_bmrb_evaluation_seal,
    verify_bmrb_evaluation_seal_mapping,
    write_bmrb_evaluation_seal,
)
from quantumbci.bmrb_validation_operating import (
    BMRBOperatingStudyPolicy,
    OperatingCurveGrid,
)
from quantumbci.preregistration import (
    PreregistrationEvidence,
    canonical_scientific_fingerprint,
)


def _evaluation_policy() -> BMRBOperatingStudyPolicy:
    return BMRBOperatingStudyPolicy(
        study_id="ci-final-operating-evaluation",
        source_sha="473c1337d5a706087352e7a8af8e4c5779735849",
        partition="evaluation",
        grid=OperatingCurveGrid(
            scenario_ids=(
                "equivalence-null",
                "predictive-shortcut",
                "shared-mechanism-positive",
            ),
            participant_counts=(8,),
            effect_scales=(1.0,),
            heterogeneity_scales=(1.0,),
            measurement_noise_scales=(1.0,),
        ),
        replicates_per_cell=20,
        bootstrap_resamples=100,
        primary_calibration_per_class=10,
    )


def _criteria() -> tuple[OperatingAcceptanceCriterion, ...]:
    return (
        OperatingAcceptanceCriterion(
            criterion_id="known-positive-recovery",
            metric="aggregate.known_positive_recovery_rate",
            lower_bound=0.80,
            rationale="CI-only example bound; final scientific value requires external justification.",
        ),
        OperatingAcceptanceCriterion(
            criterion_id="false-promotion",
            metric="aggregate.false_promotion_rate",
            upper_bound=0.10,
            rationale="CI-only example bound; final scientific value requires external justification.",
        ),
        OperatingAcceptanceCriterion(
            criterion_id="equivalence-null-pass",
            metric="scenario.observed_pass_rate",
            scenario_id="equivalence-null",
            reducer="maximum",
            upper_bound=0.10,
            rationale="CI-only example; the equivalence null must be explicitly bounded.",
        ),
        OperatingAcceptanceCriterion(
            criterion_id="shortcut-null-pass",
            metric="scenario.observed_pass_rate",
            scenario_id="predictive-shortcut",
            reducer="maximum",
            upper_bound=0.10,
            rationale="CI-only example; predictive shortcuts must be explicitly bounded.",
        ),
        OperatingAcceptanceCriterion(
            criterion_id="shared-positive-pass",
            metric="scenario.observed_pass_rate",
            scenario_id="shared-mechanism-positive",
            reducer="minimum",
            lower_bound=0.80,
            rationale="CI-only example; known-positive recovery must be explicitly bounded.",
        ),
    )


def _plan() -> BMRBOperatingAcceptancePlan:
    return BMRBOperatingAcceptancePlan(
        study_id="ci-bmrb-evaluation-seal",
        development_evidence_ref="artifact://development/operating.json",
        development_artifact_fingerprint="a" * 64,
        development_policy_fingerprint="b" * 64,
        evaluation_policy=_evaluation_policy(),
        criteria=_criteria(),
        multiplicity_policy=(
            "CI-only example policy: every declared criterion must be reported; no endpoint may "
            "be silently dropped after evaluation."
        ),
        scientific_rationale=(
            "CI-only example plan used to qualify sealing mechanics, not final BMRB thresholds."
        ),
    )


def _seal() -> BMRBEvaluationSeal:
    plan = _plan()
    registration = PreregistrationEvidence(
        registration_uri="https://registry.example.invalid/bmrb-ci-seal",
        registered_at="2026-08-28T21:30:00Z",
        registration_document_sha256="c" * 64,
        registered_policy_sha256=plan.plan_fingerprint,
        registry="CI-only example registry",
    )
    return BMRBEvaluationSeal(plan=plan, preregistration=registration)


def _refingerprint_outer(payload: dict) -> None:
    core = {key: value for key, value in payload.items() if key != "artifact_fingerprint"}
    payload["artifact_fingerprint"] = canonical_scientific_fingerprint(
        "quantumbci.bmrb-operating-evaluation-seal.v1",
        core,
    )


def test_acceptance_criterion_requires_explicit_bound_and_rationale() -> None:
    with pytest.raises(ValueError, match="at least one bound"):
        OperatingAcceptanceCriterion(
            criterion_id="missing-bound",
            metric="aggregate.false_promotion_rate",
            rationale="A bound is scientifically required before evaluation.",
        )
    with pytest.raises(ValueError, match="rationale"):
        OperatingAcceptanceCriterion(
            criterion_id="missing-rationale",
            metric="aggregate.false_promotion_rate",
            rationale="",
            upper_bound=0.1,
        )


def test_acceptance_plan_requires_evaluation_partition() -> None:
    development_policy = replace(_evaluation_policy(), partition="development")
    with pytest.raises(ValueError, match="partition='evaluation'"):
        BMRBOperatingAcceptancePlan(
            study_id="bad-partition",
            development_evidence_ref="artifact://development/operating.json",
            development_artifact_fingerprint="a" * 64,
            development_policy_fingerprint="b" * 64,
            evaluation_policy=development_policy,
            criteria=_criteria(),
            multiplicity_policy="Report every criterion.",
            scientific_rationale="Reject a development-partition plan.",
        )


def test_acceptance_plan_requires_core_null_and_positive_endpoints() -> None:
    criteria = tuple(
        criterion
        for criterion in _criteria()
        if criterion.scenario_id != "predictive-shortcut"
    )
    with pytest.raises(ValueError, match="predictive-shortcut"):
        BMRBOperatingAcceptancePlan(
            study_id="missing-shortcut",
            development_evidence_ref="artifact://development/operating.json",
            development_artifact_fingerprint="a" * 64,
            development_policy_fingerprint="b" * 64,
            evaluation_policy=_evaluation_policy(),
            criteria=criteria,
            multiplicity_policy="Report every criterion.",
            scientific_rationale="Core known-truth endpoints are mandatory.",
        )


def test_acceptance_plan_fingerprint_binds_thresholds_and_evaluation_policy() -> None:
    plan = _plan()
    changed_criteria = tuple(
        replace(criterion, upper_bound=0.20)
        if criterion.criterion_id == "false-promotion"
        else criterion
        for criterion in plan.criteria
    )
    changed_threshold = replace(plan, criteria=changed_criteria)
    changed_policy = replace(
        plan,
        evaluation_policy=replace(plan.evaluation_policy, replicates_per_cell=21),
    )

    assert plan.plan_fingerprint != changed_threshold.plan_fingerprint
    assert plan.plan_fingerprint != changed_policy.plan_fingerprint
    assert plan.to_mapping()["evaluation_executed"] is False


def test_external_preregistration_must_bind_exact_acceptance_plan() -> None:
    plan = _plan()
    wrong_registration = PreregistrationEvidence(
        registration_uri="https://registry.example.invalid/wrong-plan",
        registered_at="2026-08-28T21:30:00Z",
        registration_document_sha256="c" * 64,
        registered_policy_sha256="d" * 64,
    )

    with pytest.raises(ValueError, match="does not bind the exact"):
        BMRBEvaluationSeal(plan=plan, preregistration=wrong_registration)


def test_evaluation_seal_round_trip_is_canonical_and_verified(tmp_path) -> None:
    seal = _seal()
    output = write_bmrb_evaluation_seal(seal, tmp_path / "evaluation-seal.json")
    loaded = load_bmrb_evaluation_seal(output)

    assert loaded == seal.to_mapping()
    assert loaded["plan"]["plan_fingerprint"] == seal.plan.plan_fingerprint
    assert loaded["preregistration"]["registered_policy_sha256"] == seal.plan.plan_fingerprint
    assert loaded["evaluation_executed"] is False
    verify_bmrb_evaluation_seal_mapping(loaded)


def test_evaluation_seal_rejects_stale_and_nested_tampering() -> None:
    stale = deepcopy(_seal().to_mapping())
    stale["plan"]["criteria"][0]["lower_bound"] = 0.70
    with pytest.raises(ValueError, match="artifact fingerprint mismatch"):
        verify_bmrb_evaluation_seal_mapping(stale)

    nested = deepcopy(_seal().to_mapping())
    nested["plan"]["criteria"][0]["lower_bound"] = 0.70
    _refingerprint_outer(nested)
    with pytest.raises(ValueError, match="plan fingerprint mismatch"):
        verify_bmrb_evaluation_seal_mapping(nested)


def test_evaluation_seal_cannot_claim_evaluation_already_ran() -> None:
    payload = deepcopy(_seal().to_mapping())
    payload["evaluation_executed"] = True
    _refingerprint_outer(payload)

    with pytest.raises(ValueError, match="before evaluation execution"):
        verify_bmrb_evaluation_seal_mapping(payload)


def test_seal_surface_contains_no_final_evaluation_runner() -> None:
    source = Path(seal_module.__file__).read_text(encoding="utf-8")

    assert "run_bmrb_operating_characteristics" not in source
    assert "does not execute the final evaluation partition" in source
    assert "Numeric acceptance thresholds are never supplied" in source
