from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from quantumbci.bmrb_adaptive_search import BMRBAdaptiveSearchPlan
from quantumbci.bmrb_multiplicity import winner_picking_demo_plan
from quantumbci.bmrb_study_evaluation_seal import (
    BMRBStudyEvaluationSeal,
    BMRBStudyHierarchyAuthority,
    BMRBStudyOperatingAcceptancePlan,
    BMRBStudySearchAuthority,
    StudyOperatingAcceptanceCriterion,
    verify_bmrb_study_evaluation_seal_mapping,
)
from quantumbci.bmrb_study_operating import (
    BMRBStudyOperatingPolicy,
    StudySimulationSeedPartition,
    qualification_smoke_grid,
    run_bmrb_study_operating_characteristics,
)
from quantumbci.bmrb_study_operating_artifacts import (
    verify_bmrb_study_operating_mapping,
)
from quantumbci.preregistration import PreregistrationEvidence, canonical_scientific_fingerprint

SOURCE_SHA = "307edfd42c4138090807b2c5ab1e8223d99a7a9e"


def _policy(partition: str = "development", *, seed: StudySimulationSeedPartition | None = None) -> BMRBStudyOperatingPolicy:
    return BMRBStudyOperatingPolicy(
        study_id="study-operating-seal-test-v1",
        source_sha=SOURCE_SHA,
        partition=partition,  # type: ignore[arg-type]
        grid=qualification_smoke_grid(),
        replicates_per_cell=1,
        bootstrap_resamples=100,
        seed_partition=seed or StudySimulationSeedPartition(),
    )


@pytest.fixture(scope="module")
def development_artifact() -> dict[str, object]:
    return run_bmrb_study_operating_characteristics(_policy()).to_mapping()


def _criteria() -> tuple[StudyOperatingAcceptanceCriterion, ...]:
    # Values are deterministic software-test fixtures, not scientific defaults.
    return (
        StudyOperatingAcceptanceCriterion(
            criterion_id="false-promotion",
            metric="aggregate.mean_false_promotion_rate",
            upper_bound=0.20,
            rationale="Synthetic fixture bound on broad false promotion.",
        ),
        StudyOperatingAcceptanceCriterion(
            criterion_id="positive-recovery",
            metric="aggregate.mean_known_positive_recovery_rate",
            lower_bound=0.80,
            rationale="Synthetic fixture bound on broad positive recovery.",
        ),
        StudyOperatingAcceptanceCriterion(
            criterion_id="positive-four",
            metric="scenario.observed_replication_pass_rate",
            scenario_id="homogeneous-positive-4",
            reducer="minimum",
            lower_bound=0.80,
            rationale="Synthetic positive-control fixture.",
        ),
        StudyOperatingAcceptanceCriterion(
            criterion_id="null-four",
            metric="scenario.observed_replication_pass_rate",
            scenario_id="homogeneous-null-4",
            reducer="maximum",
            upper_bound=0.20,
            rationale="Synthetic null-control fixture.",
        ),
        StudyOperatingAcceptanceCriterion(
            criterion_id="primary-protection",
            metric="scenario.primary_role_protection_rate",
            scenario_id="primary-fail-replications-positive-4",
            reducer="minimum",
            lower_bound=0.90,
            rationale="Later replications cannot replace the frozen primary.",
        ),
        StudyOperatingAcceptanceCriterion(
            criterion_id="fragile-conflict",
            metric="scenario.fragile_claim_detection_rate",
            scenario_id="fragile-one-conflict-4",
            reducer="minimum",
            lower_bound=0.90,
            rationale="Zero-margin directional conflict must remain visible.",
        ),
        StudyOperatingAcceptanceCriterion(
            criterion_id="redundant-conflict",
            metric="scenario.sensitivity_warning_match_rate",
            scenario_id="redundant-one-conflict-5",
            reducer="minimum",
            lower_bound=0.90,
            rationale="Positive replication margin must not suppress heterogeneity warning.",
        ),
    )


def _search_authority() -> BMRBStudySearchAuthority:
    return BMRBStudySearchAuthority(
        authority_id="study-seal-search-v1",
        multiplicity_plan=winner_picking_demo_plan(exploratory_candidates=2),
        adaptive_search_mode="forbidden",
        scientific_rationale=(
            "Confirmatory evidence uses one complete closed family; adaptive discovery is disabled."
        ),
    )


def _plan(development_artifact: dict[str, object]) -> BMRBStudyOperatingAcceptancePlan:
    evaluation = _policy("evaluation")
    return BMRBStudyOperatingAcceptancePlan.from_verified_development_artifact(
        study_id="study-evaluation-seal-test-v1",
        development_evidence_ref="artifact://study-operating-development-test",
        development_artifact=development_artifact,
        evaluation_policy=evaluation,
        hierarchy_authority=BMRBStudyHierarchyAuthority.from_operating_policy(evaluation),
        search_authority=_search_authority(),
        criteria=_criteria(),
        scientific_rationale=(
            "Freeze higher-level synthetic operating acceptance authority before evaluation."
        ),
    )


def _refingerprint(payload: dict[str, object]) -> None:
    policy = payload["policy"]
    assert isinstance(policy, dict)
    grid = policy["grid"]
    assert isinstance(grid, dict)
    policy["grid_fingerprint"] = canonical_scientific_fingerprint(
        "quantumbci.bmrb-study-operating-grid.v1", grid
    )
    policy_core = {key: value for key, value in policy.items() if key != "policy_fingerprint"}
    policy["policy_fingerprint"] = canonical_scientific_fingerprint(
        "quantumbci.bmrb-study-operating-policy.v1", policy_core
    )
    core = {key: value for key, value in payload.items() if key != "artifact_fingerprint"}
    payload["artifact_fingerprint"] = canonical_scientific_fingerprint(
        "quantumbci.bmrb-study-operating-result.v1", core
    )


def test_study_operating_verifier_accepts_real_production_hierarchy(
    development_artifact: dict[str, object],
) -> None:
    verify_bmrb_study_operating_mapping(development_artifact)


def test_verifier_rejects_semantic_tampering_even_after_refingerprinting(
    development_artifact: dict[str, object],
) -> None:
    tampered = deepcopy(development_artifact)
    cells = tampered["cells"]
    assert isinstance(cells, list)
    assert isinstance(cells[0], dict)
    cells[0]["decision_error_rate"] = 0.5
    _refingerprint(tampered)
    with pytest.raises(ValueError, match="decision_error_rate"):
        verify_bmrb_study_operating_mapping(tampered)


def test_verifier_rejects_duplicate_grid_axes_even_with_fresh_fingerprints(
    development_artifact: dict[str, object],
) -> None:
    tampered = deepcopy(development_artifact)
    policy = tampered["policy"]
    assert isinstance(policy, dict)
    grid = policy["grid"]
    assert isinstance(grid, dict)
    grid["participant_counts"] = [8, 8]
    grid["cell_count"] = 16
    _refingerprint(tampered)
    with pytest.raises(ValueError, match="duplicate"):
        verify_bmrb_study_operating_mapping(tampered)


def test_acceptance_plan_binds_identical_development_and_evaluation_semantics(
    development_artifact: dict[str, object],
) -> None:
    plan = _plan(development_artifact)
    assert plan.development_policy.partition == "development"
    assert plan.evaluation_policy.partition == "evaluation"
    assert (
        plan.development_policy.seed_partition.fingerprint
        == plan.evaluation_policy.seed_partition.fingerprint
    )
    mapping = plan.to_mapping()
    assert mapping["evaluation_executed"] is False
    assert mapping["acceptance_criteria_frozen_before_evaluation"] is True
    assert BMRBStudyOperatingAcceptancePlan.from_mapping(mapping).to_mapping() == mapping
    with pytest.raises(RuntimeError, match="evaluation partition remains sealed"):
        run_bmrb_study_operating_characteristics(plan.evaluation_policy)


def test_acceptance_plan_rejects_scientific_policy_drift(
    development_artifact: dict[str, object],
) -> None:
    evaluation = replace(_policy("evaluation"), sensitivity_max_effect_range=0.2)
    with pytest.raises(ValueError, match="identical scientific semantics"):
        BMRBStudyOperatingAcceptancePlan.from_verified_development_artifact(
            study_id="drift-test",
            development_evidence_ref="artifact://development",
            development_artifact=development_artifact,
            evaluation_policy=evaluation,
            hierarchy_authority=BMRBStudyHierarchyAuthority.from_operating_policy(evaluation),
            search_authority=_search_authority(),
            criteria=_criteria(),
            scientific_rationale="Reject policy drift after development evidence is observed.",
        )


def test_acceptance_plan_rejects_seed_authority_drift(
    development_artifact: dict[str, object],
) -> None:
    changed_seed = StudySimulationSeedPartition(evaluation_offset=2_131_000_000)
    evaluation = _policy("evaluation", seed=changed_seed)
    with pytest.raises(ValueError, match="seed_partition_fingerprint"):
        BMRBStudyOperatingAcceptancePlan.from_verified_development_artifact(
            study_id="seed-drift-test",
            development_evidence_ref="artifact://development",
            development_artifact=development_artifact,
            evaluation_policy=evaluation,
            hierarchy_authority=BMRBStudyHierarchyAuthority.from_operating_policy(evaluation),
            search_authority=_search_authority(),
            criteria=_criteria(),
            scientific_rationale="Reject hidden final-evaluation RNG authority changes.",
        )


def test_required_acceptance_bounds_are_explicit_and_directional(
    development_artifact: dict[str, object],
) -> None:
    criteria = list(_criteria())
    criteria[0] = StudyOperatingAcceptanceCriterion(
        criterion_id="false-promotion",
        metric="aggregate.mean_false_promotion_rate",
        lower_bound=0.0,
        rationale="Wrong-direction software fixture.",
    )
    evaluation = _policy("evaluation")
    with pytest.raises(ValueError, match="explicit upper bound"):
        BMRBStudyOperatingAcceptancePlan.from_verified_development_artifact(
            study_id="criteria-direction-test",
            development_evidence_ref="artifact://development",
            development_artifact=development_artifact,
            evaluation_policy=evaluation,
            hierarchy_authority=BMRBStudyHierarchyAuthority.from_operating_policy(evaluation),
            search_authority=_search_authority(),
            criteria=tuple(criteria),
            scientific_rationale="Acceptance direction must be declared before evaluation.",
        )


def test_search_authority_freezes_multiplicity_and_keeps_adaptive_discovery_nonconfirmatory() -> None:
    multiplicity = winner_picking_demo_plan(exploratory_candidates=2)
    adaptive = BMRBAdaptiveSearchPlan(
        plan_id="study-seal-adaptive-test",
        multiplicity_plan=multiplicity,
        max_evaluations=2,
        routing_effect_cutoff=0.05,
        above_cutoff_stride=1,
        below_cutoff_stride=2,
        scientific_rationale="Synthetic adaptive-routing authority fixture.",
    )
    authority = BMRBStudySearchAuthority(
        authority_id="adaptive-study-search",
        multiplicity_plan=multiplicity,
        adaptive_search_mode="predeclared_plan",
        adaptive_search_plan=adaptive,
        scientific_rationale="Adaptive inspection cannot redefine confirmatory evidence.",
    )
    mapping = authority.to_mapping()
    assert mapping["adaptive_discovery_defines_confirmatory_evidence_set"] is False
    assert mapping["confirmatory_evidence_set"] == "complete_closed_multiplicity_family"
    assert BMRBStudySearchAuthority.from_mapping(mapping).to_mapping() == mapping

    mismatched = winner_picking_demo_plan(exploratory_candidates=3)
    with pytest.raises(ValueError, match="same closed family"):
        BMRBStudySearchAuthority(
            authority_id="mismatched-adaptive-study-search",
            multiplicity_plan=mismatched,
            adaptive_search_mode="predeclared_plan",
            adaptive_search_plan=adaptive,
            scientific_rationale="Mismatch must fail closed.",
        )


def test_hierarchy_authority_keeps_equal_study_votes_and_sensitivity_nonpromotion() -> None:
    authority = BMRBStudyHierarchyAuthority.from_operating_policy(_policy("evaluation"))
    mapping = authority.to_mapping()
    assert mapping["primary_must_pass"] is True
    assert mapping["study_weighting"] == "one_independent_study_one_vote"
    assert mapping["participant_weighting_role"] == "diagnostic_only"
    assert mapping["sensitivity_promotion_authoritative"] is False
    assert BMRBStudyHierarchyAuthority.from_mapping(mapping).to_mapping() == mapping

    tampered = dict(mapping)
    tampered["sensitivity_promotion_authoritative"] = True
    with pytest.raises(ValueError, match="non-promotion-authoritative"):
        BMRBStudyHierarchyAuthority.from_mapping(tampered)


def test_external_preregistration_must_bind_exact_plan_and_seal_is_canonical(
    development_artifact: dict[str, object],
) -> None:
    plan = _plan(development_artifact)
    registration = PreregistrationEvidence(
        registration_uri="https://osf.io/example/register/study-operating-final-v1",
        registered_at="2026-08-31T22:00:00Z",
        registration_document_sha256="a" * 64,
        registered_policy_sha256=plan.plan_fingerprint,
        registry="synthetic-test-registry",
    )
    seal = BMRBStudyEvaluationSeal(plan=plan, preregistration=registration)
    mapping = seal.to_mapping()
    assert mapping["evaluation_executed"] is False
    assert mapping["sensitivity_promotion_authoritative"] is False
    assert mapping["physical_quantum_promotion_eligible"] is False
    assert verify_bmrb_study_evaluation_seal_mapping(mapping).to_mapping() == mapping

    wrong = replace(registration, registered_policy_sha256="b" * 64)
    with pytest.raises(ValueError, match="does not bind"):
        BMRBStudyEvaluationSeal(plan=plan, preregistration=wrong)


def test_seal_tampering_fails_closed_even_if_nested_claim_is_changed(
    development_artifact: dict[str, object],
) -> None:
    plan = _plan(development_artifact)
    registration = PreregistrationEvidence(
        registration_uri="https://osf.io/example/register/study-operating-final-v1",
        registered_at="2026-08-31T22:00:00Z",
        registration_document_sha256="c" * 64,
        registered_policy_sha256=plan.plan_fingerprint,
        registry="synthetic-test-registry",
    )
    mapping = BMRBStudyEvaluationSeal(plan=plan, preregistration=registration).to_mapping()
    mapping["physical_quantum_promotion_eligible"] = True
    core = {key: value for key, value in mapping.items() if key != "artifact_fingerprint"}
    mapping["artifact_fingerprint"] = canonical_scientific_fingerprint(
        "quantumbci.bmrb-study-operating-evaluation-seal.v1", core
    )
    with pytest.raises(ValueError, match="physical-quantum"):
        verify_bmrb_study_evaluation_seal_mapping(mapping)
