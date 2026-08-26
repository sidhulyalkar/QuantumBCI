from __future__ import annotations

from copy import deepcopy

import pytest

from quantumbci.neuros_mechint_artifacts import (
    DOSE_RESPONSE_SCHEMA,
    EVIDENCE_PACK_SCHEMA,
    derive_evidence_pack_validation,
    neuros_mechint_stable_hash,
    verify_dose_response_result,
    verify_evidence_pack_result,
)


def make_dose_response(*, endpoint_effect: float = 0.7, passed: bool = True) -> dict:
    scientific = {
        "spec": {
            "study_id": "dose-study",
            "intervention_id": "erase_candidate_information",
            "expected_direction": 1,
            "manifold": {
                "kind": "conditional_resample",
                "description": "held-out donor intervention",
                "donor_pool_id": "discovery-donors",
                "fitted_on_partition_id": "discovery",
                "expected_in_manifold": True,
                "metadata": {},
            },
            "policy": {
                "min_doses": 3,
                "min_units": 2,
                "min_monotonic_fraction": 0.75,
                "min_endpoint_effect": 0.1,
                "require_endpoints": True,
                "require_common_grid": True,
            },
            "metadata": {},
        },
        "unit_summaries": [],
        "aggregate_doses": [0.0, 0.5, 1.0],
        "aggregate_metrics": [0.0, 0.4, endpoint_effect],
        "endpoint_effect": endpoint_effect,
        "mean_monotonic_fraction": 1.0,
        "normalized_auc": 0.55,
        "passed": passed,
        "reasons": [] if passed else ["fixture failure"],
    }
    return {
        "schema_version": DOSE_RESPONSE_SCHEMA,
        **scientific,
        "study_fingerprint": neuros_mechint_stable_hash(scientific),
    }


def _faithfulness_report(*, joint: float = 0.82, passed: bool = True) -> dict:
    return {
        "all_targets": ["candidate", "control"],
        "baseline_metric": 1.0,
        "candidate": {
            "name": "candidate",
            "targets": ["candidate"],
            "scores": {},
            "source": "fixture",
        },
        "circuit_metric": 0.92,
        "complement_metric": 0.18,
        "higher_is_better": True,
        "joint_faithfulness": joint,
        "joint_random_percentile": 1.0,
        "metadata": {},
        "necessity_fraction": joint,
        "necessity_random_percentile": 1.0,
        "null_metric": 0.0,
        "passed": passed,
        "policy": {
            "min_sufficiency_fraction": 0.8,
            "min_necessity_fraction": 0.5,
            "min_random_percentile": 0.95,
        },
        "random_controls": [],
        "seed": 0,
        "sufficiency_fraction": 0.90,
        "sufficiency_random_percentile": 1.0,
    }


def _case(example: str, split: str, baseline: str, *, joint: float) -> dict:
    return {
        "example_id": example,
        "input_hash": f"hash-{example}",
        "intervention_baseline": baseline,
        "invalid_reason": None,
        "metadata": {},
        "report": _faithfulness_report(joint=joint),
        "split": split,
        "valid": True,
    }


def _pack_identity(result: dict) -> dict:
    keys = (
        "candidate",
        "candidate_cases",
        "discovery_example_ids",
        "faithfulness_policy",
        "magnitude_candidate",
        "magnitude_cases",
        "mean_ablation_references",
        "policy",
        "spec",
        "validation_example_ids",
    )
    return {key: result[key] for key in keys}


def _refingerprint_pack(result: dict) -> None:
    result["study_fingerprint"] = neuros_mechint_stable_hash(_pack_identity(result))


def make_evidence_pack() -> dict:
    candidate_cases = [
        _case("d1", "discovery", "zero", joint=0.86),
        _case("d1", "discovery", "mean", joint=0.84),
        _case("v1", "validation", "zero", joint=0.82),
        _case("v1", "validation", "mean", joint=0.82),
        _case("v2", "validation", "zero", joint=0.80),
        _case("v2", "validation", "mean", joint=0.80),
    ]
    identity = {
        "candidate": {
            "name": "candidate",
            "targets": ["candidate"],
            "scores": {},
            "source": "fixture",
        },
        "candidate_cases": candidate_cases,
        "discovery_example_ids": ["d1"],
        "faithfulness_policy": {
            "min_sufficiency_fraction": 0.8,
            "min_necessity_fraction": 0.5,
            "min_random_percentile": 0.95,
        },
        "magnitude_candidate": None,
        "magnitude_cases": [],
        "mean_ablation_references": {},
        "policy": {
            "bootstrap_samples": 1000,
            "max_joint_generalization_drop": 0.25,
            "min_validation_examples": 2,
            "min_validation_joint_advantage_vs_magnitude": 0.0,
            "min_validation_joint_median": 0.5,
            "min_validation_pass_rate": 0.8,
            "require_all_cases_valid": True,
            "require_multiple_intervention_baselines": True,
        },
        "spec": {
            "dataset_id": "fixture-dataset",
            "dataset_revision": "fixture-v1",
            "discovery_method": "fixture",
            "evidence_tier": {"label": "integration", "level": 3},
            "intervention_baselines": ["zero", "mean"],
            "metadata": {},
            "metric_name": "fixture_metric",
            "model_id": "fixture-model",
            "model_revision": "fixture-v1",
            "pack_id": "fixture-pack",
            "random_trials": 100,
            "schema_version": EVIDENCE_PACK_SCHEMA,
            "seed": 0,
            "target_universe": ["candidate", "control"],
            "tokenizer_id": None,
            "tokenizer_revision": None,
        },
        "validation_example_ids": ["v1", "v2"],
    }
    result = {
        "schema_version": EVIDENCE_PACK_SCHEMA,
        **identity,
        "study_fingerprint": neuros_mechint_stable_hash(identity),
    }
    derived = derive_evidence_pack_validation(result)
    result["validation_aggregate"] = {
        "n_cases": derived["n_cases"],
        "n_valid_cases": derived["n_valid_cases"],
        "n_invalid_cases": derived["n_invalid_cases"],
        "n_examples": derived["n_examples"],
        "pass_rate": derived["pass_rate"],
        "valid_case_rate": derived["valid_case_rate"],
        "mean_sufficiency": derived["mean_sufficiency"],
        "mean_necessity": derived["mean_necessity"],
        "mean_joint_faithfulness": derived["mean_joint_faithfulness"],
        "median_joint_faithfulness": derived["median_joint_faithfulness"],
        "mean_joint_random_percentile": derived["mean_joint_random_percentile"],
        "joint_mean_ci95_low": 0.75,
        "joint_mean_ci95_high": 0.90,
    }
    result["promotion"] = {
        "passed": derived["promotion_passed"],
        "reasons": list(derived["promotion_reasons"]),
        "discovery_joint_median": derived["discovery_joint_median"],
        "validation_joint_median": derived["median_joint_faithfulness"],
        "joint_generalization_drop": derived["joint_generalization_drop"],
        "validation_pass_rate": derived["pass_rate"],
        "validation_joint_advantage_vs_magnitude": derived[
            "validation_joint_advantage_vs_magnitude"
        ],
    }
    return result


def test_dose_response_fingerprint_rejects_changed_endpoint() -> None:
    artifact = make_dose_response()
    assert verify_dose_response_result(artifact)["endpoint_effect"] == 0.7
    artifact["endpoint_effect"] = 9.0
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_dose_response_result(artifact)


def test_dose_response_rejects_fingerprint_valid_nonboolean_passed() -> None:
    artifact = make_dose_response()
    artifact["passed"] = "false"
    scientific = {
        key: value
        for key, value in artifact.items()
        if key not in {"schema_version", "study_fingerprint"}
    }
    artifact["study_fingerprint"] = neuros_mechint_stable_hash(scientific)
    with pytest.raises(TypeError, match="JSON boolean"):
        verify_dose_response_result(artifact)


def test_evidence_pack_derives_validation_from_fingerprint_bound_cases() -> None:
    artifact = make_evidence_pack()
    result = verify_evidence_pack_result(artifact)
    derived = derive_evidence_pack_validation(result)
    assert derived["promotion_passed"] is True
    assert derived["n_examples"] == 2
    assert derived["n_cases"] == 4
    assert derived["mean_necessity"] == pytest.approx(0.81)
    assert derived["mean_joint_random_percentile"] == 1.0


def test_evidence_pack_rejects_tampered_convenience_aggregate() -> None:
    artifact = make_evidence_pack()
    artifact["validation_aggregate"]["mean_necessity"] = 0.999
    with pytest.raises(ValueError, match="validation_aggregate mismatch"):
        verify_evidence_pack_result(artifact)


def test_evidence_pack_rejects_tampered_fingerprint_bound_case() -> None:
    artifact = make_evidence_pack()
    tampered = deepcopy(artifact)
    tampered["candidate_cases"][2]["report"]["necessity_fraction"] = 0.01
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_evidence_pack_result(tampered)


def test_evidence_pack_rejects_tampered_promotion_summary() -> None:
    artifact = make_evidence_pack()
    artifact["promotion"]["passed"] = False
    with pytest.raises(ValueError, match="promotion summary mismatch"):
        verify_evidence_pack_result(artifact)


def test_evidence_pack_rejects_fingerprint_valid_string_case_valid_flag() -> None:
    artifact = make_evidence_pack()
    artifact["candidate_cases"][2]["valid"] = "false"
    _refingerprint_pack(artifact)
    with pytest.raises(TypeError, match="case\[2\]\.valid.*JSON boolean"):
        verify_evidence_pack_result(artifact)


def test_evidence_pack_rejects_fingerprint_valid_integer_report_passed() -> None:
    artifact = make_evidence_pack()
    artifact["candidate_cases"][2]["report"]["passed"] = 1
    _refingerprint_pack(artifact)
    with pytest.raises(TypeError, match="report\.passed.*JSON boolean"):
        verify_evidence_pack_result(artifact)


def test_evidence_pack_rejects_fingerprint_valid_string_policy_flag() -> None:
    artifact = make_evidence_pack()
    artifact["policy"]["require_all_cases_valid"] = "false"
    _refingerprint_pack(artifact)
    with pytest.raises(TypeError, match="require_all_cases_valid.*JSON boolean"):
        verify_evidence_pack_result(artifact)


def test_evidence_pack_rejects_nonboolean_unbound_promotion_flag() -> None:
    artifact = make_evidence_pack()
    artifact["promotion"]["passed"] = "true"
    with pytest.raises(TypeError, match="promotion\.passed.*JSON boolean"):
        verify_evidence_pack_result(artifact)
