from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from quantumbci.bmrb import build_bmrb_dynamics_bundle, write_bmrb_dynamics_bundle
from quantumbci.bmrb_causal import build_bmrb_causal_bundle, write_bmrb_causal_bundle
from quantumbci.matched_recovery import build_matched_classical_recovery_evidence
from quantumbci.neuros_mechint_artifacts import (
    DOSE_RESPONSE_SCHEMA,
    EVIDENCE_PACK_SCHEMA,
    derive_evidence_pack_validation,
    neuros_mechint_stable_hash,
)
from quantumbci.recapitulation import EvidenceTier, GateStatus


def _stability(participant: int, occasion: int) -> dict:
    base = 0.2 * participant
    return {
        "schema_version": 2,
        "experiment": "E002",
        "claim_class": "quantum_inspired",
        "artifact_role": "bootstrap_stability_evidence",
        "status": "pass",
        "evaluation_resampled": False,
        "single_case_bootstrap_is_icc": False,
        "participant_icc_computed": False,
        "stability_gate_defined": False,
        "stability_gate_pass": None,
        "predictive_adversary_ladder_complete": True,
        "dynamical_information_novel": False,
        "authority_fingerprint": f"authority-{participant}-{occasion}",
        "data_sha256": f"data-{participant}-{occasion}",
        "point_estimates": {
            "omega_x": 0.7 + base + 0.01 * occasion,
            "omega_z": -0.5 - base - 0.01 * occasion,
            "gamma_dephasing": 0.18 + 0.02 * participant + 0.005 * occasion,
            "gamma_relaxation": 0.25 + 0.03 * participant + 0.004 * occasion,
            "canonical_structure_residual": 0.08 + 0.01 * participant,
            "canonical_minus_affine_one_step_rmse": 0.03 + 0.002 * occasion,
            "canonical_minus_affine_rollout_rmse": 0.05 + 0.003 * occasion,
            "direct_minus_nonlinear_mean_nll": 0.02 + 0.004 * participant,
            "direct_minus_nonlinear_one_step_rmse": 0.01 + 0.002 * participant,
        },
    }


def _write_upstream(root: Path) -> Path:
    cases = []
    for participant in range(1, 4):
        for occasion in range(1, 3):
            name = f"stability-p{participant}-s{occasion}.json"
            (root / name).write_text(
                json.dumps(_stability(participant, occasion)), encoding="utf-8"
            )
            cases.append(
                {
                    "participant_id": f"p{participant}",
                    "occasion_id": f"s{occasion}",
                    "case_id": f"p{participant}-s{occasion}",
                    "artifact": name,
                }
            )
    manifest = root / "dynamics-cases.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "study_id": "causal-study",
                "metadata": {"fixture": "v016-chain"},
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )
    bundle = build_bmrb_dynamics_bundle(manifest, n_resamples=100, seed=41)
    path, _ = write_bmrb_dynamics_bundle(bundle, root / "upstream")
    return path


def _dose(participant: str) -> dict:
    scientific = {
        "spec": {
            "study_id": f"dose-{participant}",
            "intervention_id": "erase_candidate_information",
            "expected_direction": 1,
            "manifold": {
                "kind": "conditional_resample",
                "description": "discovery-fitted donor intervention",
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
        "aggregate_metrics": [0.0, 0.4, 0.7],
        "endpoint_effect": 0.7,
        "mean_monotonic_fraction": 1.0,
        "normalized_auc": 0.55,
        "passed": True,
        "reasons": [],
    }
    return {
        "schema_version": DOSE_RESPONSE_SCHEMA,
        **scientific,
        "study_fingerprint": neuros_mechint_stable_hash(scientific),
    }


def _faith_report(joint: float) -> dict:
    return {
        "joint_faithfulness": joint,
        "joint_random_percentile": 1.0,
        "necessity_fraction": joint,
        "passed": True,
        "sufficiency_fraction": 0.90,
    }


def _faith_case(example: str, split: str, baseline: str, joint: float) -> dict:
    return {
        "example_id": example,
        "input_hash": f"hash-{example}",
        "intervention_baseline": baseline,
        "invalid_reason": None,
        "metadata": {},
        "report": _faith_report(joint),
        "split": split,
        "valid": True,
    }


def _faithfulness(participant: str) -> dict:
    candidate_cases = [
        _faith_case("d1", "discovery", "zero", 0.86),
        _faith_case("d1", "discovery", "mean", 0.84),
        _faith_case("v1", "validation", "zero", 0.82),
        _faith_case("v1", "validation", "mean", 0.82),
        _faith_case("v2", "validation", "zero", 0.80),
        _faith_case("v2", "validation", "mean", 0.80),
    ]
    identity = {
        "candidate": {"name": "candidate", "targets": ["candidate"], "scores": {}, "source": "fixture"},
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
            "dataset_id": f"fixture-{participant}",
            "dataset_revision": "v1",
            "discovery_method": "fixture",
            "evidence_tier": {"label": "integration", "level": 3},
            "intervention_baselines": ["zero", "mean"],
            "metadata": {},
            "metric_name": "held_out_score",
            "model_id": "fixture-model",
            "model_revision": "v1",
            "pack_id": f"pack-{participant}",
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
        "validation_joint_advantage_vs_magnitude": derived["validation_joint_advantage_vs_magnitude"],
    }
    return result


def _write_causal_manifest(root: Path, upstream: Path, *, preregistered: bool = True) -> Path:
    cases = []
    for participant_index in range(1, 4):
        participant = f"p{participant_index}"
        case_id = f"{participant}-s1"
        dose_name = f"dose-{participant}.json"
        faith_name = f"faith-{participant}.json"
        recovery_name = f"recovery-{participant}.json"
        dose = _dose(participant)
        faith = _faithfulness(participant)
        (root / dose_name).write_text(json.dumps(dose), encoding="utf-8")
        (root / faith_name).write_text(json.dumps(faith), encoding="utf-8")
        recovery = build_matched_classical_recovery_evidence(
            study_id="causal-study",
            participant_id=participant,
            occasion_id="s1",
            case_id=case_id,
            mechanism_id="lindblad_latent_dynamics",
            classical_model_id="matched_nonlinear_control",
            information_set_id="same-evidence-budget-v1",
            metric_name="held_out_score",
            higher_is_better=True,
            baseline_metric=1.0,
            ablated_metric=0.5,
            recovered_metric=0.58,
            candidate_evidence_fingerprint=faith["study_fingerprint"],
            classical_evidence_fingerprint=f"classical-{participant}",
        )
        (root / recovery_name).write_text(
            json.dumps(recovery.to_mapping()), encoding="utf-8"
        )
        cases.append(
            {
                "participant_id": participant,
                "occasion_id": "s1",
                "case_id": case_id,
                "dose_response_artifact": dose_name,
                "faithfulness_artifact": faith_name,
                "matched_recovery_artifact": recovery_name,
            }
        )
    manifest = root / "causal-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "study_id": "causal-study",
                "upstream_bmrb": str(upstream),
                "policy": {
                    "policy_id": "causal-fixture-v1",
                    "preregistered": preregistered,
                    "min_participants": 3,
                    "min_direction_match_fraction": 0.8,
                    "min_dose_response_pass_fraction": 0.8,
                    "min_faithfulness_pass_fraction": 0.8,
                    "min_mean_necessity_fraction": 0.5,
                    "min_mean_joint_random_percentile": 0.95,
                    "max_mean_classical_recovery_fraction": 0.25,
                },
                "metadata": {"fixture": "v016-causal"},
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_end_to_end_causal_bundle_preserves_falsifying_upstream_ladder(tmp_path: Path) -> None:
    upstream = _write_upstream(tmp_path)
    manifest = _write_causal_manifest(tmp_path, upstream)
    bundle = build_bmrb_causal_bundle(manifest)

    assert bundle.causal_result.scientific_criteria_passed is True
    assert bundle.causal_result.promotion_eligible is True
    # The existing E002 fixture is already falsified by matched classical predictive controls.
    # Strong later causal evidence therefore cannot jump the ladder.
    assert bundle.profile.promotion_ceiling is EvidenceTier.DESCRIPTIVE
    assert bundle.profile.first_failing_gate == "matched_classical_adversaries"
    causal = next(
        gate for gate in bundle.profile.gates if gate.tier is EvidenceTier.CAUSAL_MECHANISTIC
    )
    assert causal.status is GateStatus.CHARACTERIZED
    assert bundle.causal_result.mean_classical_recovery_fraction == pytest.approx(0.16)

    json_path, html_path = write_bmrb_causal_bundle(bundle, tmp_path / "causal-out")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    html = html_path.read_text(encoding="utf-8")
    assert payload["upstream_artifact_fingerprint"]
    assert payload["cases"][0]["matched_recovery_sha256"]
    assert "Matched-classical recovery evidence" in html
    assert "upstream_artifact_fingerprint" in html


def test_causal_stage_rejects_tampered_upstream_profile(tmp_path: Path) -> None:
    upstream = _write_upstream(tmp_path)
    payload = json.loads(upstream.read_text(encoding="utf-8"))
    payload["mechanism_profile"]["gates"][0]["summary"] = "tampered summary"
    upstream.write_text(json.dumps(payload), encoding="utf-8")
    manifest = _write_causal_manifest(tmp_path, upstream)
    with pytest.raises(ValueError, match="artifact fingerprint mismatch"):
        build_bmrb_causal_bundle(manifest)


def test_causal_stage_rejects_legacy_unverifiable_upstream(tmp_path: Path) -> None:
    upstream = _write_upstream(tmp_path)
    payload = json.loads(upstream.read_text(encoding="utf-8"))
    payload["schema_version"] = 1
    payload.pop("source_identity", None)
    payload.pop("artifact_fingerprint", None)
    upstream.write_text(json.dumps(payload), encoding="utf-8")
    manifest = _write_causal_manifest(tmp_path, upstream)
    with pytest.raises(ValueError, match="regenerate"):
        build_bmrb_causal_bundle(manifest)


def test_causal_manifest_rejects_string_preregistration(tmp_path: Path) -> None:
    upstream = _write_upstream(tmp_path)
    manifest = _write_causal_manifest(tmp_path, upstream)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["policy"]["preregistered"] = "false"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TypeError, match="JSON boolean"):
        build_bmrb_causal_bundle(manifest)


def test_causal_stage_rejects_mixed_information_sets(tmp_path: Path) -> None:
    upstream = _write_upstream(tmp_path)
    manifest = _write_causal_manifest(tmp_path, upstream)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    recovery_path = tmp_path / manifest_payload["cases"][0]["matched_recovery_artifact"]
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    # Rebuild a valid artifact under a different information-set authority.
    different = build_matched_classical_recovery_evidence(
        study_id=recovery["study_id"],
        participant_id=recovery["participant_id"],
        occasion_id=recovery["occasion_id"],
        case_id=recovery["case_id"],
        mechanism_id=recovery["mechanism_id"],
        classical_model_id=recovery["classical_model_id"],
        information_set_id="different-budget",
        metric_name=recovery["metric_name"],
        higher_is_better=recovery["higher_is_better"],
        baseline_metric=recovery["baseline_metric"],
        ablated_metric=recovery["ablated_metric"],
        recovered_metric=recovery["recovered_metric"],
        candidate_evidence_fingerprint=recovery["candidate_evidence_fingerprint"],
        classical_evidence_fingerprint=recovery["classical_evidence_fingerprint"],
    )
    recovery_path.write_text(json.dumps(different.to_mapping()), encoding="utf-8")
    with pytest.raises(ValueError, match="one declared information_set_id"):
        build_bmrb_causal_bundle(manifest)
