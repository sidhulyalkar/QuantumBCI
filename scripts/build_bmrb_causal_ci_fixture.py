"""Generate deterministic source artifacts for the installed BMRB causal CI contract.

This script creates evidence inputs only. CI must run ``quantumbci-bmrb dynamics`` and
``quantumbci-bmrb causal`` themselves so packaging/console integration is exercised.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantumbci import (
    EVIDENCE_PACK_SCHEMA,
    build_matched_classical_recovery_evidence,
    derive_evidence_pack_validation,
    neuros_mechint_stable_hash,
)
from quantumbci.neuros_mechint_artifacts import DOSE_RESPONSE_SCHEMA


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        "authority_fingerprint": f"ci-authority-{participant}-{occasion}",
        "data_sha256": f"ci-data-{participant}-{occasion}",
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


def _dose(participant: str) -> dict:
    scientific = {
        "spec": {
            "study_id": f"ci-dose-{participant}",
            "intervention_id": "erase_candidate_information",
            "expected_direction": 1,
            "manifold": {
                "kind": "conditional_resample",
                "description": "CI discovery-fitted donor intervention",
                "donor_pool_id": "ci-discovery-donors",
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
        "input_hash": f"ci-hash-{example}",
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
        "candidate": {
            "name": "candidate",
            "targets": ["candidate"],
            "scores": {},
            "source": "ci-fixture",
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
            "dataset_id": f"ci-dataset-{participant}",
            "dataset_revision": "v1",
            "discovery_method": "ci-fixture",
            "evidence_tier": {"label": "integration", "level": 3},
            "intervention_baselines": ["zero", "mean"],
            "metadata": {},
            "metric_name": "held_out_score",
            "model_id": "ci-model",
            "model_revision": "v1",
            "pack_id": f"ci-pack-{participant}",
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


def build_fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    dynamics_cases = []
    for participant in range(1, 4):
        for occasion in range(1, 3):
            filename = f"stability-p{participant}-s{occasion}.json"
            _write(root / filename, _stability(participant, occasion))
            dynamics_cases.append(
                {
                    "participant_id": f"p{participant}",
                    "occasion_id": f"s{occasion}",
                    "case_id": f"p{participant}-s{occasion}",
                    "artifact": filename,
                }
            )
    _write(
        root / "dynamics-cases.json",
        {
            "schema_version": 1,
            "study_id": "ci-bmrb-causal-v016",
            "metadata": {"fixture": "installed-cli"},
            "cases": dynamics_cases,
        },
    )

    causal_cases = []
    for participant_index in range(1, 4):
        participant = f"p{participant_index}"
        case_id = f"{participant}-s1"
        dose = _dose(participant)
        faith = _faithfulness(participant)
        dose_name = f"dose-{participant}.json"
        faith_name = f"faith-{participant}.json"
        recovery_name = f"recovery-{participant}.json"
        _write(root / dose_name, dose)
        _write(root / faith_name, faith)
        recovery = build_matched_classical_recovery_evidence(
            study_id="ci-bmrb-causal-v016",
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
            classical_evidence_fingerprint=f"ci-classical-{participant}",
        )
        _write(root / recovery_name, recovery.to_mapping())
        causal_cases.append(
            {
                "participant_id": participant,
                "occasion_id": "s1",
                "case_id": case_id,
                "dose_response_artifact": dose_name,
                "faithfulness_artifact": faith_name,
                "matched_recovery_artifact": recovery_name,
            }
        )

    _write(
        root / "causal-manifest.json",
        {
            "schema_version": 1,
            "study_id": "ci-bmrb-causal-v016",
            "upstream_bmrb": "upstream/bmrb_dynamics.json",
            "policy": {
                "policy_id": "ci-causal-policy-v1",
                "preregistered": True,
                "min_participants": 3,
                "min_direction_match_fraction": 0.8,
                "min_dose_response_pass_fraction": 0.8,
                "min_faithfulness_pass_fraction": 0.8,
                "min_mean_necessity_fraction": 0.5,
                "min_mean_joint_random_percentile": 0.95,
                "max_mean_classical_recovery_fraction": 0.25,
            },
            "metadata": {"fixture": "installed-cli"},
            "cases": causal_cases,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root")
    args = parser.parse_args()
    build_fixture(Path(args.output_root).expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
