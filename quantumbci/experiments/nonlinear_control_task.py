"""Materialize the v0.13 E002 flexible nonlinear classical control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..nonlinear_dynamics import NONLINEAR_MODEL_ID, run_nonlinear_residual_control
from ..trajectory_authority import load_trajectory_contract_descriptor
from .probabilistic_ssm_task import _read_json, _write_json
from .switching_state_task import build_switching_state_artifact


def _verify_switching_artifact(
    supplied: dict[str, Any],
    *,
    descriptor_path: Path,
    trajectory_index_path: Path,
    matched_path: Path,
    classical_controls_path: Path,
    probabilistic_ssm_path: Path,
) -> dict[str, Any]:
    if supplied.get("experiment") != "E002":
        raise ValueError("switching-state artifact is not E002")
    if supplied.get("artifact_role") != "switching_state_classical_control":
        raise ValueError("switching-state artifact has the wrong artifact role")
    if supplied.get("status") != "pass":
        raise ValueError("switching-state artifact did not pass")
    if not bool(supplied.get("switching_state_control_complete", False)):
        raise ValueError("v0.12 switching-state control is incomplete")
    if not bool(supplied.get("flexible_nonlinear_control_required_when_powered", False)):
        raise ValueError("v0.12 artifact does not require the nonlinear control")
    if bool(supplied.get("intervention_stage_eligible", True)):
        raise ValueError("v0.12 artifact incorrectly permits intervention promotion")
    if bool(supplied.get("physical_quantum_promotion_eligible", True)):
        raise ValueError("v0.12 artifact incorrectly permits physical-quantum promotion")

    expected = build_switching_state_artifact(
        descriptor_path=descriptor_path,
        trajectory_index_path=trajectory_index_path,
        matched_path=matched_path,
        classical_controls_path=classical_controls_path,
        probabilistic_ssm_path=probabilistic_ssm_path,
    )
    if supplied != expected:
        raise ValueError(
            "switching-state artifact differs from independent v0.12 reconstruction"
        )
    return expected


def _full_var_mean(classical: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    controls = classical.get("controls")
    if not isinstance(controls, dict):
        raise ValueError("classical controls artifact is missing controls")
    full = controls.get("full_var1")
    if not isinstance(full, dict):
        raise ValueError("classical controls artifact is missing full_var1")
    if full.get("model_id") != "full_var1_affine":
        raise ValueError("v0.13 requires the v0.10 full_var1_affine mean model")
    transition = np.asarray(full.get("transition"), dtype=float)
    intercept = np.asarray(full.get("intercept"), dtype=float).reshape(-1)
    if transition.shape != (3, 3) or intercept.shape != (3,):
        raise ValueError("v0.13 E002 nonlinear control requires a 3D full VAR mean model")
    return transition, intercept, full


def build_nonlinear_control_artifact(
    *,
    descriptor_path: Path,
    trajectory_index_path: Path,
    matched_path: Path,
    classical_controls_path: Path,
    probabilistic_ssm_path: Path,
    switching_state_path: Path,
) -> dict[str, Any]:
    data, authority = load_trajectory_contract_descriptor(descriptor_path)
    classical = _read_json(classical_controls_path, label="classical controls artifact")
    probabilistic = _read_json(probabilistic_ssm_path, label="probabilistic state-space artifact")
    switching = _read_json(switching_state_path, label="switching-state artifact")

    expected_switching = _verify_switching_artifact(
        switching,
        descriptor_path=descriptor_path,
        trajectory_index_path=trajectory_index_path,
        matched_path=matched_path,
        classical_controls_path=classical_controls_path,
        probabilistic_ssm_path=probabilistic_ssm_path,
    )
    transition, intercept, full_var = _full_var_mean(classical)
    result = run_nonlinear_residual_control(
        data,
        authority,
        transition,
        intercept,
    )
    payload = result.to_mapping()
    if payload.get("model", {}).get("model_id") != NONLINEAR_MODEL_ID:
        raise RuntimeError("nonlinear executor returned an unexpected model identity")
    if not np.array_equal(np.asarray(payload["model"]["transition"]), transition):
        raise RuntimeError("nonlinear stage refit or changed the frozen v0.10 transition")
    if not np.array_equal(np.asarray(payload["model"]["intercept"]), intercept):
        raise RuntimeError("nonlinear stage refit or changed the frozen v0.10 intercept")

    for field in (
        "authority_fingerprint",
        "data_sha256",
        "fit_transition_sha256",
        "calibration_transition_sha256",
        "evaluation_transition_sha256",
    ):
        expected = str(expected_switching.get(field))
        if str(payload.get(field)) != expected:
            raise RuntimeError(f"nonlinear control produced mismatched {field}")

    direct_sequential = probabilistic.get("direct_gaussian_var", {}).get(
        "evaluation_sequential"
    )
    if not isinstance(direct_sequential, dict):
        raise ValueError("v0.11 artifact is missing direct Gaussian sequential evaluation")
    full_metrics = full_var.get("evaluation_metrics")
    if not isinstance(full_metrics, dict):
        raise ValueError("v0.10 full VAR artifact is missing evaluation metrics")

    nonlinear_metrics = payload["evaluation_metrics"]
    return {
        **payload,
        "schema_version": 2,
        "status": "pass",
        "artifact_role": "flexible_nonlinear_classical_control",
        "descriptor_name": descriptor_path.name,
        "trajectory_index_name": trajectory_index_path.name,
        "matched_baseline_name": matched_path.name,
        "classical_controls_name": classical_controls_path.name,
        "probabilistic_ssm_name": probabilistic_ssm_path.name,
        "switching_state_name": switching_state_path.name,
        "upstream_switching_artifact_verified": True,
        "upstream_switching_artifact_reconstructed": True,
        "affine_mean_source": "v0.10:controls.full_var1",
        "affine_mean_refit": False,
        "matched_information_set_comparisons": {
            "direct_gaussian_var_minus_nonlinear_one_step_mean_nll": float(
                direct_sequential["mean_nll"] - nonlinear_metrics["one_step_mean_nll"]
            ),
            "direct_gaussian_var_minus_nonlinear_one_step_rmse": float(
                direct_sequential["predictive_mean_rmse"] - nonlinear_metrics["one_step_rmse"]
            ),
            "full_var_minus_nonlinear_rollout_rmse": float(
                full_metrics["rollout_rmse"] - nonlinear_metrics["rollout_rmse"]
            ),
        },
        "comparison_exclusions": {
            "kalman_sequential": (
                "not a matched information set: Kalman prediction uses filtered latent history"
            ),
            "switching_sequential": (
                "not a matched information set: switching prediction uses regime-belief history"
            ),
            "nonlinear_rollout_likelihood": (
                "not implemented: nonlinear predictive uncertainty is not propagated in v0.13"
            ),
        },
        "flexible_nonlinear_control_complete": True,
        "bootstrap_stability_required": True,
        "intervention_direction_evidence_required": True,
        "intervention_stage_eligible": False,
        "dynamical_information_novel": False,
        "physical_quantum_promotion_eligible": False,
        "interpretation_ceiling": (
            "This artifact establishes a calibrated flexible classical nonlinear residual "
            "control around the frozen v0.10 affine mean. A predictive gain is evidence for "
            "classical nonlinear dynamics, not quantum novelty. Bootstrap stability and "
            "intervention-direction evidence remain required before mechanistic promotion."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", default="trajectory_contract.json")
    parser.add_argument("--trajectory-index", default="trajectory_index.json")
    parser.add_argument("--matched", default="matched_dynamics.json")
    parser.add_argument("--classical-controls", default="classical_controls.json")
    parser.add_argument("--probabilistic-ssm", default="probabilistic_ssm.json")
    parser.add_argument("--switching-state", default="switching_state.json")
    parser.add_argument("--output", default="nonlinear_control.json")
    args = parser.parse_args(argv)

    output = Path(args.output)
    try:
        payload = build_nonlinear_control_artifact(
            descriptor_path=Path(args.descriptor),
            trajectory_index_path=Path(args.trajectory_index),
            matched_path=Path(args.matched),
            classical_controls_path=Path(args.classical_controls),
            probabilistic_ssm_path=Path(args.probabilistic_ssm),
            switching_state_path=Path(args.switching_state),
        )
        _write_json(output, payload)
        print(json.dumps({**payload, "artifact": str(output)}, sort_keys=True))
        return 0
    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        np.linalg.LinAlgError,
    ) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
