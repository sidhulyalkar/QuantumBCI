"""Materialize the v0.12 E002 switching-state classical control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..switching_dynamics import (
    INITIALIZATION_IDS,
    SWITCHING_MODEL_ID,
    _fit_from_initialization,
    _pairs_xy,
    _transition_row_chains,
    run_switching_state_control,
)
from ..trajectory_authority import load_trajectory_contract_descriptor
from .probabilistic_ssm_task import (
    _read_json,
    _transition_sha256,
    _write_json,
    build_probabilistic_ssm_artifact,
)


def _verify_probabilistic_artifact(
    supplied: dict[str, Any],
    *,
    descriptor_path: Path,
    trajectory_index_path: Path,
    matched_path: Path,
    classical_controls_path: Path,
) -> dict[str, Any]:
    if supplied.get("experiment") != "E002":
        raise ValueError("probabilistic state-space artifact is not E002")
    if supplied.get("artifact_role") != "probabilistic_latent_state_space_control":
        raise ValueError("probabilistic state-space artifact has the wrong artifact role")
    if supplied.get("status") != "pass":
        raise ValueError("probabilistic state-space artifact did not pass")
    if not bool(supplied.get("probabilistic_latent_state_space_control_complete", False)):
        raise ValueError("v0.11 probabilistic state-space control is incomplete")
    if not bool(supplied.get("switching_state_control_required", False)):
        raise ValueError("v0.11 artifact does not require the switching-state control")
    if bool(supplied.get("intervention_stage_eligible", True)):
        raise ValueError("v0.11 artifact incorrectly permits intervention promotion")
    if bool(supplied.get("physical_quantum_promotion_eligible", True)):
        raise ValueError("v0.11 artifact incorrectly permits physical-quantum promotion")

    expected = build_probabilistic_ssm_artifact(
        descriptor_path=descriptor_path,
        trajectory_index_path=trajectory_index_path,
        matched_path=matched_path,
        classical_controls_path=classical_controls_path,
    )
    if supplied != expected:
        raise ValueError(
            "probabilistic state-space artifact differs from independent v0.11 reconstruction"
        )
    return expected


def _multistart_diagnostics(data: Any, authority: Any) -> dict[str, Any]:
    pairs = authority.transition_pairs(data, "fit")
    x, y = _pairs_xy(data, pairs)
    chains = _transition_row_chains(pairs)
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for initialization_id in INITIALIZATION_IDS:
        try:
            result = _fit_from_initialization(x, y, chains, initialization_id)
            successes.append(
                {
                    "initialization_id": initialization_id,
                    "status": "success",
                    "fit_log_likelihood": float(result.fit_log_likelihood),
                    "fit_mean_nll": float(result.fit_mean_nll),
                    "iterations": int(result.iterations),
                    "converged": bool(result.converged),
                    "canonicalization_permutation": [
                        int(value) for value in result.canonicalization_permutation
                    ],
                }
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            failures.append(
                {
                    "initialization_id": initialization_id,
                    "status": "failure",
                    "message": str(exc),
                }
            )

    if not successes:
        raise RuntimeError("all switching-state initializations failed")
    best = sorted(
        successes,
        key=lambda item: (-float(item["fit_log_likelihood"]), str(item["initialization_id"])),
    )[0]
    return {
        "initialization_ids": list(INITIALIZATION_IDS),
        "successes": sorted(successes, key=lambda item: str(item["initialization_id"])),
        "failures": sorted(failures, key=lambda item: str(item["initialization_id"])),
        "success_count": len(successes),
        "failure_count": len(failures),
        "best_initialization_id": str(best["initialization_id"]),
        "best_fit_log_likelihood": float(best["fit_log_likelihood"]),
    }


def build_switching_state_artifact(
    *,
    descriptor_path: Path,
    trajectory_index_path: Path,
    matched_path: Path,
    classical_controls_path: Path,
    probabilistic_ssm_path: Path,
) -> dict[str, Any]:
    data, authority = load_trajectory_contract_descriptor(descriptor_path)
    probabilistic = _read_json(
        probabilistic_ssm_path,
        label="probabilistic state-space artifact",
    )
    expected_probabilistic = _verify_probabilistic_artifact(
        probabilistic,
        descriptor_path=descriptor_path,
        trajectory_index_path=trajectory_index_path,
        matched_path=matched_path,
        classical_controls_path=classical_controls_path,
    )

    diagnostics = _multistart_diagnostics(data, authority)
    result = run_switching_state_control(data, authority)
    payload = result.to_mapping()
    if payload.get("model", {}).get("model_id") != SWITCHING_MODEL_ID:
        raise RuntimeError("switching-state executor returned an unexpected model identity")
    if result.model.initialization_id != diagnostics["best_initialization_id"]:
        raise RuntimeError("selected switching initialization differs from multistart audit")
    if not np.isclose(
        result.model.fit_log_likelihood,
        float(diagnostics["best_fit_log_likelihood"]),
        rtol=0.0,
        atol=1e-10,
    ):
        raise RuntimeError("selected switching likelihood differs from multistart audit")

    fit_pairs = authority.transition_pairs(data, "fit")
    calibration_pairs = authority.transition_pairs(data, "calibration")
    evaluation_pairs = authority.transition_pairs(data, "evaluation")
    fit_sha = _transition_sha256(fit_pairs)
    calibration_sha = _transition_sha256(calibration_pairs)
    evaluation_sha = _transition_sha256(evaluation_pairs)

    for field, expected in (
        ("authority_fingerprint", authority.authority_fingerprint),
        ("data_sha256", data.data_sha256),
    ):
        if str(payload.get(field)) != expected:
            raise RuntimeError(f"switching-state control produced mismatched {field}")
        if str(expected_probabilistic.get(field)) != expected:
            raise ValueError(f"v0.11 probabilistic artifact has mismatched {field}")
    if str(expected_probabilistic.get("fit_transition_sha256")) != fit_sha:
        raise ValueError("v0.11 fit-transition identity differs from current authority")
    if str(expected_probabilistic.get("calibration_transition_sha256")) != calibration_sha:
        raise ValueError("v0.11 calibration-transition identity differs from current authority")
    if str(expected_probabilistic.get("evaluation_transition_sha256")) != evaluation_sha:
        raise ValueError("v0.11 evaluation-transition identity differs from current authority")

    direct = expected_probabilistic["direct_gaussian_var"]["evaluation_sequential"]
    kalman = expected_probabilistic["identity_observation_kalman"]["evaluation_sequential"]
    switching = payload["evaluation_metrics"]

    return {
        **payload,
        "schema_version": 2,
        "status": "pass",
        "artifact_role": "switching_state_classical_control",
        "descriptor_name": descriptor_path.name,
        "trajectory_index_name": trajectory_index_path.name,
        "matched_baseline_name": matched_path.name,
        "classical_controls_name": classical_controls_path.name,
        "probabilistic_ssm_name": probabilistic_ssm_path.name,
        "fit_transition_sha256": fit_sha,
        "calibration_transition_sha256": calibration_sha,
        "evaluation_transition_sha256": evaluation_sha,
        "upstream_probabilistic_artifact_verified": True,
        "upstream_probabilistic_artifact_reconstructed": True,
        "multistart_diagnostics": diagnostics,
        "sequential_information_set": (
            "Observed x_t plus regime belief updated only from prior observations within the "
            "same evidence-role trajectory chain. Regime belief resets at each chain boundary."
        ),
        "matched_sequential_comparisons": {
            "direct_gaussian_var_minus_switching_mean_nll": float(
                direct["mean_nll"] - switching["mean_nll"]
            ),
            "kalman_minus_switching_mean_nll": float(
                kalman["mean_nll"] - switching["mean_nll"]
            ),
            "direct_gaussian_var_minus_switching_rmse": float(
                direct["predictive_mean_rmse"] - switching["predictive_mean_rmse"]
            ),
            "kalman_minus_switching_rmse": float(
                kalman["predictive_mean_rmse"] - switching["predictive_mean_rmse"]
            ),
        },
        "switching_state_control_complete": True,
        "exact_open_loop_switching_forecast_complete": False,
        "open_loop_promotion_eligible": False,
        "flexible_nonlinear_control_required_when_powered": True,
        "bootstrap_stability_required": True,
        "intervention_direction_evidence_required": True,
        "intervention_stage_eligible": False,
        "dynamical_information_novel": False,
        "physical_quantum_promotion_eligible": False,
        "interpretation_ceiling": (
            "This artifact establishes a classical two-regime Markov-switching adversary "
            "under the same frozen E002 temporal authority. Sequential likelihood gains are "
            "evidence for classical regime switching, not biological state identity or quantum "
            "novelty. Exact open-loop switching prediction, flexible nonlinear controls, "
            "bootstrap stability, and intervention-direction evidence remain open gates."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", default="trajectory_contract.json")
    parser.add_argument("--trajectory-index", default="trajectory_index.json")
    parser.add_argument("--matched", default="matched_dynamics.json")
    parser.add_argument("--classical-controls", default="classical_controls.json")
    parser.add_argument("--probabilistic-ssm", default="probabilistic_ssm.json")
    parser.add_argument("--output", default="switching_state.json")
    args = parser.parse_args(argv)

    output = Path(args.output)
    try:
        payload = build_switching_state_artifact(
            descriptor_path=Path(args.descriptor),
            trajectory_index_path=Path(args.trajectory_index),
            matched_path=Path(args.matched),
            classical_controls_path=Path(args.classical_controls),
            probabilistic_ssm_path=Path(args.probabilistic_ssm),
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
