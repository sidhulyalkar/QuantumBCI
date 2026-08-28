from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from quantumbci.experiments.classical_controls_task import main as classical_main
from quantumbci.experiments.nonlinear_control_task import main as nonlinear_main
from quantumbci.experiments.probabilistic_ssm_task import main as probabilistic_main
from quantumbci.experiments.stability_task import main as stability_main
from quantumbci.experiments.switching_state_task import main as switching_main
from quantumbci.experiments.tasks import main as experiment_main


def _write_descriptor(tmp_path: Path) -> Path:
    rng = np.random.default_rng(311)
    transitions = np.asarray(
        [
            [[0.87, 0.14, 0.00], [-0.09, 0.84, 0.05], [0.03, -0.05, 0.84]],
            [[0.76, -0.16, 0.05], [0.14, 0.80, -0.08], [-0.03, 0.10, 0.81]],
        ],
        dtype=float,
    )
    intercepts = np.asarray(
        [[0.007, -0.004, 0.003], [-0.009, 0.008, -0.004]], dtype=float
    )
    variances = np.asarray(
        [[0.0009, 0.0010, 0.0008], [0.0009, 0.0010, 0.0010]], dtype=float
    )
    regime_transition = np.asarray([[0.95, 0.05], [0.08, 0.92]], dtype=float)
    n_trajectories = 6
    n_steps = 72
    states: list[np.ndarray] = []
    ids: list[str] = []
    for trajectory in range(n_trajectories):
        state = rng.normal(0.0, 0.16, size=3)
        regime = trajectory % 2
        for step in range(n_steps):
            if step > 0:
                regime = int(rng.choice(2, p=regime_transition[regime]))
                nonlinear = np.asarray(
                    [
                        0.020 * np.sin(3.0 * state[1]),
                        0.018 * np.sin(3.0 * state[2]),
                        0.019 * np.sin(3.0 * state[0]),
                    ]
                )
                state = (
                    transitions[regime] @ state
                    + intercepts[regime]
                    + nonlinear
                    + rng.normal(0.0, np.sqrt(variances[regime]))
                )
            states.append(state.copy())
            ids.append(f"trajectory-{trajectory}")

    state_array = np.asarray(states, dtype=float)
    starts = np.tile(np.arange(n_steps, dtype=float), n_trajectories)
    stops = starts + 0.5
    np.save(tmp_path / "states.npy", state_array)
    np.save(tmp_path / "ids.npy", np.asarray(ids))
    np.save(tmp_path / "starts.npy", starts)
    np.save(tmp_path / "stops.npy", stops)

    fit: list[int] = []
    calibration: list[int] = []
    evaluation: list[int] = []
    for trajectory in range(n_trajectories):
        base = trajectory * n_steps
        fit.extend(range(base, base + 50))
        calibration.extend(range(base + 50, base + 62))
        evaluation.extend(range(base + 62, base + n_steps))

    descriptor = {
        "schema_version": 1,
        "dataset_id": "ci-v014-stability",
        "case_id": "ci-v014-case",
        "latent_dimension": 3,
        "time_step_policy": "fixed",
        "expected_window_seconds": 0.5,
        "expected_step_seconds": 1.0,
        "step_tolerance_seconds": 1e-12,
        "purge_seconds": 0.0,
        "upstream_authority_fingerprint": "neuros-ci-v014",
        "source_revisions": {"quantumbci": "ci-v014", "encoder": "frozen-ci"},
        "data_metadata": {"state_surface": "observed_coordinates", "fixture": "v014"},
        "data": {
            "states": "states.npy",
            "trajectory_ids": "ids.npy",
            "start_times_s": "starts.npy",
            "stop_times_s": "stops.npy",
        },
        "split": {
            "fit_indices": fit,
            "calibration_indices": calibration,
            "evaluation_indices": evaluation,
            "representation_fit_indices": fit,
        },
    }
    path = tmp_path / "trajectory_contract.json"
    path.write_text(json.dumps(descriptor), encoding="utf-8")
    return path


def _materialize_v013_chain(tmp_path: Path, capsys):
    descriptor = _write_descriptor(tmp_path)
    trajectory = tmp_path / "trajectory_index.json"
    matched = tmp_path / "matched_dynamics.json"
    classical = tmp_path / "classical_controls.json"
    probabilistic = tmp_path / "probabilistic_ssm.json"
    switching = tmp_path / "switching_state.json"
    nonlinear = tmp_path / "nonlinear_control.json"

    assert experiment_main([
        "trajectory-contract", "E002", "--input", str(descriptor), "--output", str(trajectory)
    ]) == 0
    capsys.readouterr()
    assert experiment_main([
        "fit-matched-dynamics", "E002", "--input", str(descriptor),
        "--trajectory-index", str(trajectory), "--output", str(matched), "--ridge", "0"
    ]) == 0
    capsys.readouterr()
    assert classical_main([
        "--descriptor", str(descriptor), "--trajectory-index", str(trajectory),
        "--matched", str(matched), "--output", str(classical)
    ]) == 0
    capsys.readouterr()
    assert probabilistic_main([
        "--descriptor", str(descriptor), "--trajectory-index", str(trajectory),
        "--matched", str(matched), "--classical-controls", str(classical),
        "--output", str(probabilistic)
    ]) == 0
    capsys.readouterr()
    assert switching_main([
        "--descriptor", str(descriptor), "--trajectory-index", str(trajectory),
        "--matched", str(matched), "--classical-controls", str(classical),
        "--probabilistic-ssm", str(probabilistic), "--output", str(switching)
    ]) == 0
    capsys.readouterr()
    assert nonlinear_main([
        "--descriptor", str(descriptor), "--trajectory-index", str(trajectory),
        "--matched", str(matched), "--classical-controls", str(classical),
        "--probabilistic-ssm", str(probabilistic), "--switching-state", str(switching),
        "--output", str(nonlinear)
    ]) == 0
    capsys.readouterr()
    return descriptor, trajectory, matched, classical, probabilistic, switching, nonlinear


def test_stability_task_materializes_full_chain_and_rejects_tampering(
    tmp_path: Path, capsys
) -> None:
    descriptor, trajectory, matched, classical, probabilistic, switching, nonlinear = (
        _materialize_v013_chain(tmp_path, capsys)
    )
    output = tmp_path / "bootstrap_stability.json"
    args = [
        "--descriptor", str(descriptor),
        "--trajectory-index", str(trajectory),
        "--matched", str(matched),
        "--classical-controls", str(classical),
        "--probabilistic-ssm", str(probabilistic),
        "--switching-state", str(switching),
        "--nonlinear-control", str(nonlinear),
        "--replicates", "4",
        "--seed", "1401",
        "--minimum-success-fraction", "0.5",
        "--output", str(output),
    ]
    assert stability_main(args) == 0
    stdout = json.loads(capsys.readouterr().out)
    artifact = json.loads(output.read_text())

    assert stdout["status"] == "pass"
    assert artifact["artifact_role"] == "bootstrap_stability_evidence"
    assert artifact["upstream_nonlinear_artifact_verified"] is True
    assert artifact["upstream_nonlinear_artifact_reconstructed"] is True
    assert artifact["execution_complete"] is True
    assert artifact["predictive_adversary_ladder_complete"] is True
    assert artifact["evaluation_resampled"] is False
    assert artifact["single_case_bootstrap_is_icc"] is False
    assert artifact["participant_icc_computed"] is False
    assert artifact["success_count"] + artifact["failure_count"] == 4
    assert artifact["success_count"] >= 1
    assert artifact["bootstrap_coverage_sufficient"] == (
        artifact["success_fraction"] >= artifact["minimum_success_fraction"]
    )
    assert artifact["stability_gate_defined"] is False
    assert artifact["stability_gate_pass"] is None
    assert artifact["intervention_direction_evidence_required"] is True
    assert artifact["intervention_stage_eligible"] is False
    assert artifact["physical_quantum_promotion_eligible"] is False
    assert set(artifact["parameter_summaries"]) == {
        "omega_x", "omega_z", "gamma_dephasing", "gamma_relaxation",
        "canonical_structure_residual",
    }
    assert set(artifact["predictive_summaries"]) == {
        "canonical_minus_affine_one_step_rmse",
        "canonical_minus_affine_rollout_rmse",
        "direct_minus_nonlinear_mean_nll",
        "direct_minus_nonlinear_one_step_rmse",
    }

    payload = json.loads(nonlinear.read_text())
    payload["model"]["innovation_variance"][0] *= 2.0
    tampered = tmp_path / "nonlinear_tampered.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    should_not_exist = tmp_path / "should_not_exist.json"
    tampered_args = list(args)
    tampered_args[tampered_args.index(str(nonlinear))] = str(tampered)
    tampered_args[tampered_args.index(str(output))] = str(should_not_exist)
    assert stability_main(tampered_args) == 2
    error = json.loads(capsys.readouterr().out)
    assert error["status"] == "error"
    assert "independent v0.13 reconstruction" in error["message"]
    assert not should_not_exist.exists()
