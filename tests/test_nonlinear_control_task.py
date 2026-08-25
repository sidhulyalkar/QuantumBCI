from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from quantumbci.experiments.classical_controls_task import main as classical_main
from quantumbci.experiments.nonlinear_control_task import main as nonlinear_main
from quantumbci.experiments.probabilistic_ssm_task import main as probabilistic_main
from quantumbci.experiments.switching_state_task import main as switching_main
from quantumbci.experiments.tasks import main as experiment_main


def _write_descriptor(tmp_path: Path) -> Path:
    rng = np.random.default_rng(211)
    transitions = np.asarray(
        [
            [[0.88, 0.15, 0.00], [-0.10, 0.83, 0.06], [0.03, -0.06, 0.85]],
            [[0.74, -0.18, 0.06], [0.16, 0.79, -0.09], [-0.04, 0.12, 0.80]],
        ],
        dtype=float,
    )
    intercepts = np.asarray(
        [[0.008, -0.005, 0.003], [-0.010, 0.009, -0.004]], dtype=float
    )
    variances = np.asarray(
        [[0.0010, 0.0010, 0.0009], [0.0009, 0.0011, 0.0010]], dtype=float
    )
    regime_transition = np.asarray([[0.95, 0.05], [0.07, 0.93]], dtype=float)
    n_trajectories = 6
    n_steps = 72
    states: list[np.ndarray] = []
    ids: list[str] = []
    for trajectory in range(n_trajectories):
        state = rng.normal(0.0, 0.18, size=3)
        regime = trajectory % 2
        for step in range(n_steps):
            if step > 0:
                regime = int(rng.choice(2, p=regime_transition[regime]))
                nonlinear = np.asarray(
                    [
                        0.025 * np.sin(3.0 * state[1]),
                        0.020 * np.sin(3.0 * state[2]),
                        0.022 * np.sin(3.0 * state[0]),
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
        "dataset_id": "ci-v013-nonlinear",
        "case_id": "ci-v013-case",
        "latent_dimension": 3,
        "time_step_policy": "fixed",
        "expected_window_seconds": 0.5,
        "expected_step_seconds": 1.0,
        "step_tolerance_seconds": 1e-12,
        "purge_seconds": 0.0,
        "upstream_authority_fingerprint": "neuros-ci-v013",
        "source_revisions": {"quantumbci": "ci-v013", "encoder": "frozen-ci"},
        "data_metadata": {"state_surface": "observed_coordinates", "fixture": "v013"},
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


def _materialize_v012_chain(tmp_path: Path, capsys):
    descriptor = _write_descriptor(tmp_path)
    trajectory = tmp_path / "trajectory_index.json"
    matched = tmp_path / "matched_dynamics.json"
    classical = tmp_path / "classical_controls.json"
    probabilistic = tmp_path / "probabilistic_ssm.json"
    switching = tmp_path / "switching_state.json"

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
    return descriptor, trajectory, matched, classical, probabilistic, switching


def test_nonlinear_task_materializes_v013_authority_chain(tmp_path: Path, capsys) -> None:
    descriptor, trajectory, matched, classical, probabilistic, switching = (
        _materialize_v012_chain(tmp_path, capsys)
    )
    output = tmp_path / "nonlinear_control.json"

    assert nonlinear_main([
        "--descriptor", str(descriptor), "--trajectory-index", str(trajectory),
        "--matched", str(matched), "--classical-controls", str(classical),
        "--probabilistic-ssm", str(probabilistic), "--switching-state", str(switching),
        "--output", str(output)
    ]) == 0
    stdout = json.loads(capsys.readouterr().out)
    artifact = json.loads(output.read_text())

    assert stdout["status"] == "pass"
    assert artifact["artifact_role"] == "flexible_nonlinear_classical_control"
    assert artifact["upstream_switching_artifact_verified"] is True
    assert artifact["upstream_switching_artifact_reconstructed"] is True
    assert artifact["affine_mean_source"] == "v0.10:controls.full_var1"
    assert artifact["affine_mean_refit"] is False
    assert artifact["model"]["affine_mean_refit"] is False
    assert artifact["model"]["rff_seed"] == 1301
    assert artifact["flexible_nonlinear_control_complete"] is True
    assert artifact["bootstrap_stability_required"] is True
    assert artifact["intervention_direction_evidence_required"] is True
    assert artifact["intervention_stage_eligible"] is False
    assert artifact["dynamical_information_novel"] is False
    assert artifact["physical_quantum_promotion_eligible"] is False
    assert artifact["evaluation_used_for_model_selection"] is False
    assert artifact["nonlinear_uncertainty_rollout_complete"] is False
    assert artifact["rollout_likelihood_promotion_eligible"] is False
    assert len(artifact["model"]["model_sha256"]) == 64
    assert set(artifact["matched_information_set_comparisons"]) == {
        "direct_gaussian_var_minus_nonlinear_one_step_mean_nll",
        "direct_gaussian_var_minus_nonlinear_one_step_rmse",
        "full_var_minus_nonlinear_rollout_rmse",
    }
    assert set(artifact["comparison_exclusions"]) == {
        "kalman_sequential", "switching_sequential", "nonlinear_rollout_likelihood"
    }

    switching_payload = json.loads(switching.read_text())
    for field in (
        "authority_fingerprint", "data_sha256", "fit_transition_sha256",
        "calibration_transition_sha256", "evaluation_transition_sha256"
    ):
        assert artifact[field] == switching_payload[field]


def test_nonlinear_task_rejects_tampered_v012_before_output(tmp_path: Path, capsys) -> None:
    descriptor, trajectory, matched, classical, probabilistic, switching = (
        _materialize_v012_chain(tmp_path, capsys)
    )
    payload = json.loads(switching.read_text())
    payload["model"]["regime_transition"][0][0] -= 0.1
    payload["model"]["regime_transition"][0][1] += 0.1
    switching.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "should_not_exist.json"

    assert nonlinear_main([
        "--descriptor", str(descriptor), "--trajectory-index", str(trajectory),
        "--matched", str(matched), "--classical-controls", str(classical),
        "--probabilistic-ssm", str(probabilistic), "--switching-state", str(switching),
        "--output", str(output)
    ]) == 2
    error = json.loads(capsys.readouterr().out)
    assert error["status"] == "error"
    assert "independent v0.12 reconstruction" in error["message"]
    assert not output.exists()
