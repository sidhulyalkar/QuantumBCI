from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from quantumbci.experiments.classical_controls_task import main as classical_main
from quantumbci.experiments.probabilistic_ssm_task import main as probabilistic_main
from quantumbci.experiments.tasks import main as experiment_main


def _write_noisy_descriptor(tmp_path: Path) -> Path:
    rng = np.random.default_rng(17)
    transition = np.asarray(
        [
            [0.86, 0.12, 0.00],
            [-0.08, 0.88, 0.06],
            [0.03, -0.07, 0.84],
        ],
        dtype=float,
    )
    intercept = np.asarray([0.004, -0.003, 0.002], dtype=float)
    n_trajectories = 8
    n_steps = 56
    states: list[np.ndarray] = []
    ids: list[str] = []
    for trajectory in range(n_trajectories):
        latent = rng.normal(0.0, 0.12, size=3)
        for step in range(n_steps):
            if step > 0:
                latent = transition @ latent + intercept + rng.normal(0.0, 0.008, size=3)
            observed = latent + rng.normal(0.0, 0.025, size=3)
            states.append(observed)
            ids.append(f"trajectory-{trajectory}")

    state_array = np.asarray(states, dtype=float)
    starts = np.tile(np.arange(n_steps, dtype=float), n_trajectories)
    stops = starts + 0.5
    np.save(tmp_path / "states.npy", state_array)
    np.save(tmp_path / "ids.npy", np.asarray(ids))
    np.save(tmp_path / "starts.npy", starts)
    np.save(tmp_path / "stops.npy", stops)

    fit_indices: list[int] = []
    calibration_indices: list[int] = []
    evaluation_indices: list[int] = []
    for trajectory in range(n_trajectories):
        base = trajectory * n_steps
        fit_indices.extend(range(base, base + 32))
        calibration_indices.extend(range(base + 32, base + 44))
        evaluation_indices.extend(range(base + 44, base + n_steps))

    descriptor = {
        "schema_version": 1,
        "dataset_id": "ci-probabilistic-ssm",
        "case_id": "ci-probabilistic-case",
        "latent_dimension": 3,
        "time_step_policy": "fixed",
        "expected_window_seconds": 0.5,
        "expected_step_seconds": 1.0,
        "step_tolerance_seconds": 1e-12,
        "purge_seconds": 0.0,
        "upstream_authority_fingerprint": "neuros-ci-authority",
        "source_revisions": {"quantumbci": "ci-head", "encoder": "frozen-ci"},
        "data_metadata": {
            "state_surface": "observed_latent_coordinates",
            "fixture": "probabilistic-ssm-ci",
        },
        "data": {
            "states": "states.npy",
            "trajectory_ids": "ids.npy",
            "start_times_s": "starts.npy",
            "stop_times_s": "stops.npy",
        },
        "split": {
            "fit_indices": fit_indices,
            "calibration_indices": calibration_indices,
            "evaluation_indices": evaluation_indices,
            "representation_fit_indices": fit_indices,
        },
    }
    path = tmp_path / "trajectory_contract.json"
    path.write_text(json.dumps(descriptor), encoding="utf-8")
    return path


def _materialize_upstream(tmp_path: Path, capsys) -> tuple[Path, Path, Path, Path]:
    descriptor = _write_noisy_descriptor(tmp_path)
    trajectory_index = tmp_path / "trajectory_index.json"
    matched = tmp_path / "matched_dynamics.json"
    classical = tmp_path / "classical_controls.json"

    assert experiment_main(
        [
            "trajectory-contract",
            "E002",
            "--input",
            str(descriptor),
            "--output",
            str(trajectory_index),
        ]
    ) == 0
    capsys.readouterr()
    assert experiment_main(
        [
            "fit-matched-dynamics",
            "E002",
            "--input",
            str(descriptor),
            "--trajectory-index",
            str(trajectory_index),
            "--output",
            str(matched),
            "--ridge",
            "0",
        ]
    ) == 0
    capsys.readouterr()
    assert classical_main(
        [
            "--descriptor",
            str(descriptor),
            "--trajectory-index",
            str(trajectory_index),
            "--matched",
            str(matched),
            "--output",
            str(classical),
        ]
    ) == 0
    capsys.readouterr()
    return descriptor, trajectory_index, matched, classical


def test_probabilistic_task_materializes_full_authority_bound_chain(
    tmp_path: Path,
    capsys,
) -> None:
    descriptor, trajectory_index, matched, classical = _materialize_upstream(tmp_path, capsys)
    output = tmp_path / "probabilistic_ssm.json"

    assert probabilistic_main(
        [
            "--descriptor",
            str(descriptor),
            "--trajectory-index",
            str(trajectory_index),
            "--matched",
            str(matched),
            "--classical-controls",
            str(classical),
            "--output",
            str(output),
        ]
    ) == 0
    stdout = json.loads(capsys.readouterr().out)
    artifact = json.loads(output.read_text())

    assert stdout["status"] == "pass"
    assert artifact["artifact_role"] == "probabilistic_latent_state_space_control"
    assert artifact["upstream_trajectory_authority_verified"] is True
    assert artifact["upstream_matched_baseline_verified"] is True
    assert artifact["upstream_classical_controls_verified"] is True
    assert artifact["mean_transition_source"] == "v0.10:controls.full_var1"
    assert len(artifact["mean_model_sha256"]) == 64
    assert artifact["mean_transition_refit"] is False
    assert artifact["observation_matrix_fixed"] is True
    assert artifact["latent_coordinate_gauge_fixed"] is True
    assert artifact["calibration_used"] is True
    assert artifact["evaluation_used_for_hyperparameter_selection"] is False
    assert artifact["role_boundary_filter_reset"] is True
    assert artifact["probabilistic_latent_state_space_control_complete"] is True
    assert artifact["switching_state_control_required"] is True
    assert artifact["flexible_nonlinear_control_required_when_powered"] is True
    assert artifact["bootstrap_stability_required"] is True
    assert artifact["intervention_stage_eligible"] is False
    assert artifact["dynamical_information_novel"] is False
    assert artifact["physical_quantum_promotion_eligible"] is False

    classical_payload = json.loads(classical.read_text())
    full = classical_payload["controls"]["full_var1"]
    assert artifact["transition"] == full["transition"]
    assert artifact["intercept"] == full["intercept"]
    assert artifact["authority_fingerprint"] == full["authority_fingerprint"]
    assert artifact["data_sha256"] == full["data_sha256"]
    assert artifact["fit_transition_sha256"] == full["fit_transition_sha256"]
    assert artifact["evaluation_transition_sha256"] == full["evaluation_transition_sha256"]
    assert len(artifact["calibration_transition_sha256"]) == 64


def test_probabilistic_task_rejects_tampered_v010_mean_model_before_output(
    tmp_path: Path,
    capsys,
) -> None:
    descriptor, trajectory_index, matched, classical = _materialize_upstream(tmp_path, capsys)
    payload = json.loads(classical.read_text())
    payload["controls"]["full_var1"]["transition"][0][0] += 0.1
    classical.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "should_not_exist.json"

    assert probabilistic_main(
        [
            "--descriptor",
            str(descriptor),
            "--trajectory-index",
            str(trajectory_index),
            "--matched",
            str(matched),
            "--classical-controls",
            str(classical),
            "--output",
            str(output),
        ]
    ) == 2
    error = json.loads(capsys.readouterr().out)
    assert error["status"] == "error"
    assert "independent reconstruction" in error["message"]
    assert not output.exists()


def test_probabilistic_task_rejects_upstream_promotion_ceiling_violation(
    tmp_path: Path,
    capsys,
) -> None:
    descriptor, trajectory_index, matched, classical = _materialize_upstream(tmp_path, capsys)
    payload = json.loads(classical.read_text())
    payload["physical_quantum_promotion_eligible"] = True
    classical.write_text(json.dumps(payload), encoding="utf-8")

    assert probabilistic_main(
        [
            "--descriptor",
            str(descriptor),
            "--trajectory-index",
            str(trajectory_index),
            "--matched",
            str(matched),
            "--classical-controls",
            str(classical),
        ]
    ) == 2
    error = json.loads(capsys.readouterr().out)
    assert error["status"] == "error"
    assert "physical-quantum" in error["message"]
