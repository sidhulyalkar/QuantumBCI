from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from quantumbci.experiments.classical_controls_task import main as classical_main
from quantumbci.experiments.probabilistic_ssm_task import main as probabilistic_main
from quantumbci.experiments.switching_state_task import main as switching_main
from quantumbci.experiments.tasks import main as experiment_main


def _write_switching_descriptor(tmp_path: Path) -> Path:
    rng = np.random.default_rng(101)
    transitions = np.asarray(
        [
            [
                [0.90, 0.16, 0.00],
                [-0.10, 0.84, 0.06],
                [0.03, -0.07, 0.86],
            ],
            [
                [0.73, -0.20, 0.07],
                [0.18, 0.78, -0.10],
                [-0.04, 0.13, 0.80],
            ],
        ],
        dtype=float,
    )
    intercepts = np.asarray(
        [[0.010, -0.006, 0.004], [-0.012, 0.010, -0.005]],
        dtype=float,
    )
    variances = np.asarray(
        [[0.0012, 0.0010, 0.0009], [0.0010, 0.0011, 0.0012]],
        dtype=float,
    )
    regime_transition = np.asarray([[0.95, 0.05], [0.07, 0.93]], dtype=float)
    n_trajectories = 8
    n_steps = 68

    states: list[np.ndarray] = []
    ids: list[str] = []
    for trajectory in range(n_trajectories):
        state = rng.normal(0.0, 0.16, size=3)
        regime = int(trajectory % 2)
        for step in range(n_steps):
            if step > 0:
                regime = int(rng.choice(2, p=regime_transition[regime]))
                state = (
                    transitions[regime] @ state
                    + intercepts[regime]
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

    fit_indices: list[int] = []
    calibration_indices: list[int] = []
    evaluation_indices: list[int] = []
    for trajectory in range(n_trajectories):
        base = trajectory * n_steps
        fit_indices.extend(range(base, base + 40))
        calibration_indices.extend(range(base + 40, base + 54))
        evaluation_indices.extend(range(base + 54, base + n_steps))

    descriptor = {
        "schema_version": 1,
        "dataset_id": "ci-switching-state",
        "case_id": "ci-switching-state-case",
        "latent_dimension": 3,
        "time_step_policy": "fixed",
        "expected_window_seconds": 0.5,
        "expected_step_seconds": 1.0,
        "step_tolerance_seconds": 1e-12,
        "purge_seconds": 0.0,
        "upstream_authority_fingerprint": "neuros-ci-authority",
        "source_revisions": {"quantumbci": "ci-v012", "encoder": "frozen-ci"},
        "data_metadata": {
            "state_surface": "observed_coordinates",
            "fixture": "switching-state-ci",
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


def _materialize_v011_chain(
    tmp_path: Path,
    capsys,
) -> tuple[Path, Path, Path, Path, Path]:
    descriptor = _write_switching_descriptor(tmp_path)
    trajectory_index = tmp_path / "trajectory_index.json"
    matched = tmp_path / "matched_dynamics.json"
    classical = tmp_path / "classical_controls.json"
    probabilistic = tmp_path / "probabilistic_ssm.json"

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
            str(probabilistic),
        ]
    ) == 0
    capsys.readouterr()
    return descriptor, trajectory_index, matched, classical, probabilistic


def test_switching_task_materializes_complete_authority_bound_chain(
    tmp_path: Path,
    capsys,
) -> None:
    descriptor, trajectory_index, matched, classical, probabilistic = _materialize_v011_chain(
        tmp_path, capsys
    )
    output = tmp_path / "switching_state.json"

    assert switching_main(
        [
            "--descriptor",
            str(descriptor),
            "--trajectory-index",
            str(trajectory_index),
            "--matched",
            str(matched),
            "--classical-controls",
            str(classical),
            "--probabilistic-ssm",
            str(probabilistic),
            "--output",
            str(output),
        ]
    ) == 0
    stdout = json.loads(capsys.readouterr().out)
    artifact = json.loads(output.read_text())

    assert stdout["status"] == "pass"
    assert artifact["artifact_role"] == "switching_state_classical_control"
    assert artifact["upstream_probabilistic_artifact_verified"] is True
    assert artifact["upstream_probabilistic_artifact_reconstructed"] is True
    assert len(artifact["fit_transition_sha256"]) == 64
    assert len(artifact["calibration_transition_sha256"]) == 64
    assert len(artifact["evaluation_transition_sha256"]) == 64
    assert artifact["model"]["regime_count"] == 2
    assert artifact["model"]["parameter_count"] == 33
    assert artifact["model"]["regime_labels_mechanistically_identifiable"] is False
    assert artifact["role_boundary_regime_belief_reset"] is True
    assert artifact["sequential_predictive_density_complete"] is True
    assert artifact["switching_state_control_complete"] is True
    assert artifact["exact_open_loop_switching_forecast_complete"] is False
    assert artifact["open_loop_promotion_eligible"] is False
    assert artifact["flexible_nonlinear_control_required_when_powered"] is True
    assert artifact["bootstrap_stability_required"] is True
    assert artifact["intervention_direction_evidence_required"] is True
    assert artifact["intervention_stage_eligible"] is False
    assert artifact["dynamical_information_novel"] is False
    assert artifact["physical_quantum_promotion_eligible"] is False

    diagnostics = artifact["multistart_diagnostics"]
    assert diagnostics["success_count"] + diagnostics["failure_count"] == 4
    assert diagnostics["success_count"] >= 1
    assert diagnostics["best_initialization_id"] == artifact["model"]["selected_initialization"]
    assert diagnostics["best_fit_log_likelihood"] == artifact["model"]["fit_log_likelihood"]
    assert set(artifact["matched_sequential_comparisons"]) == {
        "direct_gaussian_var_minus_switching_mean_nll",
        "kalman_minus_switching_mean_nll",
        "direct_gaussian_var_minus_switching_rmse",
        "kalman_minus_switching_rmse",
    }

    probabilistic_payload = json.loads(probabilistic.read_text())
    assert artifact["authority_fingerprint"] == probabilistic_payload["authority_fingerprint"]
    assert artifact["data_sha256"] == probabilistic_payload["data_sha256"]
    assert artifact["fit_transition_sha256"] == probabilistic_payload["fit_transition_sha256"]
    assert artifact["calibration_transition_sha256"] == probabilistic_payload[
        "calibration_transition_sha256"
    ]
    assert artifact["evaluation_transition_sha256"] == probabilistic_payload[
        "evaluation_transition_sha256"
    ]


def test_switching_task_rejects_tampered_v011_artifact_before_output(
    tmp_path: Path,
    capsys,
) -> None:
    descriptor, trajectory_index, matched, classical, probabilistic = _materialize_v011_chain(
        tmp_path, capsys
    )
    payload = json.loads(probabilistic.read_text())
    payload["selected_q_scale"] = float(payload["selected_q_scale"]) * 3.0
    probabilistic.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "should_not_exist.json"

    assert switching_main(
        [
            "--descriptor",
            str(descriptor),
            "--trajectory-index",
            str(trajectory_index),
            "--matched",
            str(matched),
            "--classical-controls",
            str(classical),
            "--probabilistic-ssm",
            str(probabilistic),
            "--output",
            str(output),
        ]
    ) == 2
    error = json.loads(capsys.readouterr().out)
    assert error["status"] == "error"
    assert "independent v0.11 reconstruction" in error["message"]
    assert not output.exists()


def test_switching_task_rejects_v011_promotion_ceiling_violation(
    tmp_path: Path,
    capsys,
) -> None:
    descriptor, trajectory_index, matched, classical, probabilistic = _materialize_v011_chain(
        tmp_path, capsys
    )
    payload = json.loads(probabilistic.read_text())
    payload["physical_quantum_promotion_eligible"] = True
    probabilistic.write_text(json.dumps(payload), encoding="utf-8")

    assert switching_main(
        [
            "--descriptor",
            str(descriptor),
            "--trajectory-index",
            str(trajectory_index),
            "--matched",
            str(matched),
            "--classical-controls",
            str(classical),
            "--probabilistic-ssm",
            str(probabilistic),
        ]
    ) == 2
    error = json.loads(capsys.readouterr().out)
    assert error["status"] == "error"
    assert "physical-quantum" in error["message"]
