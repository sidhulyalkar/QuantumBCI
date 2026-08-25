from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from quantumbci.e002_synthetic import CanonicalQubitParameters, simulate_canonical_bloch_trajectories
from quantumbci.experiments.classical_controls_task import main as controls_main
from quantumbci.experiments.tasks import main as experiment_main


def _descriptor(tmp_path: Path) -> Path:
    parameters = CanonicalQubitParameters(0.75, -0.55, 0.17, 0.26)
    times = np.linspace(0.0, 0.8, 81)
    initial = np.asarray(
        [
            [0.50, 0.00, 0.00],
            [0.00, 0.50, 0.00],
            [0.00, 0.00, 0.50],
            [-0.30, 0.20, 0.10],
            [0.25, -0.25, 0.15],
            [-0.15, 0.30, -0.10],
        ]
    )
    trajectories = simulate_canonical_bloch_trajectories(parameters, times, initial)
    n_times = len(times)
    states = trajectories.reshape(-1, 3)
    ids = np.concatenate([np.repeat(f"trajectory-{i}", n_times) for i in range(len(initial))])
    starts = np.tile(times, len(initial))
    step = float(times[1] - times[0])
    stops = starts + step / 2.0
    np.save(tmp_path / "states.npy", states)
    np.save(tmp_path / "ids.npy", ids)
    np.save(tmp_path / "starts.npy", starts)
    np.save(tmp_path / "stops.npy", stops)
    fit_stop = 4 * n_times
    payload = {
        "schema_version": 1,
        "dataset_id": "v010-canonical-control-fixture",
        "case_id": "v010-control-case",
        "latent_dimension": 3,
        "time_step_policy": "fixed",
        "expected_window_seconds": step / 2.0,
        "expected_step_seconds": step,
        "step_tolerance_seconds": 1e-10,
        "purge_seconds": 0.0,
        "upstream_authority_fingerprint": "neuros-v010-test",
        "source_revisions": {"quantumbci": "v010-test", "encoder": "frozen-v010"},
        "data_metadata": {"state_surface": "bloch_coordinates", "fixture": "canonical"},
        "data": {
            "states": "states.npy",
            "trajectory_ids": "ids.npy",
            "start_times_s": "starts.npy",
            "stop_times_s": "stops.npy",
        },
        "split": {
            "fit_indices": list(range(fit_stop)),
            "calibration_indices": [],
            "evaluation_indices": list(range(fit_stop, len(states))),
            "representation_fit_indices": list(range(fit_stop)),
        },
    }
    path = tmp_path / "trajectory_contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _build_upstream(tmp_path: Path, capsys) -> tuple[Path, Path, Path]:
    descriptor = _descriptor(tmp_path)
    trajectory_index = tmp_path / "trajectory_index.json"
    matched = tmp_path / "matched_dynamics.json"
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
    return descriptor, trajectory_index, matched


def test_classical_controls_task_binds_v08_and_v09_evidence(tmp_path: Path, capsys) -> None:
    descriptor, trajectory_index, matched = _build_upstream(tmp_path, capsys)
    output = tmp_path / "classical_controls.json"
    assert controls_main(
        [
            "--descriptor",
            str(descriptor),
            "--trajectory-index",
            str(trajectory_index),
            "--matched",
            str(matched),
            "--output",
            str(output),
        ]
    ) == 0
    stdout = json.loads(capsys.readouterr().out)
    artifact = json.loads(output.read_text())

    assert stdout["status"] == "pass"
    assert artifact["artifact_role"] == "extended_classical_dynamics_controls"
    assert artifact["upstream_matched_baseline_verified"] is True
    assert artifact["calibration_used"] is False
    assert set(artifact["controls"]) == {"persistence", "diagonal_ar1", "full_var1"}
    assert artifact["controls"]["persistence"]["parameter_count"] == 0
    assert artifact["controls"]["diagonal_ar1"]["parameter_count"] == 6
    assert artifact["controls"]["full_var1"]["parameter_count"] == 12
    assert artifact["controls"]["full_var1"]["effective_parameter_rank"] == 12
    assert artifact["best_one_step_model"] == "full_var1_affine"
    assert artifact["best_rollout_model"] == "full_var1_affine"
    assert artifact["controls"]["full_var1"]["evaluation_metrics"]["one_step_rmse"] < 1e-9
    assert artifact["controls"]["full_var1"]["evaluation_metrics"]["rollout_rmse"] < 1e-8
    assert artifact["comparisons_to_v0_9"]["canonical_minus_full_var1_one_step_rmse"] > 0
    assert artifact["comparisons_to_v0_9"]["canonical_minus_full_var1_rollout_rmse"] > 0
    assert artifact["linear_observed_control_stage_complete"] is True
    assert artifact["probabilistic_latent_state_space_control_required"] is True
    assert artifact["switching_state_control_required"] is True
    assert artifact["flexible_nonlinear_control_required_when_powered"] is True
    assert artifact["intervention_stage_eligible"] is False
    assert artifact["physical_quantum_promotion_eligible"] is False

    matched_payload = json.loads(matched.read_text())
    for field in (
        "authority_fingerprint",
        "data_sha256",
        "fit_transition_sha256",
        "evaluation_transition_sha256",
    ):
        assert artifact[field] == matched_payload[field]
        for control in artifact["controls"].values():
            assert control[field] == matched_payload[field]

    aliases = artifact["equivalence_notes"]
    assert aliases["aliases_count_as_one_model_class"] is True
    assert aliases["kalman_forecast_mean_distinct_under_current_contract"] is False
    assert "/tmp/" not in json.dumps(artifact)


def test_classical_controls_task_rejects_tampered_matched_baseline(
    tmp_path: Path,
    capsys,
) -> None:
    descriptor, trajectory_index, matched = _build_upstream(tmp_path, capsys)
    payload = json.loads(matched.read_text())
    payload["canonical"]["data_sha256"] = "0" * 64
    matched.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "must_not_exist.json"

    assert controls_main(
        [
            "--descriptor",
            str(descriptor),
            "--trajectory-index",
            str(trajectory_index),
            "--matched",
            str(matched),
            "--output",
            str(output),
        ]
    ) == 2
    error = json.loads(capsys.readouterr().out)
    assert error["status"] == "error"
    assert "canonical.data_sha256" in error["message"]
    assert not output.exists()


def test_classical_controls_task_rejects_noncanonical_upstream_claims(
    tmp_path: Path,
    capsys,
) -> None:
    descriptor, trajectory_index, matched = _build_upstream(tmp_path, capsys)
    payload = json.loads(matched.read_text())
    payload["physical_quantum_promotion_eligible"] = True
    matched.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "must_not_exist.json"

    assert controls_main(
        [
            "--descriptor",
            str(descriptor),
            "--trajectory-index",
            str(trajectory_index),
            "--matched",
            str(matched),
            "--output",
            str(output),
        ]
    ) == 2
    error = json.loads(capsys.readouterr().out)
    assert error["status"] == "error"
    assert "physical-quantum" in error["message"]
    assert not output.exists()
