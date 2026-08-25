from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from quantumbci.e002_synthetic import CanonicalQubitParameters, simulate_canonical_bloch_trajectories
from quantumbci.experiments.tasks import main


def test_e001_equivalence_stage_materializes_a_real_gate(tmp_path: Path, capsys) -> None:
    output = tmp_path / "equivalence_audit.json"
    assert main(
        [
            "equivalence-audit",
            "E001",
            "density-covariance",
            "--output",
            str(output),
        ]
    ) == 0
    stdout = json.loads(capsys.readouterr().out)
    artifact = json.loads(output.read_text())
    assert stdout["status"] == "pass"
    assert artifact["equivalent_within_tolerance"] is True
    assert artifact["representation_information_novel"] is False
    assert artifact["equivalence_class"] == "trace_normalized_hermitian_second_moment"
    assert {probe["probe"] for probe in artifact["probes"]} == {"real", "complex"}
    assert {probe["center"] for probe in artifact["probes"]} == {True, False}


def test_e002_synthetic_and_identifiability_stages_are_executable(
    tmp_path: Path,
    capsys,
) -> None:
    recovery = tmp_path / "synthetic_recovery.json"
    gate = tmp_path / "identifiability_gate.json"
    assert main(
        [
            "synthetic-recovery",
            "E002",
            "--seed",
            "2027",
            "--noise-std",
            "0.003",
            "--output",
            str(recovery),
        ]
    ) == 0
    recovery_stdout = json.loads(capsys.readouterr().out)
    recovery_payload = json.loads(recovery.read_text())
    assert recovery_stdout["status"] == "pass"
    assert recovery_payload["synthetic_identifiability_gate_pass"] is True
    assert recovery_payload["affine_equivalence_pass"] is True
    assert recovery_payload["gauge_nonidentifiability_witness_pass"] is True
    assert recovery_payload["canonical_structure_pass"] is True
    assert recovery_payload["max_canonical_structure_residual"] <= 0.05
    assert recovery_payload["classical_adversary"]["rejected_as_canonical_family"] is True
    assert recovery_payload["classical_adversary"]["canonical_structure_residual"] >= 0.10
    assert recovery_payload["dynamical_information_novel"] is False

    assert main(
        [
            "gate",
            "E002",
            "identifiability",
            "--input",
            str(recovery),
            "--output",
            str(gate),
        ]
    ) == 0
    gate_stdout = json.loads(capsys.readouterr().out)
    gate_payload = json.loads(gate.read_text())
    assert gate_stdout["status"] == "pass"
    assert gate_payload["trajectory_contract_stage_eligible"] is True
    assert gate_payload["dynamical_information_novel"] is False
    assert gate_payload["physical_quantum_promotion_eligible"] is False
    assert gate_payload["observed"]["median_normalized_recovery_error"] <= 0.20
    assert gate_payload["observed"]["max_canonical_structure_residual"] <= 0.05
    assert gate_payload["observed"]["classical_adversary_structure_residual"] >= 0.10
    assert gate_payload["observed"]["classical_adversary_rejected"] is True


def test_e002_gate_independently_rejects_missing_family_specificity(
    tmp_path: Path,
    capsys,
) -> None:
    recovery = tmp_path / "synthetic_recovery.json"
    gate = tmp_path / "identifiability_gate.json"
    assert main(["synthetic-recovery", "E002", "--output", str(recovery)]) == 0
    capsys.readouterr()
    payload = json.loads(recovery.read_text())
    payload["classical_adversary"]["canonical_structure_residual"] = 0.01
    payload["classical_adversary"]["rejected_as_canonical_family"] = False
    recovery.write_text(json.dumps(payload), encoding="utf-8")
    assert main(
        [
            "gate",
            "E002",
            "identifiability",
            "--input",
            str(recovery),
            "--output",
            str(gate),
        ]
    ) == 2
    stdout = json.loads(capsys.readouterr().out)
    artifact = json.loads(gate.read_text())
    assert stdout["status"] == "fail"
    assert artifact["trajectory_contract_stage_eligible"] is False
    assert artifact["observed"]["classical_adversary_rejected"] is False


def _trajectory_descriptor(tmp_path: Path) -> Path:
    states = np.asarray([[0.1 * i, np.sin(i), np.cos(i)] for i in range(12)], dtype=float)
    ids = np.asarray(["source"] * 6 + ["target"] * 6)
    starts = np.asarray(list(range(6)) + list(range(6)), dtype=float)
    stops = starts + 1.0
    np.save(tmp_path / "states.npy", states)
    np.save(tmp_path / "ids.npy", ids)
    np.save(tmp_path / "starts.npy", starts)
    np.save(tmp_path / "stops.npy", stops)
    payload = {
        "schema_version": 1,
        "dataset_id": "synthetic-continuous",
        "case_id": "ci-trajectory-case",
        "latent_dimension": 3,
        "time_step_policy": "fixed",
        "expected_window_seconds": 1.0,
        "expected_step_seconds": 1.0,
        "purge_seconds": 1.0,
        "upstream_authority_fingerprint": "neuros-ci-authority",
        "source_revisions": {"quantumbci": "ci", "neuros": "ci"},
        "data": {
            "states": "states.npy",
            "trajectory_ids": "ids.npy",
            "start_times_s": "starts.npy",
            "stop_times_s": "stops.npy",
        },
        "split": {
            "fit_indices": list(range(6)),
            "calibration_indices": [6, 7],
            "evaluation_indices": [9, 10, 11],
            "representation_fit_indices": [0, 1, 2, 3],
        },
    }
    descriptor = tmp_path / "trajectory_contract.json"
    descriptor.write_text(json.dumps(payload), encoding="utf-8")
    return descriptor


def _matched_trajectory_descriptor(tmp_path: Path) -> Path:
    parameters = CanonicalQubitParameters(0.7, -0.5, 0.16, 0.24)
    times = np.linspace(0.0, 0.6, 61)
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
    np.save(tmp_path / "matched_states.npy", states)
    np.save(tmp_path / "matched_ids.npy", ids)
    np.save(tmp_path / "matched_starts.npy", starts)
    np.save(tmp_path / "matched_stops.npy", stops)
    fit_stop = 4 * n_times
    payload = {
        "schema_version": 1,
        "dataset_id": "matched-canonical-continuous",
        "case_id": "ci-matched-dynamics-case",
        "latent_dimension": 3,
        "time_step_policy": "fixed",
        "expected_window_seconds": step / 2.0,
        "expected_step_seconds": step,
        "step_tolerance_seconds": 1e-10,
        "purge_seconds": 0.0,
        "upstream_authority_fingerprint": "neuros-ci-authority",
        "source_revisions": {"quantumbci": "ci", "encoder": "frozen-ci"},
        "data_metadata": {"state_surface": "bloch_coordinates", "fixture": "canonical"},
        "data": {
            "states": "matched_states.npy",
            "trajectory_ids": "matched_ids.npy",
            "start_times_s": "matched_starts.npy",
            "stop_times_s": "matched_stops.npy",
        },
        "split": {
            "fit_indices": list(range(fit_stop)),
            "calibration_indices": [],
            "evaluation_indices": list(range(fit_stop, len(states))),
            "representation_fit_indices": list(range(fit_stop)),
        },
    }
    descriptor = tmp_path / "matched_trajectory_contract.json"
    descriptor.write_text(json.dumps(payload), encoding="utf-8")
    return descriptor


def test_e002_trajectory_contract_stage_materializes_frozen_authority(
    tmp_path: Path,
    capsys,
) -> None:
    descriptor = _trajectory_descriptor(tmp_path)
    output = tmp_path / "trajectory_index.json"
    assert main(
        [
            "trajectory-contract",
            "E002",
            "--input",
            str(descriptor),
            "--output",
            str(output),
        ]
    ) == 0
    stdout = json.loads(capsys.readouterr().out)
    artifact = json.loads(output.read_text())
    assert stdout["status"] == "pass"
    assert artifact["artifact_role"] == "trajectory_evidence_authority"
    assert artifact["authority"]["transition_counts"] == {
        "fit": 5,
        "calibration": 1,
        "evaluation": 2,
    }
    assert artifact["authority"]["representation_fit_indices"] == [0, 1, 2, 3]
    assert artifact["authority"]["expected_window_seconds"] == 1.0
    assert artifact["authority"]["purge_seconds"] == 1.0
    assert artifact["shared_tensor_contract"]["required_for_all_model_lanes"] is True
    assert len(artifact["shared_tensor_contract"]["data_sha256"]) == 64
    assert len(artifact["authority"]["authority_fingerprint"]) == 16


def test_e002_matched_fit_stage_reverifies_trajectory_authority(
    tmp_path: Path,
    capsys,
) -> None:
    descriptor = _matched_trajectory_descriptor(tmp_path)
    trajectory_index = tmp_path / "trajectory_index.json"
    matched = tmp_path / "matched_dynamics.json"

    assert main(
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
    assert main(
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
    stdout = json.loads(capsys.readouterr().out)
    artifact = json.loads(matched.read_text())
    assert stdout["status"] == "pass"
    assert artifact["artifact_role"] == "matched_dynamics_baseline"
    assert artifact["same_evidence_verified"] is True
    assert artifact["calibration_used"] is False
    assert artifact["parameter_reduction"] == 8
    assert artifact["authoritative_ridge"] == 0.0
    assert artifact["regularization_geometry_matched"] is True
    ranks = artifact["fit_rank_diagnostics"]
    assert ranks["affine_predictor_design_rank"] == 4
    assert ranks["affine_parameter_rank"] == 12
    assert ranks["affine_parameter_count"] == 12
    assert ranks["canonical_design_rank"] == 4
    assert ranks["canonical_parameter_rank"] == 4
    assert ranks["canonical_parameter_count"] == 4
    assert artifact["affine"]["authority_fingerprint"] == artifact["canonical"][
        "authority_fingerprint"
    ]
    assert artifact["affine"]["data_sha256"] == artifact["canonical"]["data_sha256"]
    assert artifact["affine"]["fit_transition_sha256"] == artifact["canonical"][
        "fit_transition_sha256"
    ]
    assert artifact["affine"]["evaluation_transition_sha256"] == artifact["canonical"][
        "evaluation_transition_sha256"
    ]
    assert artifact["canonical"]["evaluation_metrics"][
        "one_step_mean_valid_qubit_trace_distance"
    ] is not None
    assert artifact["dynamical_information_novel"] is False
    assert artifact["physical_quantum_promotion_eligible"] is False
    assert artifact["extended_classical_controls_required"] is True
    assert artifact["intervention_stage_eligible"] is False


def test_e002_matched_fit_rejects_nonzero_authoritative_ridge(
    tmp_path: Path,
    capsys,
) -> None:
    descriptor = _matched_trajectory_descriptor(tmp_path)
    trajectory_index = tmp_path / "trajectory_index.json"
    output = tmp_path / "must_not_exist.json"
    assert main(
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

    assert main(
        [
            "fit-matched-dynamics",
            "E002",
            "--input",
            str(descriptor),
            "--trajectory-index",
            str(trajectory_index),
            "--output",
            str(output),
            "--ridge",
            "0.001",
        ]
    ) == 2
    error = json.loads(capsys.readouterr().out)
    assert error["status"] == "error"
    assert "requires ridge=0" in error["message"]
    assert not output.exists()


def test_e002_matched_fit_rejects_tampered_trajectory_index(
    tmp_path: Path,
    capsys,
) -> None:
    descriptor = _matched_trajectory_descriptor(tmp_path)
    trajectory_index = tmp_path / "trajectory_index.json"
    assert main(
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
    payload = json.loads(trajectory_index.read_text())
    payload["authority"]["source_revisions"]["encoder"] = "tampered"
    trajectory_index.write_text(json.dumps(payload), encoding="utf-8")

    assert main(
        [
            "fit-matched-dynamics",
            "E002",
            "--input",
            str(descriptor),
            "--trajectory-index",
            str(trajectory_index),
        ]
    ) == 2
    error = json.loads(capsys.readouterr().out)
    assert error["status"] == "error"
    assert "differs" in error["message"]


def test_e002_identifiability_gate_fails_closed_without_recovery_artifact(
    tmp_path: Path,
    capsys,
) -> None:
    assert main(
        [
            "gate",
            "E002",
            "identifiability",
            "--input",
            str(tmp_path / "missing.json"),
        ]
    ) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert "not found" in payload["message"]


def test_unimplemented_manifest_tasks_still_fail_closed(capsys) -> None:
    assert main(["extract-embeddings", "E001"]) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "not_implemented"

    assert main(["fit-dynamics-controls", "E002"]) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "not_implemented"

    assert main(["dynamics-interventions", "E002"]) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "not_implemented"
