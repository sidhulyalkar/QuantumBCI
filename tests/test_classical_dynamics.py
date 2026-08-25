from __future__ import annotations

import numpy as np
import pytest

from quantumbci.classical_dynamics import (
    DIRECT_DISCRETE_ESTIMATOR_ID,
    DISCRETE_ROLLOUT_ID,
    evaluate_discrete_transition,
    run_extended_classical_controls,
)
from quantumbci.trajectory_authority import TrajectoryEvidenceAuthority, TrajectoryEvidenceData


def _simulate_discrete(
    transition: np.ndarray,
    intercept: np.ndarray,
    initial_states: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    output = np.empty((len(initial_states), n_steps, transition.shape[0]), dtype=float)
    output[:, 0] = initial_states
    for step in range(1, n_steps):
        output[:, step] = output[:, step - 1] @ transition.T + intercept
    return output


def _authority(
    trajectories: np.ndarray,
    *,
    n_fit: int,
    dataset_id: str,
) -> tuple[TrajectoryEvidenceData, TrajectoryEvidenceAuthority]:
    n_trajectories, n_steps, dimension = trajectories.shape
    states = trajectories.reshape(-1, dimension)
    ids = np.concatenate([np.repeat(f"trajectory-{i}", n_steps) for i in range(n_trajectories)])
    starts = np.tile(np.arange(n_steps, dtype=float), n_trajectories)
    stops = starts + 0.5
    data = TrajectoryEvidenceData(
        dataset_id=dataset_id,
        states=states,
        trajectory_ids=ids,
        start_times_s=starts,
        stop_times_s=stops,
        metadata={"fixture": dataset_id, "state_surface": "observed_latent_coordinates"},
    )
    fit_stop = n_fit * n_steps
    fit_indices = np.arange(fit_stop)
    evaluation_indices = np.arange(fit_stop, len(states))
    authority = TrajectoryEvidenceAuthority.from_data(
        data,
        case_id=f"{dataset_id}-case",
        fit_indices=fit_indices,
        calibration_indices=[],
        evaluation_indices=evaluation_indices,
        representation_fit_indices=fit_indices,
        latent_dimension=dimension,
        expected_window_seconds=0.5,
        expected_step_seconds=1.0,
        step_tolerance_seconds=1e-12,
        purge_seconds=0.0,
        upstream_authority_fingerprint="test-upstream",
        source_revisions={"quantumbci": "test", "encoder": "frozen-test"},
    )
    return data, authority


def test_full_var1_recovers_stable_cross_coupled_discrete_system() -> None:
    transition = np.asarray(
        [
            [0.92, -0.14, 0.06],
            [0.11, 0.88, -0.09],
            [0.03, 0.10, 0.84],
        ],
        dtype=float,
    )
    assert np.max(np.abs(np.linalg.eigvals(transition))) < 1.0
    intercept = np.asarray([0.004, -0.003, 0.002])
    initial = np.asarray(
        [
            [0.35, 0.00, 0.00],
            [0.00, 0.35, 0.00],
            [0.00, 0.00, 0.35],
            [-0.22, 0.14, 0.05],
            [0.18, -0.16, 0.10],
            [-0.12, 0.20, -0.08],
        ]
    )
    trajectories = _simulate_discrete(transition, intercept, initial, 50)
    data, authority = _authority(
        trajectories,
        n_fit=4,
        dataset_id="cross-coupled-var1",
    )

    result = run_extended_classical_controls(data, authority)

    assert result.calibration_used is False
    assert result.full_var1.estimator_id == DIRECT_DISCRETE_ESTIMATOR_ID
    assert result.full_var1.rollout_id == DISCRETE_ROLLOUT_ID
    assert result.persistence.parameter_count == 0
    assert result.diagonal_ar1.parameter_count == 6
    assert result.full_var1.parameter_count == 12
    assert result.diagonal_ar1.effective_parameter_rank == 6
    assert result.full_var1.effective_parameter_rank == 12
    assert result.best_one_step_model == "full_var1_affine"
    assert result.best_rollout_model == "full_var1_affine"
    assert result.full_var1.evaluation_metrics.one_step_rmse < 1e-12
    assert result.full_var1.evaluation_metrics.rollout_rmse < 1e-11
    assert result.diagonal_ar1.evaluation_metrics.rollout_rmse > 1e-3
    assert result.persistence.evaluation_metrics.rollout_rmse > 1e-3
    assert np.asarray(result.full_var1.transition) == pytest.approx(transition, abs=1e-12)
    assert np.asarray(result.full_var1.intercept) == pytest.approx(intercept, abs=1e-12)

    for lane in (result.persistence, result.diagonal_ar1, result.full_var1):
        assert lane.authority_fingerprint == result.authority_fingerprint
        assert lane.data_sha256 == result.data_sha256
        assert lane.fit_transition_sha256 == result.fit_transition_sha256
        assert lane.evaluation_transition_sha256 == result.evaluation_transition_sha256


def test_model_aliases_are_not_double_counted_as_separate_controls() -> None:
    transition = np.asarray(
        [
            [0.90, -0.08, 0.02],
            [0.08, 0.90, -0.04],
            [0.00, 0.04, 0.86],
        ]
    )
    trajectories = _simulate_discrete(
        transition,
        np.zeros(3),
        np.asarray(
            [
                [0.2, 0.0, 0.0],
                [0.0, 0.2, 0.0],
                [0.0, 0.0, 0.2],
                [-0.1, 0.1, 0.05],
            ]
        ),
        20,
    )
    data, authority = _authority(trajectories, n_fit=2, dataset_id="alias-contract")
    payload = run_extended_classical_controls(data, authority).to_mapping()
    notes = payload["equivalence_notes"]
    assert notes["aliases_count_as_one_model_class"] is True
    assert set(notes["full_var1_aliases"]) == {
        "direct discrete affine transition",
        "VAR(1) with intercept",
        "fully observed one-step discrete LDS mean with identity observation",
    }
    assert notes["kalman_forecast_mean_distinct_under_current_contract"] is False
    assert set(payload["controls"]) == {"persistence", "diagonal_ar1", "full_var1"}


def test_discrete_scoring_keeps_unphysical_predictions_out_of_trace_distance() -> None:
    transition = np.eye(3) * 0.9
    trajectories = _simulate_discrete(
        transition,
        np.zeros(3),
        np.asarray(
            [
                [0.3, 0.0, 0.0],
                [0.0, 0.3, 0.0],
                [0.2, -0.1, 0.1],
                [-0.1, 0.2, -0.1],
            ]
        ),
        12,
    )
    data, authority = _authority(trajectories, n_fit=2, dataset_id="discrete-physicality")
    metrics = evaluate_discrete_transition(
        data,
        authority,
        np.zeros((3, 3)),
        np.asarray([10.0, 0.0, 0.0]),
        role="evaluation",
    )
    assert metrics.one_step_mean_bloch_half_l2 is not None
    assert metrics.one_step_prediction_physical_fraction == pytest.approx(0.0)
    assert metrics.one_step_valid_qubit_pair_fraction == pytest.approx(0.0)
    assert metrics.one_step_mean_valid_qubit_trace_distance is None


def test_classical_controls_are_deterministic() -> None:
    transition = np.asarray(
        [
            [0.91, -0.05, 0.00],
            [0.05, 0.89, -0.03],
            [0.00, 0.03, 0.87],
        ]
    )
    trajectories = _simulate_discrete(
        transition,
        np.asarray([0.001, -0.002, 0.001]),
        np.asarray(
            [
                [0.25, 0.0, 0.0],
                [0.0, 0.25, 0.0],
                [0.0, 0.0, 0.25],
                [-0.15, 0.10, 0.05],
            ]
        ),
        25,
    )
    data, authority = _authority(trajectories, n_fit=2, dataset_id="deterministic-controls")
    first = run_extended_classical_controls(data, authority).to_mapping()
    second = run_extended_classical_controls(data, authority).to_mapping()
    assert first == second
