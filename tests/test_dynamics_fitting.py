from __future__ import annotations

import numpy as np
import pytest

from quantumbci.dynamics_fitting import (
    FIT_ESTIMATOR_ID,
    SCORE_INTEGRATOR_ID,
    evaluate_generator,
    run_matched_qubit_dynamics_benchmark,
)
from quantumbci.e002_synthetic import (
    CanonicalQubitParameters,
    simulate_canonical_bloch_trajectories,
)
from quantumbci.trajectory_authority import (
    TrajectoryEvidenceAuthority,
    TrajectoryEvidenceData,
)


def _authority_for_trajectories(
    trajectories: np.ndarray,
    times: np.ndarray,
    *,
    n_fit: int,
    dataset_id: str,
) -> tuple[TrajectoryEvidenceData, TrajectoryEvidenceAuthority]:
    values = np.asarray(trajectories, dtype=float)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("fixture trajectories must have shape (trajectories, times, 3)")
    n_trajectories, n_times, _ = values.shape
    if not 0 < n_fit < n_trajectories:
        raise ValueError("fixture requires both fit and evaluation trajectories")
    step = float(times[1] - times[0])
    states = values.reshape(-1, 3)
    trajectory_ids = np.concatenate(
        [np.repeat(f"trajectory-{index}", n_times) for index in range(n_trajectories)]
    )
    starts = np.tile(np.asarray(times, dtype=float), n_trajectories)
    stops = starts + step / 2.0
    data = TrajectoryEvidenceData(
        dataset_id=dataset_id,
        states=states,
        trajectory_ids=trajectory_ids,
        start_times_s=starts,
        stop_times_s=stops,
        metadata={"fixture": dataset_id, "state_surface": "bloch_coordinates"},
    )
    fit_stop = n_fit * n_times
    fit_indices = np.arange(fit_stop)
    evaluation_indices = np.arange(fit_stop, len(states))
    authority = TrajectoryEvidenceAuthority.from_data(
        data,
        case_id=f"{dataset_id}-case",
        fit_indices=fit_indices,
        calibration_indices=[],
        evaluation_indices=evaluation_indices,
        representation_fit_indices=fit_indices,
        latent_dimension=3,
        expected_window_seconds=step / 2.0,
        expected_step_seconds=step,
        step_tolerance_seconds=1e-10,
        purge_seconds=0.0,
        upstream_authority_fingerprint="test-upstream-authority",
        source_revisions={"quantumbci": "test", "encoder": "frozen-test"},
    )
    return data, authority


def _rk4_affine_trajectories(
    matrix: np.ndarray,
    offset: np.ndarray,
    times: np.ndarray,
    initial_states: np.ndarray,
) -> np.ndarray:
    a = np.asarray(matrix, dtype=float)
    b = np.asarray(offset, dtype=float)
    output = np.empty((len(initial_states), len(times), 3), dtype=float)
    output[:, 0] = initial_states
    for time_index in range(1, len(times)):
        dt = float(times[time_index] - times[time_index - 1])
        x = output[:, time_index - 1]

        def rhs(values: np.ndarray) -> np.ndarray:
            return values @ a.T + b

        k1 = rhs(x)
        k2 = rhs(x + 0.5 * dt * k1)
        k3 = rhs(x + 0.5 * dt * k2)
        k4 = rhs(x + dt * k3)
        output[:, time_index] = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return output


def test_canonical_family_survives_matched_affine_control_without_overclaiming() -> None:
    truth = CanonicalQubitParameters(
        omega_x=0.9,
        omega_z=-0.65,
        gamma_dephasing=0.18,
        gamma_relaxation=0.28,
    )
    times = np.linspace(0.0, 1.2, 121)
    initial = np.asarray(
        [
            [0.55, 0.00, 0.05],
            [0.00, 0.55, -0.05],
            [0.05, 0.00, 0.55],
            [-0.35, 0.25, 0.10],
            [0.30, -0.30, 0.20],
            [-0.20, 0.35, -0.10],
        ]
    )
    trajectories = simulate_canonical_bloch_trajectories(truth, times, initial)
    data, authority = _authority_for_trajectories(
        trajectories,
        times,
        n_fit=4,
        dataset_id="canonical-e002",
    )

    result = run_matched_qubit_dynamics_benchmark(data, authority)

    assert result.same_evidence_verified is True
    assert result.calibration_used is False
    assert result.dynamical_information_novel is False
    assert result.physical_quantum_promotion_eligible is False
    assert result.parameter_reduction == 8
    assert result.affine.parameter_count == 12
    assert result.canonical.parameter_count == 4
    assert result.affine.fit_estimator_id == FIT_ESTIMATOR_ID
    assert result.canonical.fit_estimator_id == FIT_ESTIMATOR_ID
    assert result.affine.score_integrator_id == SCORE_INTEGRATOR_ID
    assert result.canonical.score_integrator_id == SCORE_INTEGRATOR_ID
    assert result.affine.fit_design_rank == 4
    assert result.canonical.fit_design_rank == 4

    for field in (
        "authority_fingerprint",
        "data_sha256",
        "fit_transition_sha256",
        "evaluation_transition_sha256",
    ):
        assert getattr(result.affine, field) == getattr(result.canonical, field)
        assert getattr(result, field) == getattr(result.affine, field)

    assert result.canonical.evaluation_metrics.one_step_rmse < 0.01
    assert result.canonical.evaluation_metrics.rollout_rmse < 0.02
    assert result.canonical.canonical_structure_residual_to_affine is not None
    assert result.canonical.canonical_structure_residual_to_affine < 0.03
    assert result.canonical_minus_affine_one_step_rmse < 0.01
    assert result.canonical_minus_affine_rollout_rmse < 0.02
    assert result.canonical.evaluation_metrics.target_physical_fraction == pytest.approx(1.0)
    assert result.canonical.evaluation_metrics.one_step_prediction_physical_fraction == pytest.approx(
        1.0
    )
    assert result.canonical.evaluation_metrics.one_step_mean_valid_qubit_trace_distance is not None

    recovered = result.canonical.canonical_parameters
    assert recovered is not None
    assert recovered.omega_x == pytest.approx(truth.omega_x, abs=0.03)
    assert recovered.omega_z == pytest.approx(truth.omega_z, abs=0.03)
    assert recovered.gamma_dephasing == pytest.approx(truth.gamma_dephasing, abs=0.03)
    assert recovered.gamma_relaxation == pytest.approx(truth.gamma_relaxation, abs=0.03)


def test_stable_noncanonical_affine_adversary_is_won_by_classical_control() -> None:
    matrix = np.asarray(
        [
            [-0.20, -0.80, 0.30],
            [0.80, -0.60, -1.20],
            [0.10, 1.20, -0.35],
        ]
    )
    assert np.max(np.real(np.linalg.eigvals(matrix))) < 0.0
    offset = np.zeros(3)
    times = np.linspace(0.0, 1.5, 151)
    initial = np.asarray(
        [
            [0.25, 0.00, 0.00],
            [0.00, 0.25, 0.00],
            [0.00, 0.00, 0.25],
            [-0.18, 0.12, 0.05],
            [0.16, -0.12, 0.08],
            [-0.10, 0.18, -0.04],
        ]
    )
    trajectories = _rk4_affine_trajectories(matrix, offset, times, initial)
    assert np.max(np.linalg.norm(trajectories.reshape(-1, 3), axis=1)) < 1.0
    data, authority = _authority_for_trajectories(
        trajectories,
        times,
        n_fit=4,
        dataset_id="noncanonical-affine-adversary",
    )

    result = run_matched_qubit_dynamics_benchmark(data, authority)

    affine_rmse = result.affine.evaluation_metrics.rollout_rmse
    canonical_rmse = result.canonical.evaluation_metrics.rollout_rmse
    assert affine_rmse < 0.005
    assert canonical_rmse > affine_rmse * 3.0
    assert result.canonical_minus_affine_one_step_rmse > 0.0
    assert result.canonical_minus_affine_rollout_rmse > 0.0
    assert result.canonical.canonical_structure_residual_to_affine is not None
    assert result.canonical.canonical_structure_residual_to_affine >= 0.10
    assert result.affine.evaluation_metrics.target_physical_fraction == pytest.approx(1.0)


def test_unphysical_classical_prediction_is_not_called_qubit_trace_distance() -> None:
    truth = CanonicalQubitParameters(0.5, 0.4, 0.15, 0.20)
    times = np.linspace(0.0, 0.4, 41)
    initial = np.asarray(
        [
            [0.40, 0.00, 0.00],
            [0.00, 0.40, 0.00],
            [0.20, -0.20, 0.10],
            [-0.20, 0.20, -0.10],
        ]
    )
    trajectories = simulate_canonical_bloch_trajectories(truth, times, initial)
    data, authority = _authority_for_trajectories(
        trajectories,
        times,
        n_fit=2,
        dataset_id="physicality-metric",
    )

    metrics = evaluate_generator(
        data,
        authority,
        np.zeros((3, 3)),
        np.asarray([500.0, 0.0, 0.0]),
        role="evaluation",
    )

    assert metrics.one_step_mean_bloch_half_l2 is not None
    assert metrics.one_step_prediction_physical_fraction == pytest.approx(0.0)
    assert metrics.rollout_prediction_physical_fraction == pytest.approx(0.0)
    assert metrics.target_physical_fraction == pytest.approx(1.0)
    assert metrics.one_step_valid_qubit_pair_fraction == pytest.approx(0.0)
    assert metrics.rollout_valid_qubit_pair_fraction == pytest.approx(0.0)
    assert metrics.one_step_mean_valid_qubit_trace_distance is None
    assert metrics.rollout_mean_valid_qubit_trace_distance is None


def test_matched_benchmark_rejects_invalid_ridge() -> None:
    truth = CanonicalQubitParameters(0.5, 0.4, 0.15, 0.20)
    times = np.linspace(0.0, 0.2, 21)
    initial = np.asarray(
        [
            [0.3, 0.0, 0.0],
            [0.0, 0.3, 0.0],
            [0.0, 0.0, 0.3],
            [0.2, -0.1, 0.1],
        ]
    )
    trajectories = simulate_canonical_bloch_trajectories(truth, times, initial)
    data, authority = _authority_for_trajectories(
        trajectories,
        times,
        n_fit=2,
        dataset_id="bad-ridge",
    )
    with pytest.raises(ValueError, match="ridge"):
        run_matched_qubit_dynamics_benchmark(data, authority, ridge=-1.0)
