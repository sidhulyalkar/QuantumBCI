from __future__ import annotations

import numpy as np
import pytest

from quantumbci.probabilistic_ssm import (
    DIRECT_GAUSSIAN_BASELINE_ID,
    NOISE_CALIBRATION_OBJECTIVE,
    PROBABILISTIC_MODEL_ID,
    Q_SCALE_GRID,
    R_SCALE_GRID,
    run_probabilistic_state_space_control,
    score_identity_observation_kalman,
)
from quantumbci.trajectory_authority import TrajectoryEvidenceAuthority, TrajectoryEvidenceData


def _noisy_latent_fixture(
    *,
    seed: int = 7,
    evaluation_offset: np.ndarray | None = None,
) -> tuple[
    TrajectoryEvidenceData,
    TrajectoryEvidenceAuthority,
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(seed)
    transition = np.asarray(
        [
            [0.85, 0.18, 0.00],
            [-0.12, 0.88, 0.08],
            [0.04, -0.10, 0.82],
        ],
        dtype=float,
    )
    intercept = np.asarray([0.02, -0.015, 0.01], dtype=float)
    process_variance = np.asarray([0.0025, 0.0016, 0.0010], dtype=float)
    measurement_variance = np.asarray([0.025, 0.016, 0.020], dtype=float)
    n_trajectories = 12
    n_steps = 80

    observations: list[np.ndarray] = []
    trajectory_ids: list[str] = []
    for trajectory in range(n_trajectories):
        latent = rng.normal(0.0, 0.2, size=3)
        for step in range(n_steps):
            if step > 0:
                latent = (
                    transition @ latent
                    + intercept
                    + rng.normal(0.0, np.sqrt(process_variance))
                )
            observation = latent + rng.normal(0.0, np.sqrt(measurement_variance))
            observations.append(observation)
            trajectory_ids.append(f"trajectory-{trajectory}")

    states = np.asarray(observations, dtype=float)
    fit_indices: list[int] = []
    calibration_indices: list[int] = []
    evaluation_indices: list[int] = []
    for trajectory in range(n_trajectories):
        base = trajectory * n_steps
        fit_indices.extend(range(base, base + 45))
        calibration_indices.extend(range(base + 45, base + 60))
        evaluation_indices.extend(range(base + 60, base + n_steps))

    if evaluation_offset is not None:
        offset = np.asarray(evaluation_offset, dtype=float).reshape(3)
        states[np.asarray(evaluation_indices, dtype=int)] += offset

    starts = np.tile(np.arange(n_steps, dtype=float), n_trajectories)
    stops = starts + 0.5
    data = TrajectoryEvidenceData(
        dataset_id="noisy-latent-kalman",
        states=states,
        trajectory_ids=np.asarray(trajectory_ids),
        start_times_s=starts,
        stop_times_s=stops,
        metadata={
            "fixture": "measurement-noise-adversary",
            "observation_semantics": "latent-plus-measurement-noise",
        },
    )
    authority = TrajectoryEvidenceAuthority.from_data(
        data,
        case_id="noisy-latent-kalman-case",
        fit_indices=fit_indices,
        calibration_indices=calibration_indices,
        evaluation_indices=evaluation_indices,
        representation_fit_indices=fit_indices,
        latent_dimension=3,
        expected_window_seconds=0.5,
        expected_step_seconds=1.0,
        step_tolerance_seconds=1e-12,
        purge_seconds=0.0,
        upstream_authority_fingerprint="test-upstream",
        source_revisions={"quantumbci": "test", "encoder": "frozen-test"},
    )
    return data, authority, transition, intercept


def test_calibrated_kalman_improves_measurement_noise_adversary_without_refitting_mean() -> None:
    data, authority, transition, intercept = _noisy_latent_fixture()
    result = run_probabilistic_state_space_control(
        data,
        authority,
        transition,
        intercept,
    )
    payload = result.to_mapping()

    assert payload["model_id"] == PROBABILISTIC_MODEL_ID
    assert payload["matched_direct_baseline_id"] == DIRECT_GAUSSIAN_BASELINE_ID
    assert payload["noise_calibration_objective"] == NOISE_CALIBRATION_OBJECTIVE
    assert payload["mean_transition_refit"] is False
    assert payload["observation_matrix_fixed"] is True
    assert payload["latent_coordinate_gauge_fixed"] is True
    assert payload["role_boundary_filter_reset"] is True
    assert payload["evaluation_used_for_hyperparameter_selection"] is False
    assert payload["q_scale_grid"] == list(Q_SCALE_GRID)
    assert payload["r_scale_grid"] == list(R_SCALE_GRID)
    assert len(payload["calibration_candidates"]) == len(Q_SCALE_GRID) * len(R_SCALE_GRID)
    assert np.asarray(payload["transition"]) == pytest.approx(transition)
    assert np.asarray(payload["intercept"]) == pytest.approx(intercept)

    comparison = payload["evaluation_comparison"]
    assert comparison["direct_minus_kalman_sequential_mean_nll"] > 0.05
    assert comparison["direct_minus_kalman_sequential_rmse"] > 0.005
    kalman = payload["identity_observation_kalman"]["evaluation_sequential"]
    direct = payload["direct_gaussian_var"]["evaluation_sequential"]
    assert kalman["mean_nll"] < direct["mean_nll"]
    assert kalman["predictive_mean_rmse"] < direct["predictive_mean_rmse"]
    assert 0.80 <= kalman["marginal_95_coverage"] <= 1.0
    assert len(payload["calibration_transition_sha256"]) == 64


def test_evaluation_values_cannot_change_noise_hyperparameter_selection() -> None:
    original_data, original_authority, transition, intercept = _noisy_latent_fixture()
    shifted_data, shifted_authority, _, _ = _noisy_latent_fixture(
        evaluation_offset=np.asarray([3.0, -2.0, 1.5])
    )

    original = run_probabilistic_state_space_control(
        original_data,
        original_authority,
        transition,
        intercept,
    )
    shifted = run_probabilistic_state_space_control(
        shifted_data,
        shifted_authority,
        transition,
        intercept,
    )

    assert shifted_data.data_sha256 != original_data.data_sha256
    assert shifted.selected_q_scale == original.selected_q_scale
    assert shifted.selected_r_scale == original.selected_r_scale
    assert shifted.base_innovation_variance == pytest.approx(original.base_innovation_variance)
    assert [candidate.to_mapping() for candidate in shifted.calibration_candidates] == [
        candidate.to_mapping() for candidate in original.calibration_candidates
    ]
    assert shifted.kalman_evaluation_sequential.mean_nll != pytest.approx(
        original.kalman_evaluation_sequential.mean_nll
    )


def test_probabilistic_control_fails_closed_without_calibration_transitions() -> None:
    data, authority, transition, intercept = _noisy_latent_fixture()
    no_calibration = TrajectoryEvidenceAuthority.from_data(
        data,
        case_id="no-calibration-case",
        fit_indices=authority.fit_indices,
        calibration_indices=[],
        evaluation_indices=tuple(
            sorted(set(authority.calibration_indices) | set(authority.evaluation_indices))
        ),
        representation_fit_indices=authority.representation_fit_indices,
        latent_dimension=3,
        expected_window_seconds=authority.expected_window_seconds,
        expected_step_seconds=authority.expected_step_seconds,
        step_tolerance_seconds=authority.step_tolerance_seconds,
        purge_seconds=0.0,
        upstream_authority_fingerprint="test-upstream",
        source_revisions={"quantumbci": "test", "encoder": "frozen-test"},
    )
    with pytest.raises(ValueError, match="requires calibration transitions"):
        run_probabilistic_state_space_control(
            data,
            no_calibration,
            transition,
            intercept,
        )


def test_unphysical_kalman_means_are_not_labeled_trace_distance() -> None:
    data, authority, _, _ = _noisy_latent_fixture()
    metrics = score_identity_observation_kalman(
        data,
        authority,
        np.zeros((3, 3)),
        np.asarray([10.0, 0.0, 0.0]),
        np.asarray([0.1, 0.1, 0.1]),
        np.asarray([0.1, 0.1, 0.1]),
        role="evaluation",
        open_loop=False,
    )
    assert metrics.mean_bloch_half_l2 is not None
    assert metrics.prediction_physical_fraction == pytest.approx(0.0)
    assert metrics.valid_qubit_pair_fraction == pytest.approx(0.0)
    assert metrics.mean_valid_qubit_trace_distance is None
