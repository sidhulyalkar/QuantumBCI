from __future__ import annotations

import numpy as np
import pytest

from quantumbci.classical_dynamics import fit_full_affine_var1
from quantumbci.nonlinear_dynamics import (
    FEATURE_COUNTS,
    LENGTH_SCALE_MULTIPLIERS,
    NONLINEAR_MODEL_ID,
    NONLINEAR_SCORE_ID,
    RIDGES,
    RFF_SEED,
    NonlinearResidualModel,
    evaluate_nonlinear_model,
    run_nonlinear_residual_control,
)
from quantumbci.probabilistic_ssm import (
    fit_base_innovation_variance,
    score_direct_gaussian_var,
)
from quantumbci.trajectory_authority import TrajectoryEvidenceAuthority, TrajectoryEvidenceData


def _authority(
    trajectories: np.ndarray,
    *,
    dataset_id: str,
    fit_steps: int = 72,
    calibration_steps: int = 18,
) -> tuple[TrajectoryEvidenceData, TrajectoryEvidenceAuthority]:
    values = np.asarray(trajectories, dtype=float)
    n_trajectories, n_steps, dimension = values.shape
    states = values.reshape(-1, dimension)
    ids = np.concatenate(
        [np.repeat(f"trajectory-{index}", n_steps) for index in range(n_trajectories)]
    )
    starts = np.tile(np.arange(n_steps, dtype=float), n_trajectories)
    stops = starts + 0.5
    fit: list[int] = []
    calibration: list[int] = []
    evaluation: list[int] = []
    for trajectory in range(n_trajectories):
        base = trajectory * n_steps
        fit.extend(range(base, base + fit_steps))
        calibration.extend(range(base + fit_steps, base + fit_steps + calibration_steps))
        evaluation.extend(range(base + fit_steps + calibration_steps, base + n_steps))

    data = TrajectoryEvidenceData(
        dataset_id=dataset_id,
        states=states,
        trajectory_ids=ids,
        start_times_s=starts,
        stop_times_s=stops,
        metadata={"fixture": dataset_id, "state_surface": "observed_coordinates"},
    )
    authority = TrajectoryEvidenceAuthority.from_data(
        data,
        case_id=f"{dataset_id}-case",
        fit_indices=fit,
        calibration_indices=calibration,
        evaluation_indices=evaluation,
        representation_fit_indices=fit,
        latent_dimension=dimension,
        expected_window_seconds=0.5,
        expected_step_seconds=1.0,
        step_tolerance_seconds=1e-12,
        purge_seconds=0.0,
        upstream_authority_fingerprint="test-upstream",
        source_revisions={"quantumbci": "test", "encoder": "frozen-test"},
    )
    return data, authority


def _simulate_nonlinear(
    *,
    seed: int,
    n_trajectories: int = 18,
    n_steps: int = 140,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    transition = np.asarray(
        [
            [0.70, 0.04, 0.00],
            [-0.03, 0.72, 0.03],
            [0.02, -0.03, 0.68],
        ],
        dtype=float,
    )
    variance = np.asarray([0.0005, 0.0005, 0.0005], dtype=float)
    output = np.empty((n_trajectories, n_steps, 3), dtype=float)
    for trajectory in range(n_trajectories):
        state = rng.normal(0.0, 0.35, size=3)
        output[trajectory, 0] = state
        for step in range(1, n_steps):
            nonlinear = np.asarray(
                [
                    0.16 * np.sin(4.0 * state[1]),
                    0.14 * np.sin(4.0 * state[2]),
                    0.15 * np.sin(4.0 * state[0]),
                ],
                dtype=float,
            )
            state = (
                transition @ state
                + nonlinear
                + rng.normal(0.0, np.sqrt(variance))
            )
            output[trajectory, step] = state
    return output


def _simulate_linear(
    *,
    seed: int,
    n_trajectories: int = 14,
    n_steps: int = 120,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    transition = np.asarray(
        [
            [0.86, 0.09, -0.03],
            [-0.07, 0.88, 0.05],
            [0.04, -0.06, 0.83],
        ],
        dtype=float,
    )
    intercept = np.asarray([0.010, -0.006, 0.004], dtype=float)
    variance = np.asarray([0.0012, 0.0010, 0.0009], dtype=float)
    output = np.empty((n_trajectories, n_steps, 3), dtype=float)
    for trajectory in range(n_trajectories):
        state = rng.normal(0.0, 0.18, size=3)
        output[trajectory, 0] = state
        for step in range(1, n_steps):
            state = (
                transition @ state
                + intercept
                + rng.normal(0.0, np.sqrt(variance))
            )
            output[trajectory, step] = state
    return output


def _direct_gaussian(
    data: TrajectoryEvidenceData,
    authority: TrajectoryEvidenceAuthority,
):
    transition, intercept, _ = fit_full_affine_var1(data, authority)
    variance = fit_base_innovation_variance(data, authority, transition, intercept)
    metrics = score_direct_gaussian_var(
        data,
        authority,
        transition,
        intercept,
        variance,
        role="evaluation",
        open_loop=False,
    )
    return transition, intercept, metrics


def test_nonlinear_residual_wins_on_true_nonlinear_process() -> None:
    data, authority = _authority(
        _simulate_nonlinear(seed=17),
        dataset_id="nonlinear-positive",
    )
    transition, intercept, direct = _direct_gaussian(data, authority)
    result = run_nonlinear_residual_control(
        data,
        authority,
        transition,
        intercept,
    )
    payload = result.to_mapping()

    assert payload["model"]["model_id"] == NONLINEAR_MODEL_ID
    assert payload["evaluation_metrics"]["score_id"] == NONLINEAR_SCORE_ID
    assert payload["model"]["affine_mean_refit"] is False
    assert np.asarray(payload["model"]["transition"]) == pytest.approx(transition)
    assert np.asarray(payload["model"]["intercept"]) == pytest.approx(intercept)
    assert payload["model"]["rff_seed"] == RFF_SEED
    assert payload["candidate_grid"]["feature_counts"] == list(FEATURE_COUNTS)
    assert payload["candidate_grid"]["length_scale_multipliers"] == list(
        LENGTH_SCALE_MULTIPLIERS
    )
    assert payload["candidate_grid"]["ridges"] == list(RIDGES)
    assert payload["fit_authority_only_for_weights_and_standardization"] is True
    assert payload["calibration_used_for_complexity_selection"] is True
    assert payload["evaluation_used_for_model_selection"] is False
    assert payload["one_step_gaussian_density_complete"] is True
    assert payload["deterministic_mean_rollout_complete"] is True
    assert payload["nonlinear_uncertainty_rollout_complete"] is False
    assert payload["rollout_likelihood_promotion_eligible"] is False
    assert len(payload["model"]["model_sha256"]) == 64
    assert len(payload["candidates"]) == (
        len(FEATURE_COUNTS) * len(LENGTH_SCALE_MULTIPLIERS) * len(RIDGES)
    )

    gain = direct.mean_nll - result.evaluation_metrics.one_step_mean_nll
    assert gain > 0.25
    assert result.evaluation_metrics.one_step_rmse < direct.predictive_mean_rmse


def test_nonlinear_complexity_does_not_manufacture_linear_null_gain() -> None:
    data, authority = _authority(
        _simulate_linear(seed=29),
        dataset_id="nonlinear-null",
    )
    transition, intercept, direct = _direct_gaussian(data, authority)
    result = run_nonlinear_residual_control(
        data,
        authority,
        transition,
        intercept,
    )

    gain = direct.mean_nll - result.evaluation_metrics.one_step_mean_nll
    assert gain < 0.08


def test_evaluation_corruption_cannot_change_nonlinear_selection_or_weights() -> None:
    trajectories = _simulate_nonlinear(seed=41)
    original_data, original_authority = _authority(
        trajectories,
        dataset_id="nonlinear-evaluation-readonly",
    )
    shifted = trajectories.copy()
    shifted[:, 90:, :] += np.asarray([2.0, -1.5, 1.0])
    shifted_data, shifted_authority = _authority(
        shifted,
        dataset_id="nonlinear-evaluation-readonly",
    )
    transition, intercept, _ = _direct_gaussian(original_data, original_authority)

    original = run_nonlinear_residual_control(
        original_data,
        original_authority,
        transition,
        intercept,
    )
    corrupted = run_nonlinear_residual_control(
        shifted_data,
        shifted_authority,
        transition,
        intercept,
    )

    assert original_data.data_sha256 != shifted_data.data_sha256
    assert corrupted.model.feature_count == original.model.feature_count
    assert corrupted.model.length_scale_multiplier == original.model.length_scale_multiplier
    assert corrupted.model.ridge == original.model.ridge
    assert corrupted.model.model_sha256 == original.model.model_sha256
    assert corrupted.model.residual_weights == pytest.approx(
        original.model.residual_weights, abs=1e-12
    )
    assert [candidate.to_mapping() for candidate in corrupted.candidates] == [
        candidate.to_mapping() for candidate in original.candidates
    ]
    assert corrupted.evaluation_metrics.one_step_mean_nll != pytest.approx(
        original.evaluation_metrics.one_step_mean_nll
    )


def test_nonlinear_control_fails_closed_without_calibration_authority() -> None:
    data, authority = _authority(
        _simulate_linear(seed=53, n_trajectories=8, n_steps=80),
        dataset_id="nonlinear-no-calibration",
        fit_steps=48,
        calibration_steps=12,
    )
    transition, intercept, _ = _direct_gaussian(data, authority)
    no_calibration = TrajectoryEvidenceAuthority.from_data(
        data,
        case_id="nonlinear-no-calibration-case",
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
        run_nonlinear_residual_control(
            data,
            no_calibration,
            transition,
            intercept,
        )


def test_unphysical_nonlinear_predictions_are_not_labeled_trace_distance() -> None:
    data, authority = _authority(
        _simulate_linear(seed=67, n_trajectories=8, n_steps=80),
        dataset_id="nonlinear-physicality",
        fit_steps=48,
        calibration_steps=12,
    )
    model = NonlinearResidualModel(
        transition=np.zeros((3, 3), dtype=float),
        intercept=np.asarray([10.0, 0.0, 0.0], dtype=float),
        state_mean=np.zeros(3, dtype=float),
        state_scale=np.ones(3, dtype=float),
        frequencies=np.zeros((16, 3), dtype=float),
        phases=np.zeros(16, dtype=float),
        residual_weights=np.zeros((16, 3), dtype=float),
        innovation_variance=np.ones(3, dtype=float) * 0.1,
        feature_count=16,
        length_scale_multiplier=1.0,
        ridge=1.0,
        effective_feature_rank=1,
    )
    metrics = evaluate_nonlinear_model(
        data,
        authority,
        model,
        role="evaluation",
    )

    assert metrics.one_step_mean_bloch_half_l2 is not None
    assert metrics.one_step_prediction_physical_fraction == pytest.approx(0.0)
    assert metrics.one_step_valid_qubit_pair_fraction == pytest.approx(0.0)
    assert metrics.one_step_mean_valid_qubit_trace_distance is None
