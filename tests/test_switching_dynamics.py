from __future__ import annotations

import numpy as np
import pytest

from quantumbci.classical_dynamics import fit_full_affine_var1
from quantumbci.probabilistic_ssm import (
    fit_base_innovation_variance,
    score_direct_gaussian_var,
)
from quantumbci.switching_dynamics import (
    INITIALIZATION_IDS,
    LABEL_CANONICALIZATION_ID,
    REGIME_COUNT,
    SWITCHING_MODEL_ID,
    SWITCHING_SCORE_ID,
    SwitchingFitResult,
    fit_switching_affine_var,
    run_switching_state_control,
    score_switching_affine_var,
)
from quantumbci.trajectory_authority import TrajectoryEvidenceAuthority, TrajectoryEvidenceData


def _authority_from_trajectories(
    trajectories: np.ndarray,
    *,
    dataset_id: str,
    fit_steps: int,
    calibration_steps: int,
) -> tuple[TrajectoryEvidenceData, TrajectoryEvidenceAuthority]:
    values = np.asarray(trajectories, dtype=float)
    if values.ndim != 3:
        raise ValueError("trajectories must be trajectory x time x state")
    n_trajectories, n_steps, dimension = values.shape
    if fit_steps + calibration_steps >= n_steps:
        raise ValueError("evaluation role must contain at least one sample")

    states = values.reshape(-1, dimension)
    ids = np.concatenate(
        [np.repeat(f"trajectory-{index}", n_steps) for index in range(n_trajectories)]
    )
    starts = np.tile(np.arange(n_steps, dtype=float), n_trajectories)
    stops = starts + 0.5
    fit_indices: list[int] = []
    calibration_indices: list[int] = []
    evaluation_indices: list[int] = []
    for trajectory in range(n_trajectories):
        base = trajectory * n_steps
        fit_indices.extend(range(base, base + fit_steps))
        calibration_indices.extend(
            range(base + fit_steps, base + fit_steps + calibration_steps)
        )
        evaluation_indices.extend(
            range(base + fit_steps + calibration_steps, base + n_steps)
        )

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
        fit_indices=fit_indices,
        calibration_indices=calibration_indices,
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


def _simulate_switching(
    *,
    seed: int,
    n_trajectories: int = 14,
    n_steps: int = 120,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    transitions = np.asarray(
        [
            [
                [0.91, 0.17, 0.00],
                [-0.12, 0.84, 0.07],
                [0.02, -0.08, 0.87],
            ],
            [
                [0.72, -0.23, 0.08],
                [0.20, 0.76, -0.12],
                [-0.05, 0.15, 0.78],
            ],
        ],
        dtype=float,
    )
    intercepts = np.asarray(
        [[0.015, -0.010, 0.006], [-0.020, 0.018, -0.008]],
        dtype=float,
    )
    variances = np.asarray(
        [[0.0009, 0.0012, 0.0008], [0.0011, 0.0008, 0.0010]],
        dtype=float,
    )
    regime_transition = np.asarray([[0.965, 0.035], [0.055, 0.945]], dtype=float)

    output = np.empty((n_trajectories, n_steps, 3), dtype=float)
    for trajectory in range(n_trajectories):
        state = rng.normal(0.0, 0.18, size=3)
        regime = int(rng.integers(0, 2))
        output[trajectory, 0] = state
        for step in range(1, n_steps):
            regime = int(rng.choice(2, p=regime_transition[regime]))
            state = (
                transitions[regime] @ state
                + intercepts[regime]
                + rng.normal(0.0, np.sqrt(variances[regime]))
            )
            output[trajectory, step] = state
    return output


def _simulate_single_regime(
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


def _direct_gaussian_evaluation(
    data: TrajectoryEvidenceData,
    authority: TrajectoryEvidenceAuthority,
):
    transition, intercept, _ = fit_full_affine_var1(data, authority)
    variance = fit_base_innovation_variance(data, authority, transition, intercept)
    return score_direct_gaussian_var(
        data,
        authority,
        transition,
        intercept,
        variance,
        role="evaluation",
        open_loop=False,
    )


def test_switching_model_wins_on_true_two_regime_process() -> None:
    data, authority = _authority_from_trajectories(
        _simulate_switching(seed=17),
        dataset_id="switching-positive",
        fit_steps=72,
        calibration_steps=18,
    )
    result = run_switching_state_control(data, authority)
    direct = _direct_gaussian_evaluation(data, authority)
    payload = result.to_mapping()

    assert payload["model"]["model_id"] == SWITCHING_MODEL_ID
    assert payload["evaluation_metrics"]["score_id"] == SWITCHING_SCORE_ID
    assert payload["model"]["regime_count"] == REGIME_COUNT == 2
    assert payload["model"]["label_canonicalization_id"] == LABEL_CANONICALIZATION_ID
    assert payload["model"]["regime_labels_mechanistically_identifiable"] is False
    assert payload["role_boundary_regime_belief_reset"] is True
    assert payload["sequential_predictive_density_complete"] is True
    assert payload["exact_open_loop_switching_forecast_complete"] is False
    assert payload["open_loop_promotion_eligible"] is False
    assert payload["calibration_used_for_model_selection"] is False
    assert payload["evaluation_used_for_model_selection"] is False
    assert result.model.parameter_count == 33
    assert result.model.converged is True
    assert set(name for name, _ in result.model.initialization_log_likelihoods).issubset(
        set(INITIALIZATION_IDS)
    )
    assert len(result.model.initialization_log_likelihoods) >= 2
    assert np.all(result.model.effective_transition_counts >= 8.0)
    assert np.allclose(np.sum(result.model.regime_transition, axis=1), 1.0)
    assert np.isclose(np.sum(result.model.initial_probabilities), 1.0)

    # A real switching adversary must materially improve held-out predictive density.
    assert direct.mean_nll - result.evaluation_metrics.mean_nll > 0.12
    assert result.evaluation_metrics.mean_max_predictive_regime_probability > 0.60


def test_switching_complexity_does_not_manufacture_large_null_gain() -> None:
    data, authority = _authority_from_trajectories(
        _simulate_single_regime(seed=29),
        dataset_id="switching-null",
        fit_steps=72,
        calibration_steps=18,
    )
    result = run_switching_state_control(data, authority)
    direct = _direct_gaussian_evaluation(data, authority)

    # On a true one-regime Gaussian system, held-out switching gains must stay small.
    # A negative value means the simpler direct Gaussian VAR wins, which is acceptable.
    direct_minus_switching = direct.mean_nll - result.evaluation_metrics.mean_nll
    assert direct_minus_switching < 0.10


def test_regime_label_permutation_is_predictively_invariant() -> None:
    data, authority = _authority_from_trajectories(
        _simulate_switching(seed=41),
        dataset_id="switching-label-invariance",
        fit_steps=72,
        calibration_steps=18,
    )
    fitted = fit_switching_affine_var(data, authority)
    original = score_switching_affine_var(
        data, authority, fitted, role="evaluation"
    )
    order = np.asarray([1, 0], dtype=int)
    permuted = SwitchingFitResult(
        transitions=fitted.transitions[order],
        intercepts=fitted.intercepts[order],
        variances=fitted.variances[order],
        regime_transition=fitted.regime_transition[np.ix_(order, order)],
        initial_probabilities=fitted.initial_probabilities[order],
        effective_transition_counts=fitted.effective_transition_counts[order],
        fit_log_likelihood=fitted.fit_log_likelihood,
        fit_mean_nll=fitted.fit_mean_nll,
        iterations=fitted.iterations,
        converged=fitted.converged,
        initialization_id=fitted.initialization_id,
        initialization_log_likelihoods=fitted.initialization_log_likelihoods,
        canonicalization_permutation=tuple(
            int(value) for value in reversed(fitted.canonicalization_permutation)
        ),
    )
    swapped = score_switching_affine_var(
        data, authority, permuted, role="evaluation"
    )

    assert swapped.mean_nll == pytest.approx(original.mean_nll, abs=1e-12)
    assert swapped.predictive_mean_rmse == pytest.approx(
        original.predictive_mean_rmse, abs=1e-12
    )
    assert swapped.mean_predictive_regime_entropy == pytest.approx(
        original.mean_predictive_regime_entropy, abs=1e-12
    )


def test_evaluation_values_cannot_change_fitted_switching_model() -> None:
    trajectories = _simulate_switching(seed=53)
    original_data, original_authority = _authority_from_trajectories(
        trajectories,
        dataset_id="switching-evaluation-readonly",
        fit_steps=72,
        calibration_steps=18,
    )
    shifted = trajectories.copy()
    shifted[:, 90:, :] += np.asarray([3.0, -2.0, 1.5])
    shifted_data, shifted_authority = _authority_from_trajectories(
        shifted,
        dataset_id="switching-evaluation-readonly",
        fit_steps=72,
        calibration_steps=18,
    )

    original = fit_switching_affine_var(original_data, original_authority)
    corrupted = fit_switching_affine_var(shifted_data, shifted_authority)

    assert shifted_data.data_sha256 != original_data.data_sha256
    assert corrupted.initialization_id == original.initialization_id
    assert corrupted.fit_log_likelihood == pytest.approx(original.fit_log_likelihood, abs=1e-12)
    assert corrupted.transitions == pytest.approx(original.transitions, abs=1e-12)
    assert corrupted.intercepts == pytest.approx(original.intercepts, abs=1e-12)
    assert corrupted.variances == pytest.approx(original.variances, abs=1e-12)
    assert corrupted.regime_transition == pytest.approx(
        original.regime_transition, abs=1e-12
    )
    assert corrupted.initial_probabilities == pytest.approx(
        original.initial_probabilities, abs=1e-12
    )


def test_unphysical_switching_predictive_means_are_not_trace_distance() -> None:
    data, authority = _authority_from_trajectories(
        _simulate_single_regime(seed=67, n_trajectories=6, n_steps=60),
        dataset_id="switching-physicality",
        fit_steps=36,
        calibration_steps=10,
    )
    model = SwitchingFitResult(
        transitions=np.zeros((2, 3, 3), dtype=float),
        intercepts=np.asarray([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float),
        variances=np.ones((2, 3), dtype=float) * 0.1,
        regime_transition=np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=float),
        initial_probabilities=np.asarray([0.5, 0.5], dtype=float),
        effective_transition_counts=np.asarray([100.0, 100.0], dtype=float),
        fit_log_likelihood=0.0,
        fit_mean_nll=0.0,
        iterations=1,
        converged=True,
        initialization_id="test",
        initialization_log_likelihoods=(("test", 0.0),),
        canonicalization_permutation=(0, 1),
    )
    metrics = score_switching_affine_var(
        data, authority, model, role="evaluation"
    )

    assert metrics.mean_bloch_half_l2 is not None
    assert metrics.prediction_physical_fraction == pytest.approx(0.0)
    assert metrics.valid_qubit_pair_fraction == pytest.approx(0.0)
    assert metrics.mean_valid_qubit_trace_distance is None
