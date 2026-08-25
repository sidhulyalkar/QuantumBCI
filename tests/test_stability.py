from __future__ import annotations

import numpy as np
import pytest

from quantumbci.e002_synthetic import (
    CanonicalQubitParameters,
    simulate_canonical_bloch_trajectories,
)
from quantumbci.stability import (
    BOOTSTRAP_METHOD_ID,
    run_e002_bootstrap_stability,
)
from quantumbci.trajectory_authority import TrajectoryEvidenceAuthority, TrajectoryEvidenceData


def _case(
    trajectories: np.ndarray,
    *,
    dataset_id: str,
    step_seconds: float,
    fit_steps: int = 48,
    calibration_steps: int = 18,
) -> tuple[TrajectoryEvidenceData, TrajectoryEvidenceAuthority]:
    values = np.asarray(trajectories, dtype=float)
    n_trajectories, n_steps, dimension = values.shape
    states = values.reshape(-1, dimension)
    ids = np.concatenate(
        [np.repeat(f"trajectory-{index}", n_steps) for index in range(n_trajectories)]
    )
    starts = np.tile(np.arange(n_steps, dtype=float) * step_seconds, n_trajectories)
    stops = starts + 0.5 * step_seconds
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
        metadata={"fixture": dataset_id, "state_surface": "bloch_coordinates"},
    )
    authority = TrajectoryEvidenceAuthority.from_data(
        data,
        case_id=f"{dataset_id}-case",
        fit_indices=fit,
        calibration_indices=calibration,
        evaluation_indices=evaluation,
        representation_fit_indices=fit,
        latent_dimension=3,
        expected_window_seconds=0.5 * step_seconds,
        expected_step_seconds=step_seconds,
        step_tolerance_seconds=1e-10,
        purge_seconds=0.0,
        upstream_authority_fingerprint="test-upstream",
        source_revisions={"quantumbci": "test", "encoder": "frozen-test"},
    )
    return data, authority


def _stable_canonical_case(seed: int = 7):
    rng = np.random.default_rng(seed)
    parameters = CanonicalQubitParameters(0.70, -0.45, 0.14, 0.22)
    step = 0.04
    times = np.arange(92, dtype=float) * step
    initial = rng.normal(0.0, 0.22, size=(12, 3))
    initial /= np.maximum(1.0, np.linalg.norm(initial, axis=1, keepdims=True) / 0.65)
    trajectories = simulate_canonical_bloch_trajectories(parameters, times, initial)
    trajectories = trajectories + rng.normal(0.0, 2e-4, size=trajectories.shape)
    return _case(trajectories, dataset_id="stable-canonical", step_seconds=step), parameters


def _heterogeneous_case(seed: int = 11):
    rng = np.random.default_rng(seed)
    step = 0.04
    times = np.arange(92, dtype=float) * step
    initial = rng.normal(0.0, 0.22, size=(12, 3))
    initial /= np.maximum(1.0, np.linalg.norm(initial, axis=1, keepdims=True) / 0.65)
    left = simulate_canonical_bloch_trajectories(
        CanonicalQubitParameters(0.45, -0.25, 0.08, 0.14),
        times,
        initial[:6],
    )
    right = simulate_canonical_bloch_trajectories(
        CanonicalQubitParameters(1.00, -0.75, 0.27, 0.38),
        times,
        initial[6:],
    )
    trajectories = np.concatenate([left, right], axis=0)
    trajectories = trajectories + rng.normal(0.0, 2e-4, size=trajectories.shape)
    return _case(trajectories, dataset_id="heterogeneous-canonical", step_seconds=step)


def test_stable_canonical_case_produces_reproducible_bootstrap_evidence() -> None:
    (data, authority), truth = _stable_canonical_case()
    first = run_e002_bootstrap_stability(data, authority, n_replicates=8, seed=1401)
    second = run_e002_bootstrap_stability(data, authority, n_replicates=8, seed=1401)
    payload = first.to_mapping()

    assert first.to_mapping() == second.to_mapping()
    assert payload["bootstrap_method_id"] == BOOTSTRAP_METHOD_ID
    assert payload["success_count"] == 8
    assert payload["failure_count"] == 0
    assert payload["evaluation_resampled"] is False
    assert payload["fit_trajectory_blocks_resampled"] is True
    assert payload["calibration_trajectory_blocks_resampled"] is True
    assert payload["single_case_bootstrap_is_icc"] is False
    assert payload["participant_icc_computed"] is False
    assert payload["intervention_stage_eligible"] is False
    assert payload["physical_quantum_promotion_eligible"] is False

    point = payload["point_estimates"]
    assert point["omega_x"] == pytest.approx(truth.omega_x, abs=0.04)
    assert point["omega_z"] == pytest.approx(truth.omega_z, abs=0.04)
    assert point["gamma_dephasing"] == pytest.approx(truth.gamma_dephasing, abs=0.04)
    assert point["gamma_relaxation"] == pytest.approx(truth.gamma_relaxation, abs=0.04)
    for name in ("omega_x", "omega_z", "gamma_dephasing", "gamma_relaxation"):
        summary = payload["parameter_summaries"][name]
        assert summary["finite_fraction"] == 1.0
        assert summary["sign_consistency"] >= 0.875
        assert summary["ci_low"] <= summary["bootstrap_median"] <= summary["ci_high"]

    selection = payload["nonlinear_selection_stability"]
    assert 0.0 < selection["mode_frequency"] <= 1.0
    assert selection["unique_configurations"] >= 1
    assert all(len(row["source_draw_sha256"]) == 64 for row in payload["replicates"])


def test_heterogeneous_source_population_widens_mechanism_uncertainty() -> None:
    (stable_data, stable_authority), _ = _stable_canonical_case(seed=19)
    heterogeneous_data, heterogeneous_authority = _heterogeneous_case(seed=23)
    stable = run_e002_bootstrap_stability(
        stable_data, stable_authority, n_replicates=10, seed=1411
    )
    heterogeneous = run_e002_bootstrap_stability(
        heterogeneous_data, heterogeneous_authority, n_replicates=10, seed=1411
    )

    stable_widths = [
        stable.parameter_summaries[name].ci_high
        - stable.parameter_summaries[name].ci_low
        for name in ("omega_x", "omega_z", "gamma_dephasing", "gamma_relaxation")
    ]
    heterogeneous_widths = [
        heterogeneous.parameter_summaries[name].ci_high
        - heterogeneous.parameter_summaries[name].ci_low
        for name in ("omega_x", "omega_z", "gamma_dephasing", "gamma_relaxation")
    ]
    assert np.mean(heterogeneous_widths) > 1.5 * np.mean(stable_widths)


def test_evaluation_corruption_changes_scores_but_not_source_parameter_bootstrap() -> None:
    (data, authority), _ = _stable_canonical_case(seed=31)
    corrupted_states = np.asarray(data.states, dtype=float).copy()
    corrupted_states[np.asarray(authority.evaluation_indices, dtype=int)] += np.asarray(
        [0.8, -0.6, 0.4]
    )
    corrupted_data = TrajectoryEvidenceData(
        dataset_id=data.dataset_id,
        states=corrupted_states,
        trajectory_ids=data.trajectory_ids,
        start_times_s=data.start_times_s,
        stop_times_s=data.stop_times_s,
        valid_mask=data.valid_mask,
        metadata=dict(data.metadata),
    )
    corrupted_authority = TrajectoryEvidenceAuthority.from_data(
        corrupted_data,
        case_id=authority.case_id,
        fit_indices=authority.fit_indices,
        calibration_indices=authority.calibration_indices,
        evaluation_indices=authority.evaluation_indices,
        representation_fit_indices=authority.representation_fit_indices,
        latent_dimension=authority.latent_dimension,
        expected_window_seconds=authority.expected_window_seconds,
        expected_step_seconds=authority.expected_step_seconds,
        step_tolerance_seconds=authority.step_tolerance_seconds,
        purge_seconds=authority.purge_seconds,
        upstream_authority_fingerprint=authority.upstream_authority_fingerprint,
        source_revisions=dict(authority.source_revisions),
    )

    original = run_e002_bootstrap_stability(data, authority, n_replicates=6, seed=1421)
    corrupted = run_e002_bootstrap_stability(
        corrupted_data, corrupted_authority, n_replicates=6, seed=1421
    )

    assert data.data_sha256 != corrupted_data.data_sha256
    for name in ("omega_x", "omega_z", "gamma_dephasing", "gamma_relaxation"):
        assert original.point_estimates[name] == pytest.approx(corrupted.point_estimates[name])
        assert original.parameter_summaries[name].to_mapping() == pytest.approx(
            corrupted.parameter_summaries[name].to_mapping()
        )
    assert (
        original.predictive_summaries["direct_minus_nonlinear_mean_nll"].bootstrap_mean
        != pytest.approx(
            corrupted.predictive_summaries[
                "direct_minus_nonlinear_mean_nll"
            ].bootstrap_mean
        )
    )
    assert (
        original.nonlinear_selection_stability.to_mapping()
        == corrupted.nonlinear_selection_stability.to_mapping()
    )


def test_bootstrap_stability_fails_closed_without_calibration_role() -> None:
    (data, authority), _ = _stable_canonical_case(seed=43)
    no_calibration = TrajectoryEvidenceAuthority.from_data(
        data,
        case_id="no-calibration",
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
        run_e002_bootstrap_stability(data, no_calibration, n_replicates=4)
