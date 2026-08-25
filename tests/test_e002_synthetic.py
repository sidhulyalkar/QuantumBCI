from __future__ import annotations

import numpy as np
import pytest

from quantumbci.dynamics_equivalence import compile_qubit_lindblad_to_affine
from quantumbci.e002_synthetic import (
    CANONICAL_STRUCTURE_RESIDUAL_MAX,
    CLASSICAL_ADVERSARY_RESIDUAL_MIN,
    CanonicalQubitParameters,
    canonical_qubit_model,
    canonical_structure_residual,
    fit_affine_generator_from_trajectories,
    recover_canonical_parameters,
    run_e002_synthetic_recovery_grid,
    simulate_canonical_bloch_trajectories,
)


def test_canonical_parameters_recover_from_exact_generator() -> None:
    truth = CanonicalQubitParameters(1.2, -0.8, 0.25, 0.35)
    hamiltonian, collapses = canonical_qubit_model(truth)
    generator = compile_qubit_lindblad_to_affine(hamiltonian, collapses)
    recovered = recover_canonical_parameters(generator.matrix, generator.offset)
    assert recovered.to_mapping() == pytest.approx(truth.to_mapping(), abs=1e-12)
    assert canonical_structure_residual(
        generator.matrix,
        generator.offset,
        recovered,
    ) < 1e-12


def test_affine_fit_recovers_canonical_family_from_trajectories() -> None:
    truth = CanonicalQubitParameters(0.9, 0.6, 0.2, 0.3)
    times = np.linspace(0.0, 3.0, 301)
    initial = np.asarray(
        [
            [0.8, 0.0, 0.0],
            [0.0, 0.8, 0.0],
            [0.0, 0.0, 0.8],
            [-0.5, 0.3, 0.1],
        ]
    )
    trajectories = simulate_canonical_bloch_trajectories(truth, times, initial)
    matrix, offset = fit_affine_generator_from_trajectories(trajectories, times)
    recovered = recover_canonical_parameters(matrix, offset)
    assert recovered.omega_x == pytest.approx(truth.omega_x, rel=2e-3)
    assert recovered.omega_z == pytest.approx(truth.omega_z, rel=2e-3)
    assert recovered.gamma_dephasing == pytest.approx(truth.gamma_dephasing, rel=3e-3)
    assert recovered.gamma_relaxation == pytest.approx(truth.gamma_relaxation, rel=3e-3)
    assert canonical_structure_residual(matrix, offset, recovered) < 0.01


def test_noncanonical_affine_lookalike_is_not_accepted_as_canonical_family() -> None:
    matrix = np.asarray(
        [
            [-0.20, -0.80, 0.30],
            [0.80, -0.60, -1.20],
            [0.10, 1.20, -0.35],
        ]
    )
    offset = np.asarray([0.10, 0.00, 0.35])
    projected = recover_canonical_parameters(matrix, offset)
    assert projected.omega_x == pytest.approx(1.2)
    assert projected.omega_z == pytest.approx(0.8)
    assert projected.gamma_relaxation == pytest.approx(0.35)

    residual = canonical_structure_residual(matrix, offset, projected)
    assert residual >= CLASSICAL_ADVERSARY_RESIDUAL_MIN
    assert residual > CANONICAL_STRUCTURE_RESIDUAL_MAX
    assert max(np.real(np.linalg.eigvals(matrix))) < 0.0


def test_e002_moderate_snr_recovery_gate_passes_without_overclaiming() -> None:
    result = run_e002_synthetic_recovery_grid(seed=2027, noise_std=0.003)
    assert result["schema_version"] == 2
    assert result["n_cases"] == 6
    assert result["median_normalized_recovery_error"] < 0.05
    assert result["max_case_mean_normalized_recovery_error"] < 0.08
    assert result["max_canonical_structure_residual"] <= CANONICAL_STRUCTURE_RESIDUAL_MAX
    assert result["canonical_structure_pass"] is True
    assert result["classical_adversary"]["rejected_as_canonical_family"] is True
    assert (
        result["classical_adversary"]["canonical_structure_residual"]
        >= CLASSICAL_ADVERSARY_RESIDUAL_MIN
    )
    assert result["systematic_sign_inversions"] == 0
    assert result["affine_equivalence_pass"] is True
    assert result["gauge_nonidentifiability_witness_pass"] is True
    assert result["synthetic_identifiability_gate_pass"] is True
    assert result["dynamical_information_novel"] is False
    assert result["physical_quantum_promotion_eligible"] is False
    assert all(
        case["canonical_structure_residual"] <= CANONICAL_STRUCTURE_RESIDUAL_MAX
        for case in result["cases"]
    )
    assert all(
        case["equivalence_audit"]["dynamical_information_novel"] is False
        for case in result["cases"]
    )


def test_e002_grid_is_deterministic_for_fixed_seed() -> None:
    first = run_e002_synthetic_recovery_grid(seed=7, noise_std=0.002)
    second = run_e002_synthetic_recovery_grid(seed=7, noise_std=0.002)
    assert first == second


def test_e002_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        CanonicalQubitParameters(1.0, 1.0, -0.1, 0.2)
    with pytest.raises(ValueError, match="noise_std"):
        run_e002_synthetic_recovery_grid(noise_std=-1.0)
    with pytest.raises(ValueError, match="shape"):
        fit_affine_generator_from_trajectories(np.zeros((10, 3)), np.arange(10))
    with pytest.raises(ValueError, match=r"A=\(3,3\)"):
        canonical_structure_residual(
            np.zeros((2, 2)),
            np.zeros(3),
            CanonicalQubitParameters(1.0, 1.0, 0.1, 0.1),
        )
