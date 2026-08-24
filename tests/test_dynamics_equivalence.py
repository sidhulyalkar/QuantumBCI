from __future__ import annotations

import numpy as np
import pytest

from quantumbci.dynamics_equivalence import (
    SIGMA_X,
    SIGMA_Z,
    audit_lindblad_gauge_nonidentifiability,
    audit_qubit_lindblad_affine_equivalence,
    bloch_to_density,
    compile_qubit_lindblad_to_affine,
    density_to_bloch,
    evolve_affine_bloch,
)
from quantumbci.e002_synthetic import CanonicalQubitParameters, canonical_qubit_model
from quantumbci.open_system import evolve_lindblad


def _model():
    return canonical_qubit_model(
        CanonicalQubitParameters(
            omega_x=1.2,
            omega_z=0.8,
            gamma_dephasing=0.25,
            gamma_relaxation=0.35,
        )
    )


def test_bloch_density_roundtrip() -> None:
    vector = np.asarray([0.35, -0.25, 0.40])
    assert np.allclose(density_to_bloch(bloch_to_density(vector)), vector, atol=1e-12)
    with pytest.raises(ValueError, match="outside"):
        bloch_to_density(np.asarray([1.1, 0.0, 0.0]))


def test_canonical_lindblad_compiles_to_expected_affine_bloch_generator() -> None:
    hamiltonian, collapses = _model()
    compiled = compile_qubit_lindblad_to_affine(hamiltonian, collapses)
    expected_matrix = np.asarray(
        [
            [-0.425, -0.8, 0.0],
            [0.8, -0.425, -1.2],
            [0.0, 1.2, -0.35],
        ]
    )
    expected_offset = np.asarray([0.0, 0.0, 0.35])
    assert np.allclose(compiled.matrix, expected_matrix, atol=1e-12)
    assert np.allclose(compiled.offset, expected_offset, atol=1e-12)
    assert compiled.dynamical_information_novel is False


def test_lindblad_and_affine_bloch_trajectories_are_equivalent() -> None:
    hamiltonian, collapses = _model()
    audit = audit_qubit_lindblad_affine_equivalence(hamiltonian, collapses)
    assert audit.equivalent_within_tolerance is True
    assert audit.dynamical_information_novel is False
    assert audit.max_generator_error < 1e-12
    assert audit.max_trajectory_error < 1e-10

    initial = np.asarray([0.2, 0.4, -0.3])
    times = np.linspace(0.0, 2.0, 201)
    compiled = audit.generator
    density = evolve_lindblad(
        bloch_to_density(initial),
        hamiltonian,
        times,
        collapse_operators=collapses,
        project_each_step=False,
    )
    lindblad_bloch = np.stack(
        [density_to_bloch(state, require_physical=False) for state in density]
    )
    affine = evolve_affine_bloch(initial, times, compiled.matrix, compiled.offset)
    assert np.allclose(lindblad_bloch, affine, atol=1e-10)


def test_standard_lindblad_parameter_gauges_leave_generator_unchanged() -> None:
    hamiltonian, collapses = _model()
    audit = audit_lindblad_gauge_nonidentifiability(hamiltonian, collapses)
    mapping = audit.to_mapping()
    assert audit.equivalent_within_tolerance is True
    assert mapping["hamiltonian_identity_shift_unidentifiable"] is True
    assert mapping["collapse_global_phase_unidentifiable"] is True
    assert mapping["collapse_unitary_mixing_unidentifiable"] is True
    assert audit.hamiltonian_identity_shift_error < 1e-12
    assert audit.collapse_phase_error is not None and audit.collapse_phase_error < 1e-12
    assert audit.collapse_unitary_mixing_error is not None
    assert audit.collapse_unitary_mixing_error < 1e-12


def test_invalid_qubit_generator_fails_closed() -> None:
    with pytest.raises(ValueError, match="Hermitian"):
        compile_qubit_lindblad_to_affine(
            np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=complex),
            (),
        )
    with pytest.raises(ValueError, match="shape"):
        compile_qubit_lindblad_to_affine(np.eye(3), ())
    with pytest.raises(ValueError, match="shape"):
        compile_qubit_lindblad_to_affine(
            0.5 * (SIGMA_X + SIGMA_Z),
            (np.eye(3),),
        )
