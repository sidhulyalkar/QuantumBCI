"""Classical-equivalence and gauge audits for low-dimensional open-system dynamics.

For a qubit, a time-homogeneous GKSL/Lindblad master equation is exactly an affine
linear ordinary differential equation on the three real Bloch coordinates. This
module makes that classical compilation explicit before E002 interprets a
Lindblad-style parameterization as a novel neural dynamical mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .open_system import evolve_lindblad, lindblad_rhs
from .states import is_density_matrix

Array = np.ndarray

IDENTITY_2 = np.eye(2, dtype=complex)
SIGMA_X = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SIGMA_Y = np.asarray([[0.0, -1j], [1j, 0.0]], dtype=complex)
SIGMA_Z = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
PAULI_BASIS = (SIGMA_X, SIGMA_Y, SIGMA_Z)


def _validate_qubit_generator(
    hamiltonian: Array,
    collapse_operators: Sequence[Array],
) -> tuple[Array, tuple[Array, ...]]:
    h = np.asarray(hamiltonian, dtype=complex)
    if h.shape != (2, 2):
        raise ValueError("qubit hamiltonian must have shape (2, 2)")
    if not np.all(np.isfinite(h)):
        raise ValueError("hamiltonian contains non-finite values")
    if not np.allclose(h, h.conj().T, atol=1e-10):
        raise ValueError("hamiltonian must be Hermitian")
    collapses: list[Array] = []
    for operator in collapse_operators:
        value = np.asarray(operator, dtype=complex)
        if value.shape != (2, 2):
            raise ValueError("qubit collapse operators must have shape (2, 2)")
        if not np.all(np.isfinite(value)):
            raise ValueError("collapse operator contains non-finite values")
        collapses.append(value)
    return h, tuple(collapses)


def density_to_bloch(rho: Array, *, require_physical: bool = True) -> Array:
    """Convert a 2x2 density operator to three real Bloch coordinates."""

    state = np.asarray(rho, dtype=complex)
    if state.shape != (2, 2):
        raise ValueError("rho must have shape (2, 2)")
    if require_physical and not is_density_matrix(state, atol=1e-8):
        raise ValueError("rho must be a valid qubit density matrix")
    values = np.asarray([np.trace(sigma @ state) for sigma in PAULI_BASIS])
    if np.max(np.abs(values.imag)) > 1e-9:
        raise ValueError("Bloch coordinates have unexpected imaginary residue")
    return values.real.astype(float)


def bloch_to_density(vector: Array, *, atol: float = 1e-10) -> Array:
    """Convert a physical Bloch vector to a 2x2 density operator."""

    r = np.asarray(vector, dtype=float).reshape(-1)
    if r.shape != (3,):
        raise ValueError("Bloch vector must contain exactly three coordinates")
    if not np.all(np.isfinite(r)):
        raise ValueError("Bloch vector contains non-finite values")
    if np.linalg.norm(r) > 1.0 + atol:
        raise ValueError("Bloch vector lies outside the physical unit ball")
    state = IDENTITY_2.copy()
    for coefficient, sigma in zip(r, PAULI_BASIS, strict=True):
        state = state + float(coefficient) * sigma
    return state / 2.0


@dataclass(frozen=True)
class BlochAffineGenerator:
    """Exact classical affine representation of one qubit Lindblad generator."""

    matrix: Array
    offset: Array
    max_imaginary_residue: float
    equivalence_class: str = "qubit_gksl_affine_bloch_ode"
    dynamical_information_novel: bool = False

    def to_mapping(self) -> dict[str, Any]:
        return {
            "matrix": np.asarray(self.matrix, dtype=float).tolist(),
            "offset": np.asarray(self.offset, dtype=float).tolist(),
            "max_imaginary_residue": float(self.max_imaginary_residue),
            "equivalence_class": self.equivalence_class,
            "dynamical_information_novel": self.dynamical_information_novel,
            "interpretation": (
                "For a fully observed qubit density state, the time-homogeneous "
                "Lindblad generator contains no trajectory information beyond this "
                "three-dimensional classical affine linear ODE."
            ),
        }


def compile_qubit_lindblad_to_affine(
    hamiltonian: Array,
    collapse_operators: Sequence[Array] = (),
    *,
    atol: float = 1e-10,
) -> BlochAffineGenerator:
    """Compile ``d rho / dt = L(rho)`` exactly into ``d r / dt = A r + b``.

    With ``rho = (I + sum_j r_j sigma_j) / 2``, Bloch derivatives satisfy
    ``dot(r_i) = Tr[sigma_i L(rho)]``. Linearity of ``L`` therefore gives
    ``b_i = Tr[sigma_i L(I)] / 2`` and
    ``A_ij = Tr[sigma_i L(sigma_j)] / 2``.
    """

    if atol <= 0:
        raise ValueError("atol must be positive")
    h, collapses = _validate_qubit_generator(hamiltonian, collapse_operators)

    l_identity = lindblad_rhs(IDENTITY_2, h, collapses)
    raw_offset = np.asarray(
        [0.5 * np.trace(sigma @ l_identity) for sigma in PAULI_BASIS],
        dtype=complex,
    )
    raw_matrix = np.asarray(
        [
            [
                0.5 * np.trace(
                    sigma_i @ lindblad_rhs(sigma_j, h, collapses)
                )
                for sigma_j in PAULI_BASIS
            ]
            for sigma_i in PAULI_BASIS
        ],
        dtype=complex,
    )
    residue = float(
        max(
            np.max(np.abs(raw_offset.imag)),
            np.max(np.abs(raw_matrix.imag)),
        )
    )
    if residue > atol:
        raise RuntimeError(
            f"physical qubit generator produced complex Bloch coefficients: {residue:.3e}"
        )
    return BlochAffineGenerator(
        matrix=raw_matrix.real.astype(float),
        offset=raw_offset.real.astype(float),
        max_imaginary_residue=residue,
    )


def affine_rhs(vector: Array, matrix: Array, offset: Array) -> Array:
    r = np.asarray(vector, dtype=float).reshape(-1)
    a = np.asarray(matrix, dtype=float)
    b = np.asarray(offset, dtype=float).reshape(-1)
    if r.shape != (3,) or a.shape != (3, 3) or b.shape != (3,):
        raise ValueError("affine qubit dynamics require r=(3,), A=(3,3), b=(3,)")
    return a @ r + b


def evolve_affine_bloch(
    vector0: Array,
    times: Array,
    matrix: Array,
    offset: Array,
) -> Array:
    """Evolve an affine Bloch ODE with the same RK4 scheme used by ``evolve_lindblad``."""

    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or len(t) < 1:
        raise ValueError("times must be a non-empty 1D array")
    if np.any(np.diff(t) <= 0):
        raise ValueError("times must be strictly increasing")
    r = np.asarray(vector0, dtype=float).reshape(-1)
    if r.shape != (3,):
        raise ValueError("vector0 must contain three Bloch coordinates")
    trajectory = np.empty((len(t), 3), dtype=float)
    trajectory[0] = r
    for index, dt in enumerate(np.diff(t), start=1):
        k1 = affine_rhs(r, matrix, offset)
        k2 = affine_rhs(r + 0.5 * dt * k1, matrix, offset)
        k3 = affine_rhs(r + 0.5 * dt * k2, matrix, offset)
        k4 = affine_rhs(r + dt * k3, matrix, offset)
        r = r + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        trajectory[index] = r
    return trajectory


@dataclass(frozen=True)
class LindbladAffineEquivalenceAudit:
    max_generator_error: float
    max_trajectory_error: float
    equivalent_within_tolerance: bool
    tolerance: float
    generator: BlochAffineGenerator
    equivalence_class: str = "qubit_gksl_affine_bloch_ode"
    dynamical_information_novel: bool = False

    def to_mapping(self) -> dict[str, Any]:
        return {
            "max_generator_error": float(self.max_generator_error),
            "max_trajectory_error": float(self.max_trajectory_error),
            "equivalent_within_tolerance": self.equivalent_within_tolerance,
            "tolerance": float(self.tolerance),
            "equivalence_class": self.equivalence_class,
            "dynamical_information_novel": self.dynamical_information_novel,
            "generator": self.generator.to_mapping(),
            "interpretation": (
                "A fully observed qubit Lindblad trajectory is exactly reproducible by "
                "a classical three-state affine linear system. Any empirical value of "
                "the Lindblad parameterization must come from constraints, regularization, "
                "interpretability or intervention structure rather than extra trajectory information."
            ),
        }


def audit_qubit_lindblad_affine_equivalence(
    hamiltonian: Array,
    collapse_operators: Sequence[Array] = (),
    *,
    atol: float = 1e-9,
) -> LindbladAffineEquivalenceAudit:
    """Numerically witness generator- and trajectory-level qubit equivalence."""

    if atol <= 0:
        raise ValueError("atol must be positive")
    h, collapses = _validate_qubit_generator(hamiltonian, collapse_operators)
    compiled = compile_qubit_lindblad_to_affine(h, collapses, atol=atol)

    probes = (
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([0.55, 0.0, 0.0]),
        np.asarray([0.0, -0.40, 0.20]),
        np.asarray([0.25, 0.35, -0.30]),
    )
    generator_errors: list[float] = []
    for vector in probes:
        rho = bloch_to_density(vector)
        derivative = lindblad_rhs(rho, h, collapses)
        direct = np.asarray(
            [np.trace(sigma @ derivative).real for sigma in PAULI_BASIS]
        )
        classical = affine_rhs(vector, compiled.matrix, compiled.offset)
        generator_errors.append(float(np.max(np.abs(direct - classical))))

    initial = np.asarray([0.35, -0.25, 0.40])
    times = np.linspace(0.0, 1.5, 151)
    density_trajectory = evolve_lindblad(
        bloch_to_density(initial),
        h,
        times,
        collapse_operators=collapses,
        project_each_step=False,
    )
    direct_bloch = np.stack(
        [density_to_bloch(state, require_physical=False) for state in density_trajectory]
    )
    classical_bloch = evolve_affine_bloch(
        initial,
        times,
        compiled.matrix,
        compiled.offset,
    )
    trajectory_error = float(np.max(np.abs(direct_bloch - classical_bloch)))
    generator_error = max(generator_errors)
    equivalent = bool(max(generator_error, trajectory_error) <= atol)
    return LindbladAffineEquivalenceAudit(
        max_generator_error=generator_error,
        max_trajectory_error=trajectory_error,
        equivalent_within_tolerance=equivalent,
        tolerance=float(atol),
        generator=compiled,
    )


@dataclass(frozen=True)
class LindbladGaugeAudit:
    hamiltonian_identity_shift_error: float
    collapse_phase_error: float | None
    collapse_unitary_mixing_error: float | None
    equivalent_within_tolerance: bool
    tolerance: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "hamiltonian_identity_shift_unidentifiable": True,
            "collapse_global_phase_unidentifiable": self.collapse_phase_error is not None,
            "collapse_unitary_mixing_unidentifiable": self.collapse_unitary_mixing_error is not None,
            "hamiltonian_identity_shift_error": float(self.hamiltonian_identity_shift_error),
            "collapse_phase_error": (
                None if self.collapse_phase_error is None else float(self.collapse_phase_error)
            ),
            "collapse_unitary_mixing_error": (
                None
                if self.collapse_unitary_mixing_error is None
                else float(self.collapse_unitary_mixing_error)
            ),
            "equivalent_within_tolerance": self.equivalent_within_tolerance,
            "tolerance": float(self.tolerance),
            "interpretation": (
                "Individual Hamiltonian offsets, collapse-operator phases, and unitary "
                "rotations of a multi-channel collapse basis are gauge choices rather than "
                "separately identifiable mechanisms. E002 recovery claims must use a declared "
                "canonical gauge or operate on generator-level invariants."
            ),
        }


def _generator_distance(first: BlochAffineGenerator, second: BlochAffineGenerator) -> float:
    return float(
        max(
            np.max(np.abs(first.matrix - second.matrix)),
            np.max(np.abs(first.offset - second.offset)),
        )
    )


def audit_lindblad_gauge_nonidentifiability(
    hamiltonian: Array,
    collapse_operators: Sequence[Array] = (),
    *,
    atol: float = 1e-9,
) -> LindbladGaugeAudit:
    """Demonstrate standard parameter gauges that leave the physical generator unchanged."""

    if atol <= 0:
        raise ValueError("atol must be positive")
    h, collapses = _validate_qubit_generator(hamiltonian, collapse_operators)
    baseline = compile_qubit_lindblad_to_affine(h, collapses, atol=atol)

    shifted = compile_qubit_lindblad_to_affine(
        h + 0.731 * IDENTITY_2,
        collapses,
        atol=atol,
    )
    h_error = _generator_distance(baseline, shifted)

    phase_error: float | None = None
    if collapses:
        phased = tuple(
            np.exp(1j * (0.37 + 0.19 * index)) * operator
            for index, operator in enumerate(collapses)
        )
        phase_error = _generator_distance(
            baseline,
            compile_qubit_lindblad_to_affine(h, phased, atol=atol),
        )

    mixing_error: float | None = None
    if len(collapses) >= 2:
        mixed = list(collapses)
        first, second = collapses[0], collapses[1]
        mixed[0] = (first + second) / np.sqrt(2.0)
        mixed[1] = (first - second) / np.sqrt(2.0)
        mixing_error = _generator_distance(
            baseline,
            compile_qubit_lindblad_to_affine(h, tuple(mixed), atol=atol),
        )

    errors = [h_error]
    if phase_error is not None:
        errors.append(phase_error)
    if mixing_error is not None:
        errors.append(mixing_error)
    return LindbladGaugeAudit(
        hamiltonian_identity_shift_error=h_error,
        collapse_phase_error=phase_error,
        collapse_unitary_mixing_error=mixing_error,
        equivalent_within_tolerance=bool(max(errors) <= atol),
        tolerance=float(atol),
    )
