"""Small, inspectable Lindblad dynamics for quantum-like neural latent states."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .states import is_density_matrix, project_density_matrix

Array = np.ndarray


def lindblad_rhs(
    rho: Array,
    hamiltonian: Array,
    collapse_operators: Sequence[Array] = (),
) -> Array:
    """Evaluate the Gorini-Kossakowski-Sudarshan-Lindblad master equation."""

    state = np.asarray(rho, dtype=complex)
    h = np.asarray(hamiltonian, dtype=complex)
    if state.shape != h.shape or state.ndim != 2 or state.shape[0] != state.shape[1]:
        raise ValueError("rho and hamiltonian must be same-shape square matrices")
    derivative = -1j * (h @ state - state @ h)
    for collapse in collapse_operators:
        c = np.asarray(collapse, dtype=complex)
        if c.shape != state.shape:
            raise ValueError("collapse operators must match rho shape")
        cdag_c = c.conj().T @ c
        derivative += c @ state @ c.conj().T - 0.5 * (cdag_c @ state + state @ cdag_c)
    return derivative


def evolve_lindblad(
    rho0: Array,
    hamiltonian: Array,
    times: Array,
    *,
    collapse_operators: Sequence[Array] = (),
    project_each_step: bool = True,
) -> Array:
    """Evolve a density state with fourth-order Runge-Kutta integration.

    The small NumPy implementation keeps the mechanism transparent for interpretability
    experiments. For stiff/high-dimensional systems use a dedicated ODE package.
    """

    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or t.size < 1:
        raise ValueError("times must be a non-empty 1D array")
    if np.any(np.diff(t) <= 0):
        raise ValueError("times must be strictly increasing")
    rho = project_density_matrix(rho0)
    h = np.asarray(hamiltonian, dtype=complex)
    if h.shape != rho.shape or not np.allclose(h, h.conj().T, atol=1e-10):
        raise ValueError("hamiltonian must be Hermitian and match rho shape")

    trajectory = np.empty((t.size, *rho.shape), dtype=complex)
    trajectory[0] = rho
    for i, dt in enumerate(np.diff(t), start=1):
        k1 = lindblad_rhs(rho, h, collapse_operators)
        k2 = lindblad_rhs(rho + 0.5 * dt * k1, h, collapse_operators)
        k3 = lindblad_rhs(rho + 0.5 * dt * k2, h, collapse_operators)
        k4 = lindblad_rhs(rho + dt * k3, h, collapse_operators)
        rho = rho + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if project_each_step:
            rho = project_density_matrix(rho)
        trajectory[i] = rho
    return trajectory


def dephasing_collapse(dim: int, index: int, rate: float) -> Array:
    """Construct sqrt(rate)|index><index|, a simple pure-dephasing collapse operator."""

    if dim <= 0 or not 0 <= index < dim:
        raise ValueError("invalid dimension/index")
    if rate < 0:
        raise ValueError("rate must be non-negative")
    operator = np.zeros((dim, dim), dtype=complex)
    operator[index, index] = np.sqrt(rate)
    return operator


def trajectory_is_physical(trajectory: Array, *, atol: float = 1e-8) -> bool:
    """Return True when every element is a valid density matrix."""

    return all(is_density_matrix(rho, atol=atol) for rho in np.asarray(trajectory))
