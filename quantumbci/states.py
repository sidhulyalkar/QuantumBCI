"""Density-operator utilities for quantum-inspired neural representations."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def project_density_matrix(matrix: Array, *, atol: float = 1e-12) -> Array:
    """Project a square matrix onto the set of valid density matrices.

    This is a numerical projection, not a claim that the represented neural state is a
    microscopic quantum state.
    """

    rho = np.asarray(matrix, dtype=complex)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("matrix must be square")
    rho = (rho + rho.conj().T) / 2
    values, vectors = np.linalg.eigh(rho)
    values = np.clip(values.real, 0.0, None)
    total = float(values.sum())
    if total <= atol:
        raise ValueError("matrix has no positive trace after projection")
    rho = (vectors * values) @ vectors.conj().T
    rho /= np.trace(rho)
    return (rho + rho.conj().T) / 2


def density_from_samples(samples: Array, *, center: bool = True) -> Array:
    """Construct a trace-one PSD feature-state from samples x features data.

    The result is a normalized second-moment/covariance operator. Treating it as a
    density matrix grants useful geometry and observables while remaining explicitly
    quantum-inspired unless the features themselves come from a physical quantum system.
    """

    x = np.asarray(samples, dtype=complex)
    if x.ndim != 2:
        raise ValueError("samples must have shape (n_samples, n_features)")
    if x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("samples must be non-empty")
    if center:
        x = x - x.mean(axis=0, keepdims=True)
    second_moment = x.conj().T @ x
    trace = float(np.trace(second_moment).real)
    if trace <= 0:
        raise ValueError("samples have zero variance/norm")
    return project_density_matrix(second_moment / trace)


def is_density_matrix(rho: Array, *, atol: float = 1e-9) -> bool:
    """Check Hermiticity, trace normalization, and positive semidefiniteness."""

    x = np.asarray(rho, dtype=complex)
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        return False
    if not np.allclose(x, x.conj().T, atol=atol):
        return False
    if not np.isclose(np.trace(x).real, 1.0, atol=atol):
        return False
    return bool(np.linalg.eigvalsh(x).min() >= -atol)


def purity(rho: Array) -> float:
    """Return Tr(rho^2), in [1/d, 1] for a d-dimensional density state."""

    x = np.asarray(rho, dtype=complex)
    return float(np.trace(x @ x).real)


def von_neumann_entropy(rho: Array, *, base: float = 2.0) -> float:
    """Return -Tr(rho log rho), using the requested logarithm base."""

    if base <= 0 or np.isclose(base, 1.0):
        raise ValueError("base must be positive and not equal to one")
    values = np.linalg.eigvalsh(np.asarray(rho, dtype=complex)).real
    values = values[values > 1e-15]
    return float(-(values * np.log(values)).sum() / np.log(base))


def l1_coherence(rho: Array) -> float:
    """Return the basis-dependent L1 norm of off-diagonal entries."""

    x = np.asarray(rho, dtype=complex)
    return float(np.abs(x - np.diag(np.diag(x))).sum())


def expectation(rho: Array, observable: Array) -> complex:
    """Return Tr(rho O) for an observable/operator O."""

    x = np.asarray(rho, dtype=complex)
    op = np.asarray(observable, dtype=complex)
    if x.shape != op.shape:
        raise ValueError("rho and observable must have the same shape")
    return complex(np.trace(x @ op))
