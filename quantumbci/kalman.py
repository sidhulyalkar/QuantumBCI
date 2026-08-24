"""Numerically stable classical Kalman baseline and QLSA suitability diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Array = np.ndarray
LinearSolve = Callable[[Array, Array], Array]


@dataclass(frozen=True)
class QLSADiagnostics:
    """Resource-relevant facts for a candidate quantum linear-system subproblem."""

    dimension: int
    hermitian: bool
    positive_definite: bool
    condition_number: float
    nonzero_fraction: float
    power_of_two: bool
    caveats: tuple[str, ...]


def qlsa_diagnostics(matrix: Array, *, atol: float = 1e-10) -> QLSADiagnostics:
    """Inspect assumptions that matter to HHL/QLSA-style algorithms.

    This does not estimate quantum speedup. Data loading, Hamiltonian access, circuit
    depth, accuracy, and output/readout cost remain part of any end-to-end claim.
    """

    m = np.asarray(matrix, dtype=complex)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("matrix must be square")
    dim = m.shape[0]
    hermitian = bool(np.allclose(m, m.conj().T, atol=atol))
    positive_definite = False
    if hermitian:
        positive_definite = bool(np.linalg.eigvalsh(m).min() > atol)
    condition = float(np.linalg.cond(m))
    nonzero_fraction = float(np.count_nonzero(np.abs(m) > atol) / m.size)
    power_of_two = dim > 0 and (dim & (dim - 1) == 0)

    caveats: list[str] = []
    if not hermitian:
        caveats.append("A direct HHL-style formulation requires Hermitian embedding.")
    if condition > 100:
        caveats.append("Large condition number can erase theoretical QLSA advantages.")
    if nonzero_fraction > 0.2:
        caveats.append("The matrix is relatively dense; efficient sparse access is not established.")
    if not power_of_two:
        caveats.append("Amplitude-register implementations require padding/embedding.")
    caveats.append("Full solution-vector readout can erase an asymptotic quantum advantage.")
    caveats.append("State preparation and matrix-oracle costs must be included end to end.")

    return QLSADiagnostics(
        dimension=dim,
        hermitian=hermitian,
        positive_definite=positive_definite,
        condition_number=condition,
        nonzero_fraction=nonzero_fraction,
        power_of_two=power_of_two,
        caveats=tuple(caveats),
    )


def kalman_filter(
    measurements: Array,
    a: Array,
    h: Array,
    q: Array,
    r: Array,
    x0: Array,
    p0: Array,
    *,
    linear_solve: LinearSolve | None = None,
) -> tuple[Array, Array]:
    """Run a linear Kalman filter using solves and Joseph covariance updates.

    ``linear_solve`` receives (S, B) and must return X satisfying S @ X = B. The
    injectable boundary is intentional: a future quantum linear-system estimator can be
    benchmarked without relabeling the classical baseline as quantum.
    """

    z = np.asarray(measurements, dtype=float)
    a = np.asarray(a, dtype=float)
    h = np.asarray(h, dtype=float)
    q = np.asarray(q, dtype=float)
    r = np.asarray(r, dtype=float)
    x = np.asarray(x0, dtype=float).copy()
    p = np.asarray(p0, dtype=float).copy()
    if z.ndim == 1:
        z = z[:, None]
    state_dim = x.size
    if a.shape != (state_dim, state_dim) or p.shape != (state_dim, state_dim):
        raise ValueError("A/P0 dimensions must match x0")
    if h.shape[1] != state_dim or z.shape[1] != h.shape[0]:
        raise ValueError("measurement/H dimensions do not match")
    if q.shape != p.shape or r.shape != (h.shape[0], h.shape[0]):
        raise ValueError("Q/R dimensions are inconsistent")

    solve = np.linalg.solve if linear_solve is None else linear_solve
    estimates = np.empty((len(z), state_dim), dtype=float)
    covariances = np.empty((len(z), state_dim, state_dim), dtype=float)
    identity = np.eye(state_dim)

    for i, measurement in enumerate(z):
        x_pred = a @ x
        p_pred = a @ p @ a.T + q
        innovation = h @ p_pred @ h.T + r
        gain = solve(innovation, (p_pred @ h.T).T).T
        residual = measurement - h @ x_pred
        x = x_pred + gain @ residual
        i_kh = identity - gain @ h
        p = i_kh @ p_pred @ i_kh.T + gain @ r @ gain.T
        p = (p + p.T) / 2
        estimates[i] = x
        covariances[i] = p

    return estimates, covariances
