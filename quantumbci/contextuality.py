"""Non-commuting measurement probes for quantum-like contextuality experiments."""

from __future__ import annotations

import numpy as np

from .states import expectation, project_density_matrix

Array = np.ndarray


def projector(vector: Array) -> Array:
    """Return |v><v| for a normalized version of vector."""

    v = np.asarray(vector, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(v))
    if norm == 0:
        raise ValueError("projector vector must be non-zero")
    v = v / norm
    return np.outer(v, v.conj())


def commutator_norm(a: Array, b: Array) -> float:
    """Frobenius norm of [A, B] = AB - BA."""

    x = np.asarray(a, dtype=complex)
    y = np.asarray(b, dtype=complex)
    if x.shape != y.shape or x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError("operators must be same-shape square matrices")
    return float(np.linalg.norm(x @ y - y @ x, ord="fro"))


def lueders_update(rho: Array, measurement_projector: Array) -> tuple[float, Array]:
    """Apply a projective Lüders update and return (probability, posterior state)."""

    state = np.asarray(rho, dtype=complex)
    p = np.asarray(measurement_projector, dtype=complex)
    probability = float(expectation(state, p).real)
    if probability <= 1e-15:
        raise ValueError("measurement outcome has zero probability")
    posterior = p @ state @ p / probability
    return probability, project_density_matrix(posterior)


def sequential_joint_probability(rho: Array, first: Array, second: Array) -> float:
    """Return p(first=yes, then second=yes) under projective measurement."""

    p_first, posterior = lueders_update(rho, first)
    p_second_given_first = float(expectation(posterior, second).real)
    return float(p_first * p_second_given_first)


def order_effect(rho: Array, a: Array, b: Array) -> dict[str, float]:
    """Compare sequential AB and BA joint probabilities.

    A non-zero value is evidence that this *model* is order-sensitive. It is not, by
    itself, evidence that the underlying brain tissue is physically quantum.
    """

    p_ab = sequential_joint_probability(rho, a, b)
    p_ba = sequential_joint_probability(rho, b, a)
    return {"p_ab": p_ab, "p_ba": p_ba, "difference": p_ab - p_ba}
