"""Compatibility surface for the original QuantumBCI Kalman demo.

The old implementation labelled ``np.linalg.inv`` as a quantum-enhanced inversion and
contained a Qiskit Aqua HHL prototype. Aqua has been retired, so this module now keeps a
strong classical baseline and requires an explicit solver injection for experimental
linear-system backends.
"""

from __future__ import annotations

import warnings

import numpy as np

from quantumbci.kalman import kalman_filter, qlsa_diagnostics


def classical_kalman_filter(zs, A, H, Q, R, x0, P0):
    return kalman_filter(zs, A, H, Q, R, x0, P0)


def quantum_matrix_inversion(M):
    """Deprecated compatibility helper; this is classical and makes no quantum claim."""

    warnings.warn(
        "quantum_matrix_inversion is a legacy name for np.linalg.inv and is not quantum.",
        DeprecationWarning,
        stacklevel=2,
    )
    return np.linalg.inv(M)


def quantum_matrix_inversion_hhl(M):
    """Refuse the retired Aqua path rather than silently presenting stale code as valid."""

    report = qlsa_diagnostics(M)
    raise NotImplementedError(
        "Qiskit Aqua HHL was retired. QuantumBCI now treats QLSA/HHL as a resource-accounted "
        "research backend, not a drop-in matrix inverse. Inspect qlsa_diagnostics(M) first. "
        f"Candidate diagnostics: {report}"
    )


def quantum_kalman_filter(zs, A, H, Q, R, x0, P0, *, linear_solve=None):
    """Run the filter with an explicitly provided experimental linear-system solver.

    A solver is required so a classical solve can never be silently relabelled quantum.
    """

    if linear_solve is None:
        raise ValueError(
            "Provide linear_solve(S, B). QuantumBCI intentionally has no fake default "
            "'quantum' inversion. Use classical_kalman_filter for the baseline."
        )
    return kalman_filter(zs, A, H, Q, R, x0, P0, linear_solve=linear_solve)
