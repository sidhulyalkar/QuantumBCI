"""Run a tiny deterministic mechanism probe with ``python -m quantumbci``."""

from __future__ import annotations

import numpy as np

from .contextuality import order_effect, projector
from .interpretability import mechanism_delta, state_signature
from .open_system import dephasing_collapse, evolve_lindblad
from .states import project_density_matrix


def main() -> None:
    rho0 = project_density_matrix(np.array([[0.5, 0.45], [0.45, 0.5]], dtype=complex))
    hamiltonian = np.array([[0.0, 0.8], [0.8, 0.2]], dtype=complex)
    collapse = [dephasing_collapse(2, 0, 0.7), dephasing_collapse(2, 1, 0.7)]
    trajectory = evolve_lindblad(
        rho0, hamiltonian, np.linspace(0.0, 2.0, 201), collapse_operators=collapse
    )
    before = state_signature(trajectory[0])
    after = state_signature(trajectory[-1])
    a = projector([1.0, 0.0])
    b = projector([1.0, 1.0])
    print("Lindblad mechanism delta:", mechanism_delta(before, after))
    print("Context/order probe:", order_effect(rho0, a, b))


if __name__ == "__main__":
    main()
