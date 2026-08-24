import numpy as np

from quantumbci.open_system import dephasing_collapse, evolve_lindblad, trajectory_is_physical
from quantumbci.states import l1_coherence, project_density_matrix


def test_lindblad_trajectory_stays_physical():
    rho0 = project_density_matrix(np.array([[0.5, 0.45], [0.45, 0.5]], dtype=complex))
    h = np.zeros((2, 2), dtype=complex)
    collapse = [dephasing_collapse(2, 0, 1.0), dephasing_collapse(2, 1, 1.0)]
    trajectory = evolve_lindblad(
        rho0, h, np.linspace(0.0, 2.0, 201), collapse_operators=collapse
    )
    assert trajectory_is_physical(trajectory)
    assert l1_coherence(trajectory[-1]) < l1_coherence(trajectory[0])
