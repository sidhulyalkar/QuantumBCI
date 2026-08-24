import numpy as np

from quantumbci.contextuality import commutator_norm, order_effect, projector
from quantumbci.states import project_density_matrix


def test_noncommuting_projectors_have_nonzero_commutator():
    z = projector([1.0, 0.0])
    x = projector([1.0, 1.0])
    assert commutator_norm(z, x) > 0.0


def test_order_effect_is_exposed_not_interpreted_as_physics():
    rho = project_density_matrix(np.array([[0.8, 0.2], [0.2, 0.2]], dtype=complex))
    a = projector([1.0, 0.0])
    b = projector([1.0, 1.0])
    effect = order_effect(rho, a, b)
    assert set(effect) == {"p_ab", "p_ba", "difference"}
    assert np.isfinite(effect["difference"])
