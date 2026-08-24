import numpy as np

from quantumbci.states import (
    density_from_samples,
    is_density_matrix,
    l1_coherence,
    purity,
    von_neumann_entropy,
)


def test_density_from_samples_is_physical():
    rng = np.random.default_rng(0)
    rho = density_from_samples(rng.normal(size=(128, 4)))
    assert is_density_matrix(rho)
    assert 0.25 - 1e-9 <= purity(rho) <= 1.0 + 1e-9
    assert 0.0 <= von_neumann_entropy(rho) <= 2.0 + 1e-9
    assert l1_coherence(rho) >= 0.0


def test_correlated_features_create_off_diagonal_structure():
    x = np.linspace(-1.0, 1.0, 100)
    rho = density_from_samples(np.column_stack([x, x]))
    assert l1_coherence(rho) > 0.9
