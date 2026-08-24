import numpy as np
import pytest

from quantumbci.integrations.neuros_mechint import (
    MixWithMaximallyMixedState,
    PermuteDensityBasis,
    RemoveDensityOffDiagonals,
)
from quantumbci.states import is_density_matrix, purity, von_neumann_entropy


def _density() -> np.ndarray:
    value = np.asarray(
        [
            [0.45, 0.18 + 0.04j, 0.02],
            [0.18 - 0.04j, 0.35, 0.06],
            [0.02, 0.06, 0.20],
        ],
        dtype=complex,
    )
    values, vectors = np.linalg.eigh(value)
    values = np.clip(values, 1e-6, None)
    value = (vectors * values) @ vectors.conj().T
    return value / np.trace(value)


def test_remove_off_diagonals_preserves_trace_and_diagonal() -> None:
    rho = _density()
    edited = RemoveDensityOffDiagonals().apply(rho)
    assert np.allclose(np.diag(edited), np.diag(rho))
    assert np.isclose(np.trace(edited), 1.0)
    assert np.allclose(edited - np.diag(np.diag(edited)), 0.0)


def test_basis_permutation_preserves_spectrum_purity_and_entropy() -> None:
    rho = _density()
    edited = PermuteDensityBasis((2, 0, 1)).apply(rho)
    assert is_density_matrix(edited)
    assert np.allclose(np.linalg.eigvalsh(edited), np.linalg.eigvalsh(rho))
    assert np.isclose(purity(edited), purity(rho))
    assert np.isclose(von_neumann_entropy(edited), von_neumann_entropy(rho))


def test_maximally_mixed_intervention_has_monotone_purity_dose_response() -> None:
    rho = _density()
    quarter = MixWithMaximallyMixedState(0.25).apply(rho)
    half = MixWithMaximallyMixedState(0.5).apply(rho)
    full = MixWithMaximallyMixedState(1.0).apply(rho)
    assert is_density_matrix(quarter)
    assert is_density_matrix(half)
    assert is_density_matrix(full)
    assert purity(rho) >= purity(quarter) >= purity(half) >= purity(full)
    assert np.isclose(purity(full), 1 / rho.shape[0])


def test_basis_permutation_validates_complete_permutation() -> None:
    with pytest.raises(ValueError, match="each index"):
        PermuteDensityBasis((0, 0, 2))
