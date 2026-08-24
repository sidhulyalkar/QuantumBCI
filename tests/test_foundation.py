import numpy as np

from quantumbci.foundation import density_states_from_embeddings
from quantumbci.states import is_density_matrix


def test_density_adapter_accepts_foundation_tokens():
    rng = np.random.default_rng(5)
    embeddings = rng.normal(size=(3, 24, 6))
    states = density_states_from_embeddings(embeddings)
    assert states.shape == (3, 6, 6)
    assert all(is_density_matrix(state) for state in states)
