import numpy as np
import pytest

from quantumbci.spectral import amplitude_encode, qft_probabilities, qft_state, sample_probabilities


def test_qft_is_unitary_and_probabilities_normalize():
    signal = np.array([1.0, 2.0, -1.0, 0.5])
    state = amplitude_encode(signal)
    transformed = qft_state(signal)
    probabilities = qft_probabilities(signal)
    assert np.isclose(np.linalg.norm(state), 1.0)
    assert np.isclose(np.linalg.norm(transformed), 1.0)
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.all(probabilities >= 0)


def test_finite_shots_approach_exact_distribution():
    p = qft_probabilities(np.array([1.0, 0.0, 0.0, 0.0]))
    sampled = sample_probabilities(p, shots=100_000, seed=3)
    assert np.max(np.abs(sampled - p)) < 0.01


def test_amplitude_encoding_rejects_non_power_of_two_and_zero():
    with pytest.raises(ValueError):
        amplitude_encode(np.ones(3))
    with pytest.raises(ValueError):
        amplitude_encode(np.zeros(4))
