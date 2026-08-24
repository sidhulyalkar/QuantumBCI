"""Classical Fourier and ideal QFT-state utilities with explicit semantics."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def classical_fft(signal: Array) -> Array:
    """Return the full complex classical FFT, preserving phase information."""

    x = np.asarray(signal, dtype=complex)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("signal must be a non-empty 1D array")
    return np.fft.fft(x)


def amplitude_encode(signal: Array) -> Array:
    """Normalize a power-of-two vector as a quantum state amplitude vector."""

    x = np.asarray(signal, dtype=complex)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("signal must be a non-empty 1D array")
    if x.size & (x.size - 1):
        raise ValueError("amplitude encoding requires a power-of-two vector length")
    norm = float(np.linalg.norm(x))
    if norm == 0:
        raise ValueError("cannot amplitude-encode the zero vector")
    return x / norm


def qft_state(signal: Array) -> Array:
    """Return the exact state after the standard positive-phase QFT.

    NumPy's inverse FFT uses the same positive phase convention with a 1/N
    normalization; multiplying by sqrt(N) gives the unitary QFT normalization.
    """

    state = amplitude_encode(signal)
    return np.fft.ifft(state) * np.sqrt(state.size)


def qft_probabilities(signal: Array) -> Array:
    """Return exact computational-basis probabilities after QFT.

    These probabilities intentionally contain less information than a complex FFT:
    measurement removes Fourier phase unless a different measurement protocol is used.
    """

    transformed = qft_state(signal)
    probabilities = np.abs(transformed) ** 2
    probabilities /= probabilities.sum()
    return probabilities.real


def sample_probabilities(
    probabilities: Array, *, shots: int = 4096, seed: int | None = None
) -> Array:
    """Sample a categorical distribution to emulate finite-shot measurement."""

    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("probabilities must be a non-empty 1D array")
    if shots <= 0:
        raise ValueError("shots must be positive")
    if np.any(p < 0) or not np.isclose(p.sum(), 1.0, atol=1e-9):
        raise ValueError("probabilities must be non-negative and sum to one")
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(shots, p)
    return counts / shots
