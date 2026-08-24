"""Compatibility surface for the original QuantumBCI Fourier demo.

New research code should import :mod:`quantumbci.spectral` directly. The key semantic
change is that QFT measurement probabilities are no longer described as equivalent to a
classical FFT: they discard Fourier phase unless an additional measurement protocol is
used.
"""

from __future__ import annotations

import numpy as np

from quantumbci.spectral import classical_fft as _complex_fft
from quantumbci.spectral import qft_probabilities, sample_probabilities


def qft(circuit, n: int):
    """Apply an in-place QFT to the first ``n`` qubits of a Qiskit circuit."""

    for target in range(n):
        circuit.h(target)
        for control in range(target + 1, n):
            circuit.cp(np.pi / (2 ** (control - target)), control, target)
    for i in range(n // 2):
        circuit.swap(i, n - i - 1)
    return circuit


def generate_qft_circuit(n: int, *, measure: bool = True):
    """Build a QFT circuit lazily so base QuantumBCI does not require Qiskit."""

    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise ImportError("Install QuantumBCI with `pip install -e '.[quantum]'` for Qiskit support") from exc
    circuit = QuantumCircuit(n)
    qft(circuit, n)
    if measure:
        circuit.measure_all()
    return circuit


def classical_fft(signal):
    """Legacy magnitude-only FFT helper. Prefer quantumbci.spectral.classical_fft."""

    return np.abs(_complex_fft(signal))


def quantum_fft(signal, shots: int | None = 4096, *, seed: int | None = None):
    """Return ideal or finite-shot QFT measurement probabilities.

    This NumPy path is a reference simulator and makes no quantum-speedup claim.
    """

    exact = qft_probabilities(signal)
    if shots is None:
        return exact
    return sample_probabilities(exact, shots=shots, seed=seed)


def simulate_qft_circuit(circuit, shots: int = 1024, *, seed_simulator: int | None = None):
    """Sample an explicitly supplied Qiskit circuit with Aer."""

    try:
        from qiskit import transpile
        from qiskit_aer import AerSimulator
    except ImportError as exc:
        raise ImportError("Install QuantumBCI with `pip install -e '.[quantum]'` for Aer support") from exc
    backend = AerSimulator(seed_simulator=seed_simulator)
    compiled = transpile(circuit, backend)
    result = backend.run(compiled, shots=shots).result()
    return result.get_counts(compiled)
