"""
qfft_module.py

This module demonstrates both classical and quantum Fourier transforms.
It includes:
- A function (qft) to apply the Quantum Fourier Transform (QFT) on a quantum circuit.
- A function (generate_qft_circuit) to create a QFT circuit for a given number of qubits.
- A function (classical_fft) to compute the classical FFT on an input signal.
- A function (simulate_qft_circuit) to simulate the QFT circuit using Qiskit's Aer simulator.
- Example code in the main block to generate, draw, and simulate the QFT circuit,
  and to compute and plot the FFT of simulated EEG-like (sine wave) data.
"""

from qiskit import QuantumCircuit, transpile, assemble
try:
    # Try to import from the new namespace first.
    from qiskit.providers.aer import Aer
except ImportError:
    # Fallback: try to import directly from qiskit_aer.
    from qiskit_aer import Aer
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt

def qft(circuit, n):
    """Applies the Quantum Fourier Transform on the first n qubits of the circuit."""
    for j in range(n):
        circuit.h(j)  # Apply Hadamard gate to qubit j
        for k in range(j+1, n):
            angle = np.pi / (2 ** (k - j))
            circuit.cp(angle, k, j)  # Apply controlled phase shift
    # Swap qubits to reverse their order
    for i in range(n//2):
        circuit.swap(i, n-i-1)

def generate_qft_circuit(n):
    """
    Creates a QFT circuit for n qubits.
    
    The function builds the circuit by applying the QFT
    and then adds measurement operations to all qubits.
    """
    qc = QuantumCircuit(n)
    qft(qc, n)
    qc.measure_all()
    return qc

def classical_fft(signal):
    """Computes the Fast Fourier Transform (FFT) of the given signal."""
    return np.abs(fft(signal))

def simulate_qft_circuit(circuit, shots=1024):
    """
    Simulates the given quantum circuit using Qiskit's Aer QASM simulator.
    
    Parameters:
      circuit (QuantumCircuit): The quantum circuit to simulate.
      shots (int): The number of simulation repetitions (default: 1024).
    
    Returns:
      dict: A dictionary with measurement counts from the simulation.
    """
    backend = Aer.get_backend('qasm_simulator')
    # Transpile the circuit for the backend.
    compiled_circuit = transpile(circuit, backend)
    # Run the transpiled circuit directly (without calling assemble).
    job = backend.run(compiled_circuit, shots=shots)
    result = job.result()
    counts = result.get_counts(0)
    return counts

def quantum_fft(signal, shots=4096):
    """
    Encodes a classical signal into a quantum state, applies the QFT,
    and returns the measurement probability distribution.
    
    Parameters:
      signal (array): Classical signal (length must be a power of 2).
      shots (int): Number of simulation shots.
    
    Returns:
      probabilities (ndarray): Array of probabilities corresponding to each basis state.
    """
    n = int(np.log2(len(signal)))
    if 2**n != len(signal):
        raise ValueError("Length of signal must be a power of 2")
    # Normalize the signal to prepare a quantum state.
    norm_val = np.linalg.norm(signal)
    state = signal / norm_val
    # Create a quantum circuit on n qubits.
    qc = QuantumCircuit(n)
    qc.initialize(state, list(range(n)))
    # Apply QFT using our module’s function.
    qft(qc, n)
    qc.measure_all()
    # Simulate the circuit.
    counts = simulate_qft_circuit(qc, shots)
    # Convert counts to a probability distribution.
    probabilities = np.zeros(len(signal))
    for bitstr, count in counts.items():
        index = int(bitstr, 2)  # Convert binary string to integer index.
        probabilities[index] = count / shots
    return probabilities

# Example usage and demonstration of the functions.
if __name__ == "__main__":
    # Define number of qubits for the QFT circuit.
    n_qubits = 4
    
    # Generate and draw the QFT circuit.
    qft_circuit = generate_qft_circuit(n_qubits)
    qft_circuit.draw('mpl')  # This displays the circuit diagram when running in a Jupyter notebook.
    
    # Simulate the QFT circuit and print the measurement results.
    counts = simulate_qft_circuit(qft_circuit)
    print("Simulation result (measurement counts):")
    print(counts)
    
    # Generate simulated EEG-like data (sine wave).
    signal = np.sin(2 * np.pi * np.linspace(0, 1, 16))
    
    # Compute the classical FFT of the signal.
    fft_result = classical_fft(signal)
    
    # Plot the classical FFT result.
    plt.figure(figsize=(8, 4))
    plt.plot(fft_result, label='Classical FFT')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title('Classical FFT of Simulated EEG-like Data')
    plt.legend()
    plt.show()
