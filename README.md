# Quantum-Enhanced Signal Processing for Brain-Computer Interfaces

This project demonstrates the integration of quantum and classical signal processing techniques for brain-computer interface (BCI) applications. It includes modules for generating and simulating quantum Fourier transforms (QFT) and for implementing both classical and quantum-enhanced Kalman filters. These tools explore how quantum algorithms could eventually enhance the processing and tracking of neural signals.

## File Structure

project/ ├── qfft_module.py ├── qkalman_module.py ├── QFT_Notebook.ipynb ├── KalmanFilter_Notebook.ipynb └── README.md


## Module Descriptions

### `qfft_module.py`
- **Purpose:**  
  Implements functions to:
  - Generate a Quantum Fourier Transform (QFT) circuit using Qiskit.
  - Compute a classical Fast Fourier Transform (FFT) on simulated EEG-like signals.
  - Compare QFT simulation outcomes with classical FFT results.

- **Key Functions:**  
  - `generate_qft_circuit(n)`: Creates an n-qubit QFT circuit with measurements.
  - `classical_fft(signal)`: Computes the FFT of a signal.
  - `simulate_qft_circuit(circuit, shots)`: Simulates a given QFT circuit using Qiskit’s Aer simulator.

### `qkalman_module.py`
- **Purpose:**  
  Provides implementations for:
  - A classical Kalman filter for tracking neural signal changes.
  - A quantum-enhanced Kalman filter where the only difference is in the matrix inversion step.

- **Key Functions:**  
  - `classical_kalman_filter(zs, A, H, Q, R, x0, P0)`: Runs the standard Kalman filter.
  - `quantum_matrix_inversion(M, method="classical")`: By default, uses `np.linalg.inv(M)` but can be configured to use a prototype HHL-based inversion (`quantum_matrix_inversion_hhl`) in the future.
  - `quantum_kalman_filter(zs, A, H, Q, R, x0, P0, inversion_method="classical")`: Runs the Kalman filter using the selected inversion method.

## Notebook Descriptions

### `QFT_Notebook.ipynb`
- **Overview:**  
  Demonstrates how to:
  - Generate and visualize a QFT circuit.
  - Simulate the QFT circuit using Qiskit's Aer simulator.
  - Compute and compare a classical FFT on simulated EEG-like data.

- **Highlights:**  
  - Visualizations of the QFT circuit and FFT plots.
  - Side-by-side comparison of quantum and classical Fourier transform methods.

### `KalmanFilter_Notebook.ipynb`
- **Overview:**  
  Illustrates the tracking of complex, realistic EEG-like signals using:
  - A classical Kalman filter.
  - A quantum-enhanced Kalman filter (with a placeholder for quantum matrix inversion).

- **Highlights:**  
  - Simulation of realistic EEG amplitude envelopes featuring baseline modulation and transient spikes.
  - Visual comparisons of the true signal, noisy measurements, and estimates from both filters.
  - Performance evaluations using metrics such as MSE, RMSE, MAE, and Pearson correlation.
  - Additional plots (time-series, scatter plots, and bar charts) to assess and compare filter performance.

## Requirements & Installation

This project requires:
- Python 3.9+
- Qiskit and Qiskit-Aer (for quantum circuit simulation)
- NumPy, SciPy, Matplotlib, and Pandas
- pylatexenc (for LaTeX rendering in Qiskit circuits)

You can install the necessary dependencies using pip or conda:

```bash
pip install qiskit qiskit-aer numpy scipy matplotlib pandas pylatexenc

conda install -c conda-forge qiskit qiskit-aer numpy scipy matplotlib pandas pylatexenc
```

## How to Run
Place all files in the same directory.
Run the Notebooks:
Open QFT_Notebook.ipynb in Jupyter Notebook or JupyterLab and execute the cells sequentially to view the QFT and FFT demonstrations.
Open KalmanFilter_Notebook.ipynb to simulate EEG signals and apply both the classical and quantum-enhanced Kalman filters. Remember to reload the modules (qfft_module.py and qkalman_module.py) if you make changes.

## Notes
### Quantum-Enhanced Kalman Filter:
Currently, the quantum-enhanced Kalman filter uses a placeholder for the quantum matrix inversion step (i.e., it calls the classical inversion routine). A prototype version based on the HHL algorithm is provided as an example, but practical benefits will be seen only when running on a true quantum device with a suitable quantum subroutine.

### QFT Simulations:
The QFT simulations use Qiskit's Aer simulator. Adjust your Qiskit installation if you encounter module import issues.
