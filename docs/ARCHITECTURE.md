# Architecture

QuantumBCI is intentionally a small scientific kernel. Heavy EEG loaders, foundation models, and
quantum SDKs should remain adapters/extras so the mathematical core is auditable.

```text
raw neural data / frozen foundation tokens
                |
                v
        observation adapters
                |
      +---------+----------+
      |                    |
      v                    v
classical controls   mechanism representations
Kalman / FFT         density / contextual / Lindblad
      |                    |
      +---------+----------+
                v
       matched simple readout
                |
                v
 predictive + mechanistic ledgers
```

## Modules

- `claims.py`: claim classes and falsification contracts.
- `spectral.py`: complex FFT plus ideal QFT state/probability semantics.
- `states.py`: density-state construction and observables.
- `open_system.py`: inspectable Lindblad dynamics.
- `contextuality.py`: non-commuting projective measurement/order probes.
- `kalman.py`: strong classical state-estimation baseline and QLSA diagnostics.
- `foundation.py`: dependency-free bridge from pretrained latent tokens to mechanism states.
- `interpretability.py`: state signatures, ablations, and stability probes.
- `signals.py`: deterministic synthetic test signals only.

## Dependency policy

The core depends only on NumPy. Qiskit is optional because a research claim should be testable and
reviewable without installing a quantum SDK. EEG ecosystem dependencies (MNE, PyTorch, specific
foundation models) belong in future adapter extras, not the core.
