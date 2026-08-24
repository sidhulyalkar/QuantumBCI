# QuantumBCI

**A falsifiable workbench for quantum, quantum-inspired, and classical models of neural signals.**

QuantumBCI began as a small QFT/Kalman demonstration. v0.2 turns it into a research kernel for a
harder question: **does a specific quantum-structured mechanism add identifiable, reproducible
value to neural modelling after strong classical alternatives are tested?**

The project deliberately separates four ideas that are often blurred together:

1. **Classical controls** such as FFT and Kalman/state-space models.
2. **Quantum-inspired models** that use density operators, non-commuting observables, or open-system
   dynamics as mathematical inductive biases without claiming the brain is physically quantum.
3. **Quantum algorithms** such as QFT/QLSA, whose value must include state preparation, circuit,
   sampling, noise, and readout costs.
4. **Physical quantum neural hypotheses**, which require independent operational evidence about the
   biological substrate. Model fit is not that evidence.

> **Current scientific stance:** this repository implements useful quantum mathematics and quantum
> algorithm reference paths, but it does **not** claim that neural tissue exhibits biologically
> functional entanglement, long-lived coherence, or a demonstrated quantum computational advantage.

## Why this direction

Modern BCI modelling now includes pretrained EEG representations such as LaBraM, EEGPT, BrainWave,
and NeuroLM. A useful QuantumBCI experiment should therefore ask whether a quantum-structured layer
adds **incremental representation or mechanism value** on identical raw data or frozen embeddings,
not whether it can outperform a toy sine-wave baseline.

The most interesting near-term hypotheses are:

- **Density geometry:** are trace-one PSD latent states and their observables useful across subjects?
- **Open-system dynamics:** do interpretable Hamiltonian/collapse parameters capture latent neural
  transitions better than LDS/Kalman/neural-ODE controls?
- **Contextuality:** do non-commuting measurement models predict preregistered cue/order effects more
  compactly than history-aware classical models?
- **Quantum algorithms:** can a *specific observable-level* neural computation justify end-to-end
  QFT/QLSA/variational-circuit resources?
- **Physical mechanisms:** can an experiment operationally distinguish a proposed non-classical
  biological mechanism from classical nonlinear/stochastic dynamics?

See [the research agenda](docs/RESEARCH_AGENDA.md), [mechanism cards](docs/MECHANISM_CARDS.md), and
[interpretability protocol](docs/INTERPRETABILITY_PROTOCOL.md).

## Research kernel

```text
quantumbci/
├── claims.py            # claim classes + falsification contracts
├── spectral.py          # complex FFT + correct ideal-QFT measurement semantics
├── states.py            # density operators, purity, entropy, coherence
├── open_system.py       # transparent Lindblad dynamics
├── contextuality.py     # non-commuting operators and order effects
├── kalman.py            # stable classical Kalman + QLSA suitability diagnostics
├── foundation.py        # frozen foundation-token -> density-state bridge
├── interpretability.py  # mechanism signatures, ablations, stability
└── signals.py           # deterministic synthetic test signals
```

The original `qfft_module.py` and `qkalman_module.py` remain as compatibility surfaces, but their
scientific semantics are corrected. In particular:

- QFT computational-basis probabilities are no longer presented as equivalent to a complex FFT;
  phase is lost unless a richer measurement protocol is used.
- a classical `np.linalg.inv` is never called “quantum-enhanced”;
- the retired Qiskit Aqua HHL implementation is not used as an active backend;
- experimental linear-system solvers must be explicitly injected and resource-accounted.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python -m quantumbci
```

Optional Qiskit/Aer support:

```bash
pip install -e '.[quantum]'
```

The base package intentionally depends only on NumPy.

## Example: an interpretable open-system probe

```python
import numpy as np
from quantumbci.open_system import dephasing_collapse, evolve_lindblad
from quantumbci.states import density_from_samples, l1_coherence

latents = np.random.default_rng(0).normal(size=(128, 2))
rho0 = density_from_samples(latents)
H = np.array([[0.0, 0.8], [0.8, 0.2]], dtype=complex)
collapse = [dephasing_collapse(2, 0, 0.5), dephasing_collapse(2, 1, 0.5)]
trajectory = evolve_lindblad(rho0, H, np.linspace(0, 1, 101), collapse_operators=collapse)
print(l1_coherence(trajectory[0]), l1_coherence(trajectory[-1]))
```

This demonstrates the *model mechanism*. It does not assert that the latent state is a microscopic
quantum state.

## Foundation-model integration

`quantumbci.foundation.density_states_from_embeddings` accepts `(batch, tokens, features)` arrays,
so the same quantum-inspired probe can sit on frozen LaBraM, EEGPT, BrainWave, NeuroLM, specialist,
or random-control embeddings without coupling the core library to a particular deep-learning stack.

The recommended experiment is paired: **same subjects, same splits, same frozen embeddings, same
readout, different representation layer.** Then run off-diagonal ablations and bootstrap the
mechanism observables.

## Validation philosophy

A result is interesting only if it survives three ledgers:

- **Mathematical:** Hermiticity/PSD/trace, normalization, numerical stability, circuit semantics.
- **Predictive:** held-out subjects/sessions, calibration, transfer, data efficiency, compute.
- **Mechanistic:** parameter recovery, intervention prediction, identifiability, stability, matched
  classical alternatives, explicit falsifiers.

The test suite already enforces the first layer for the core primitives. The next release should
build the benchmark/manifest harness for the latter two.

## Reading context

- Quantum cognition is a well-developed use of quantum probability **without requiring quantum
  brain physics**: Pothos & Busemeyer (2022), https://pubmed.ncbi.nlm.nih.gov/34546804/
- A 2025 overview makes the same quantum-probability-versus-physics distinction:
  https://pubmed.ncbi.nlm.nih.gov/40608277/
- Recent work explicitly explores bridges from oscillatory neural networks to quantum-like states:
  https://pubmed.ncbi.nlm.nih.gov/40889614/ and https://pubmed.ncbi.nlm.nih.gov/41446506/
- LaBraM (ICLR 2024): https://openreview.net/forum?id=QzTpTRVtrP
- EEGPT (NeurIPS 2024): https://github.com/BINE022/EEGPT
- 2026 EEG foundation-model benchmark: https://arxiv.org/abs/2601.17883
- Qiskit removed its old linear-solver/HHL module; historical documentation also emphasizes that
  full solution readout and oracle assumptions matter to the speedup claim:
  https://quantum.cloud.ibm.com/docs/en/api/qiskit/release-notes/0.43

## Roadmap

- **v0.2:** scientific claim ledger + mechanism kernel + CI/tests (this upgrade)
- **v0.3:** MNE/BIDS benchmark harness, strict subject/session splits, experiment manifests
- **v0.4:** frozen LaBraM + EEGPT adapters and matched density-geometry experiments
- **v0.5:** identifiable Lindblad-vs-LDS latent dynamics benchmark
- **v0.6:** preregistered contextual/order-effect experiment with classical adversaries
- **v0.7:** quantum-hardware/resource sandbox for only the hypotheses that survive the ladder

## Legacy notebooks

`test_qffy.ipynb` and the empty `test_qkalman.ipynb` are retained for provenance in v0.2. They should
be replaced by reproducible example notebooks only after the benchmark harness exists, so notebooks
never become the sole source of research logic again.
