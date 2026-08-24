# QuantumBCI

**A falsifiable workbench for quantum, quantum-inspired, and classical models of neural signals.**

QuantumBCI began as a small QFT/Kalman demonstration. The modern project is a research workbench for a
harder question: **does a specific quantum-structured mechanism add identifiable, reproducible value
to neural modelling after strong classical alternatives are tested?**

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

## QuantumBCI × neurOS

QuantumBCI is designed to compose with [neurOS](https://github.com/sidhulyalkar/neurOS-v1) rather than
rebuild a second neural runtime stack.

```text
neurOS
  SignalFrame / sources / replay / timing / provenance
  grouped + longitudinal evidence authority
  models + foundation-model interoperability
                 |
                 v
QuantumBCI
  density geometry / open-system dynamics / contextual models
  falsification gates / matched controls / quantum resources
                 |
                 v
neuros-mechint
  interventions / held-out evidence / replication / evidence packs
```

The dependency direction is deliberate: **neurOS remains independent of QuantumBCI**. QuantumBCI can
optionally consume stable neurOS packages and registers the external `quantumbci-density` transform
through neurOS's normal `neuros.transforms` plugin group.

This lets a neurOS `SignalFrame` move through a QuantumBCI density representation while retaining its
stream/timing/provenance identity. E001 also reuses neurOS's leakage-resistant grouped and longitudinal
split contracts, so QuantumBCI mechanisms can be compared against neurOS EEGNet/EEG-Conformer and
future SourceWeigher lanes on the exact same sample authority rather than on look-alike splits.

See [the full neurOS integration guide](docs/NEUROS_INTEGRATION.md).

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

See [the research agenda](docs/RESEARCH_AGENDA.md), [mechanism cards](docs/MECHANISM_CARDS.md),
[interpretability protocol](docs/INTERPRETABILITY_PROTOCOL.md), and
[experiment registry](experiments/README.md).

## Research kernel

```text
quantumbci/
├── claims.py             # claim classes + falsification contracts
├── spectral.py           # complex FFT + correct ideal-QFT measurement semantics
├── states.py             # density operators, purity, entropy, coherence
├── open_system.py        # transparent Lindblad dynamics
├── contextuality.py      # non-commuting operators and order effects
├── kalman.py             # stable classical Kalman + QLSA suitability diagnostics
├── foundation.py         # frozen foundation-token -> density-state bridge
├── interpretability.py   # mechanism signatures, ablations, stability
├── signals.py            # deterministic synthetic test signals
├── experiments/          # manifests + deterministic orchestration contracts
└── integrations/
    ├── neuros.py         # neurOS runtime/foundation/evidence bridge
    └── neuros_mechint.py # density interventions for shared causal evidence
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

For a released neurOS-compatible installation:

```bash
pip install -e '.[neuros]'
```

During active co-development, install the exact sibling neurOS workspace packages instead; see
[`docs/NEUROS_INTEGRATION.md`](docs/NEUROS_INTEGRATION.md). The base QuantumBCI package intentionally
depends only on NumPy.

## Example: use QuantumBCI inside neurOS

Once QuantumBCI is installed, neurOS can discover the density transform through its existing plugin
registry:

```yaml
streams:
  - id: eeg
    source:
      plugin: mock
      options:
        sampling_rate: 250.0
        channels: 8
    transforms:
      - plugin: quantumbci-density
        options:
          sample_axis: -1
          output: observables
```

The transform annotates the resulting frame with `quantumbci_claim_class=quantum_inspired`. Runtime
success is therefore never silently promoted into a physical-quantum biological claim.

A lightweight executable demonstration lives at `examples/neuros_density_bridge.py`.

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

When `neuros-foundation` is installed, `NeurOSFoundationEncoder` can consume a runnable neurOS registry
adapter while preserving neurOS's fail-closed availability semantics. A catalog entry without a real
execution adapter raises rather than generating placeholder benchmark embeddings.

The recommended experiment is paired: **same subjects, same evidence authority, same frozen embeddings,
same readout, different representation layer.** Then run off-diagonal/basis interventions and bootstrap
the mechanism observables.

## Shared mechanistic evidence

QuantumBCI provides mechanism-specific interventions without cloning `neuros-mechint`:

- remove density off-diagonals;
- permute the density basis while preserving its spectrum;
- mix continuously toward the maximally mixed state.

When `neuros-mechint` is installed, these interventions can run through its native input-causal audit,
evidence tier, manifest, control, and result contracts. QuantumBCI does not relabel that evidence.

## Reproducible experiment orchestration

The `experiments/manifests/` registry freezes hypotheses, datasets, encoders, stage dependencies,
primary metrics, artifacts, and promotion gates before results are inspected. `quantumbci.experiments`
validates those DAGs and produces deterministic plan identities.

A plan identity is intentionally weaker than a scientific run identity. For neurOS-backed experiments,
the latter additionally binds:

- upstream/raw dataset fingerprint;
- neurOS partition fingerprint;
- neurOS calibration/final-evaluation fingerprint when applicable;
- QuantumBCI source revision;
- neurOS source revision and installed package versions.

CI separately tests QuantumBCI across Python 3.10–3.12 and qualifies the neurOS bridge against an exact
pinned neurOS source revision, including a real `SignalFrame`, plugin discovery, and a synthetic real
neurOS longitudinal evidence contract.

## Validation philosophy

A result is interesting only if it survives three ledgers:

- **Mathematical:** Hermiticity/PSD/trace, normalization, numerical stability, circuit semantics.
- **Predictive:** held-out subjects/sessions, calibration, transfer, data efficiency, compute.
- **Mechanistic:** parameter recovery, intervention prediction, identifiability, stability, matched
  classical alternatives, explicit falsifiers.

A negative scientific finding is allowed to be a successful software run. Failing a promotion gate
should make downstream claims ineligible, not turn falsification into an infrastructure error.

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

- **v0.2:** scientific claim ledger + mechanism kernel + CI/tests
- **v0.3:** experiment manifests/orchestration + neurOS runtime/evidence integration
- **v0.4:** executable E001 density benchmark on frozen neurOS/foundation representations
- **v0.5:** identifiable Lindblad-vs-LDS latent dynamics benchmark
- **v0.6:** preregistered contextual/order-effect experiment with classical adversaries
- **v0.7:** quantum-hardware/resource sandbox only for hypotheses that survive the prior ladder

## Legacy notebooks

`test_qffy.ipynb` and the empty `test_qkalman.ipynb` are retained for provenance. They should be
replaced by reproducible example notebooks only after the benchmark harness exists, so notebooks never
become the sole source of research logic again.
