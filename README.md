# QuantumBCI

**A falsifiable workbench for quantum, quantum-inspired, and classical models of neural signals.**

QuantumBCI asks a deliberately harder question than “can quantum mathematics be applied to EEG?”:
**does a specific quantum-structured mechanism add identifiable, reproducible value after strong
classical alternatives are given their best chance?**

The project separates four claim classes that are often blurred together:

1. **Classical controls** such as FFT, covariance geometry and Kalman/state-space models.
2. **Quantum-inspired models** using density operators, non-commuting observables or open-system
   dynamics as mathematical inductive biases without claiming the brain is physically quantum.
3. **Quantum algorithms** whose value must include state preparation, circuit, sampling, noise and
   readout costs.
4. **Physical quantum neural hypotheses**, which require independent operational evidence about a
   biological substrate. Better model fit is not that evidence.

> **Scientific stance:** QuantumBCI does not claim biologically functional entanglement, long-lived
> neural coherence, or demonstrated quantum computational advantage in brain tissue.

## Five-minute quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

quantumbci init
quantumbci doctor
quantumbci smoke
quantumbci runs list
```

`quantumbci smoke` runs a deterministic end-to-end density-mechanism recovery study and writes a
self-describing artifact bundle under `.quantumbci/runs/`, including `run.json`, metrics,
predictions, artifact hashes, Markdown and a standalone **HTML report**.

The smoke study is synthetic by design. Its signal lives in cross-feature correlation so density
geometry should recover it and an off-diagonal ablation should damage it. That qualifies plumbing
and mechanism recovery. It is not empirical neuroscience evidence.

See the [Research Workbench guide](docs/WORKBENCH.md) for the full CLI and artifact model.

## Bring your own embeddings

Already have frozen representations from LaBraM, EEGPT, neurOS, ORION or another encoder? You can
run the matched density/control/ablation benchmark without writing Python:

```bash
quantumbci benchmark embeddings.npy labels.npy \
  --train-indices train_indices.npy \
  --test-indices test_indices.npy \
  --split-name subject-exclusive-v1 \
  --output density_result.json
```

`embeddings.npy` must have shape `examples × tokens × features`. QuantumBCI **requires explicit split
indices** and never invents a random split on the caller’s behalf.

The same workflow is available as a NumPy-only API:

```python
import numpy as np
from quantumbci import IndexSplit, benchmark_density_embeddings

embeddings = np.load("embeddings.npy")
labels = np.load("labels.npy")

split = IndexSplit(
    train_indices=np.load("train_indices.npy"),
    test_indices=np.load("test_indices.npy"),
    name="subject-exclusive-v1",
)

result = benchmark_density_embeddings(embeddings, labels, split)
print(result.to_mapping())
```

The benchmark compares the same low-capacity readout family on:

- full density geometry;
- a diagonal-only density control;
- pooled mean/std classical features;
- the **fitted density readout after off-diagonal deletion**, which is an intervention rather than a
  separately refit model.

## QuantumBCI × neurOS

QuantumBCI composes with [neurOS](https://github.com/sidhulyalkar/neurOS-v1) rather than rebuilding a
second neural runtime stack.

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

The dependency direction is intentional: **neurOS remains independent of QuantumBCI**. QuantumBCI
optionally consumes stable neurOS packages and registers `quantumbci-density` through neurOS’s normal
`neuros.transforms` plugin group.

This lets a neurOS `SignalFrame` pass through a QuantumBCI representation while preserving
stream/timing/provenance identity. QuantumBCI can also reuse neurOS grouped and longitudinal split
contracts so quantum-inspired mechanisms, EEGNet/EEG-Conformer, foundation-model representations and
SourceWeigher controls can be compared on the **same evidence authority**.

See the [neurOS integration guide](docs/NEUROS_INTEGRATION.md).

## Install profiles

Base package and local workbench:

```bash
pip install -e '.[dev]'
```

Optional Qiskit/Aer:

```bash
pip install -e '.[quantum]'
```

Released neurOS-compatible packages:

```bash
pip install -e '.[neuros]'
```

Shared causal-evidence layer:

```bash
pip install -e '.[neuros-mechint]'
```

During active co-development, pin the exact sibling neurOS workspace packages described in
[`docs/NEUROS_INTEGRATION.md`](docs/NEUROS_INTEGRATION.md). The base QuantumBCI package intentionally
depends only on NumPy.

## Workbench commands

```text
quantumbci init                 create quantumbci.json
quantumbci doctor               show environment + optional integration readiness
quantumbci smoke                run a complete synthetic mechanism sanity study
quantumbci benchmark ...        evaluate user-supplied frozen .npy embeddings
quantumbci experiments list     discover research manifests
quantumbci experiments validate validate a scientific DAG contract
quantumbci experiments plan     bind a manifest to a source revision
quantumbci runs list            inspect local run history
quantumbci runs show <RUN_ID>   inspect one run ledger
quantumbci demo                 original compact mechanism demonstration
```

All important commands support machine-readable JSON output.

## Research kernel

```text
quantumbci/
├── benchmarking.py       # explicit-split density/control benchmark API
├── workbench.py          # config, run registry, smoke study, HTML report
├── cli.py                # installed `quantumbci` command
├── claims.py             # claim classes + falsification contracts
├── spectral.py           # complex FFT + correct ideal-QFT measurement semantics
├── states.py             # density operators, purity, entropy, coherence
├── open_system.py        # transparent Lindblad dynamics
├── contextuality.py      # non-commuting operators and order effects
├── kalman.py             # stable classical Kalman + QLSA suitability diagnostics
├── foundation.py         # frozen foundation-token → density-state bridge
├── interpretability.py   # mechanism signatures, ablations, stability
├── experiments/          # manifest + deterministic orchestration contracts
└── integrations/
    ├── neuros.py         # neurOS runtime/foundation/evidence bridge
    └── neuros_mechint.py # density interventions for shared causal evidence
```

The original `qfft_module.py` and `qkalman_module.py` remain compatibility surfaces, but their
scientific semantics are corrected: QFT measurement probabilities are not presented as a complex
FFT, a NumPy inverse is never labelled quantum-enhanced, retired Qiskit Aqua HHL code is not an
active backend, and experimental linear-system solvers must be explicit and resource-accounted.

## Main experiment program

The near-term research ladder is intentionally adversarial:

- **E001 density geometry:** identical frozen embeddings, explicit subject/session authority,
  covariance/Riemannian/bilinear/PCA/random-PSD controls, SourceWeigher adaptation controls,
  off-diagonal and basis interventions.
- **E002 open-system dynamics:** synthetic parameter recovery first, then Lindblad-style latent
  dynamics against LDS/Kalman, VAR, damped oscillator, switching-state and nonlinear controls.
- **E003 contextual/order effects:** retrospective discovery is non-confirmatory; prospective AB/BA
  work requires preregistration and applicable ethics approval.
- **E004 quantum resource sandbox:** QPU work begins only from an observable that survives prior
  classical/quantum-inspired gates.
- **E005 physical quantum mechanism screen:** requires an identified substrate, operational witness,
  discriminating perturbation, detection floor, strongest classical mimic and replication design.

Machine-readable contracts live in `experiments/manifests/`. Planning freezes the scientific DAG
before results are inspected:

```bash
quantumbci experiments validate experiments/manifests/E001_density_geometry.json
quantumbci experiments plan experiments/manifests/E001_density_geometry.json \
  --source-sha "$(git rev-parse HEAD)" \
  --output .quantumbci/plans/E001
```

A plan ID is deliberately weaker than a scientific run ID. Real studies additionally bind raw data
fingerprints and immutable split/calibration authority.

## Shared mechanistic evidence

QuantumBCI provides density-specific interventions without cloning `neuros-mechint`:

- remove density off-diagonals;
- permute the density basis while preserving its spectrum;
- mix continuously toward the maximally mixed state.

When `neuros-mechint` is installed, those interventions run through its native input-causal audit,
evidence-tier, manifest, control and result contracts. QuantumBCI does not relabel that evidence.

## Validation philosophy

A result is interesting only if it survives three ledgers:

- **Mathematical:** Hermiticity/PSD/trace, normalization, numerical stability, circuit semantics.
- **Predictive:** held-out subjects/sessions, calibration, transfer, data efficiency, compute.
- **Mechanistic:** parameter recovery, intervention prediction, identifiability, stability, matched
  classical alternatives and explicit falsifiers.

A negative scientific finding is allowed to be a successful software run. Failing a promotion gate
should make a downstream claim ineligible, not turn falsification into an infrastructure error.

CI separately qualifies Python 3.10–3.12, the installed workbench console/smoke path, and the neurOS
bridge against an exact pinned neurOS source revision.

## Reading context

- Quantum cognition can use quantum probability **without requiring quantum brain physics**:
  Pothos & Busemeyer (2022), https://pubmed.ncbi.nlm.nih.gov/34546804/
- LaBraM (ICLR 2024): https://openreview.net/forum?id=QzTpTRVtrP
- EEGPT (NeurIPS 2024): https://github.com/BINE022/EEGPT
- 2026 EEG foundation-model benchmark: https://arxiv.org/abs/2601.17883
- Qiskit removed its old linear-solver/HHL module; historical docs also emphasize readout/oracle
  assumptions: https://quantum.cloud.ibm.com/docs/en/api/qiskit/release-notes/0.43

## Roadmap

- **v0.2:** claim ledger + mechanism kernel + scientific semantics
- **v0.3:** experiment manifests/orchestration + neurOS runtime/evidence integration
- **v0.4:** usable local workbench + frozen-embedding benchmark + executable E001 foundation
- **v0.5:** authoritative real E001 longitudinal benchmark + Lindblad-vs-LDS implementation
- **v0.6:** preregistered contextual/order-effect experiment with classical adversaries
- **v0.7:** quantum-hardware/resource sandbox only for hypotheses surviving the prior ladder

## Legacy notebooks

`test_qffy.ipynb` and the empty `test_qkalman.ipynb` are retained for provenance. Reproducible CLI,
Python APIs and artifact ledgers are now the supported path; notebooks should remain teaching or
exploration surfaces rather than the sole source of research logic.
