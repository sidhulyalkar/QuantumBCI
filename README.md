# QuantumBCI

**A falsifiable workbench for quantum, quantum-inspired, and classical models of neural signals.**

QuantumBCI asks a harder question than “can quantum mathematics be applied to EEG?”:
**does a specific quantum-structured mechanism add identifiable, reproducible value after the
strongest classical equivalent and alternatives are given their best chance?**

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

## Equivalence first

Before benchmarking a quantum-looking representation, QuantumBCI now asks whether it is actually
information-distinct from an ordinary classical statistic.

For the current density constructor,

```text
rho = X^H X / Tr(X^H X)
```

after optional centering. This is exactly a **trace-normalized Hermitian second moment**. After
centering it is proportional to ordinary covariance, so the density matrix itself contains no
information unavailable to the matched normalized-covariance representation.

That does not make operator language useless. Density coordinates may still support useful
constraints, spectral observables, interventions, or later non-commuting dynamics. It does mean
that a predictive gain over weaker controls cannot be described as “additional quantum
information.”

Run the installed mathematical audit directly:

```bash
quantumbci-audit density embeddings.npy
quantumbci-audit density embeddings.npy --json
```

Or run the wider E001 representation-control gauntlet:

```bash
quantumbci-audit e001 embeddings.npy labels.npy \
  --train-indices train_indices.npy \
  --test-indices test_indices.npy \
  --output e001-audit.json
```

The E001 audit includes normalized covariance, ordinary covariance, log-covariance geometry,
bilinear second moment, pooled statistics, train-only PCA, diagonal density, full density, and a
fixed-readout off-diagonal intervention on the same token tensor.

See [Mathematical equivalence gates](docs/MATHEMATICAL_EQUIVALENCE.md).

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

`quantumbci smoke` runs a deterministic synthetic covariance-sensitive sanity study and writes a
self-describing artifact bundle under `.quantumbci/runs/`, including `run.json`, metrics,
predictions, artifact hashes, Markdown and a standalone HTML report.

The smoke signal lives in cross-feature correlation, so a second-moment representation should
recover it and deleting cross-feature terms should damage it. That qualifies plumbing and
intervention semantics. It is **not** evidence that density notation contains information beyond
covariance, and it is not empirical neuroscience evidence.

See the [Research Workbench guide](docs/WORKBENCH.md).

## Bring your own embeddings

Already have frozen token representations from LaBraM, EEGPT, neurOS, ORION or another encoder?
The compatibility benchmark remains available:

```bash
quantumbci benchmark embeddings.npy labels.npy \
  --train-indices train_indices.npy \
  --test-indices test_indices.npy \
  --split-name subject-exclusive-v1 \
  --output density_result.json
```

`embeddings.npy` must have shape `examples × tokens × features`. QuantumBCI requires explicit split
indices and never invents a random split on the caller’s behalf.

For new mechanism work, prefer `quantumbci-audit e001` because it includes the exact normalized-
covariance equivalence control and the broader classical gauntlet.

The same APIs are NumPy-first:

```python
import numpy as np
from quantumbci import IndexSplit, benchmark_e001_embeddings

embeddings = np.load("embeddings.npy")
labels = np.load("labels.npy")
split = IndexSplit(
    train_indices=np.load("train_indices.npy"),
    test_indices=np.load("test_indices.npy"),
    name="subject-exclusive-v1",
)

result = benchmark_e001_embeddings(embeddings, labels, split)
print(result.to_mapping())
```

## Authoritative longitudinal E001 with neurOS

QuantumBCI composes with [neurOS](https://github.com/sidhulyalkar/neurOS-v1) rather than rebuilding a
second neural evidence stack.

```text
neurOS
  data identity / replay / chronology
  LongitudinalCaseAuthority
  fixed calibration + evaluation frontier
                 |
                 v
QuantumBCI
  token representation fingerprint
  equivalence audit
  density/operator + matched classical controls
                 |
                 v
participant-level paired inference
                 |
                 v
neuros-mechint
  interventions / held-out evidence / replication
```

The dependency direction is intentional: **neurOS remains independent of QuantumBCI**.

The v0.6 longitudinal runner consumes neurOS `LongitudinalCaseAuthority`. Before fitting,
`authority.restore(data)` revalidates processed-data identity, chronology, source groups, partition
fingerprint, calibration fingerprint, and the frozen source/calibration/evaluation membership.
QuantumBCI then uses the same `train_indices_for_budget(k)` frontier and immutable final evaluation
indices at every calibration budget.

Repeated held-out sessions are aggregated within participant before bootstrap resampling. A window
or trial bootstrap is not accepted as participant-level promotion evidence.

This lets QuantumBCI be compared against neurOS CSP/LDA, EEGNet, EEG-Conformer, frozen-transfer and
SourceWeigher lanes on the **same evidence cases** without pretending all models expose identical
token geometry.

See the [neurOS integration guide](docs/NEUROS_INTEGRATION.md).

## Portable study recipes and evidence objects

For work that should survive outside one machine or one lab, use a recipe:

```bash
quantumbci recipe init study.json
quantumbci recipe validate study.json
quantumbci recipe run study.json --config quantumbci.json
```

A recipe binds dataset/model identifiers, benchmark parameters, explicit train/test authority, and
SHA-256 content fingerprints of the frozen input files. Validation preflights tensor shape, finite
values, split disjointness/range, and training-class support before fitting.

Scientific identity is **content-addressed, not filename-addressed**. Renaming byte-identical input
files does not change the scientific fingerprint.

Completed runs can be verified and shared:

```bash
quantumbci runs verify <RUN_ID>

quantumbci runs export <RUN_ID> \
  --format ro-crate \
  --output shared/my-study

quantumbci runs export <RUN_ID> \
  --format bids \
  --output /path/to/bids-root \
  --bids-version <DATASET_BIDS_VERSION>
```

RO-Crate export emits a self-contained RO-Crate 1.3 evidence object. The BIDS path creates a
BIDS-aware derivative container while explicitly avoiding the claim that generic QuantumBCI JSON
files are a standardized modality-specific BIDS derivative datatype.

Before export, QuantumBCI verifies a closed-world artifact ledger. Missing, modified, or newly added
top-level files invalidate export. SHA-256 provides integrity checking, not authorship signing.

See [Public research workflows](docs/PUBLIC_RESEARCH_WORKFLOWS.md).

## Install profiles

Base package and development tools:

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

The base QuantumBCI package intentionally depends only on NumPy.

## Command surfaces

```text
quantumbci init                    create quantumbci.json
quantumbci doctor                  inspect local workbench/integration readiness
quantumbci smoke                   run the deterministic synthetic sanity study
quantumbci benchmark ...           compatibility frozen-embedding benchmark
quantumbci recipe init             create a portable study contract
quantumbci recipe validate         preflight + fingerprint external inputs
quantumbci recipe run              execute a recipe into the RunStore
quantumbci experiments list        discover research manifests
quantumbci experiments validate    validate a scientific DAG contract
quantumbci experiments plan        bind a manifest to a source revision
quantumbci runs list               inspect local run history
quantumbci runs show <RUN_ID>      inspect one run ledger
quantumbci runs verify <RUN_ID>    verify artifact integrity
quantumbci runs export <RUN_ID>    export RO-Crate or BIDS-aware evidence
quantumbci demo                    compact mechanism demonstration

quantumbci-audit density ...       test density / normalized-covariance equivalence
quantumbci-audit e001 ...          run the adversarial E001 representation controls
```

All important commands support machine-readable JSON output.

## Research kernel

```text
quantumbci/
├── equivalence.py        # mathematical equivalence audits before empirical claims
├── benchmarking.py       # frozen-token representation + classical-control gauntlets
├── longitudinal.py       # neurOS-authoritative calibration-frontier E001 execution
├── audit_cli.py          # installed `quantumbci-audit` surface
├── workbench.py          # config, run registry, smoke study, HTML report
├── recipes.py            # portable, content-addressed study contracts
├── exporting.py          # integrity verification + RO-Crate/BIDS-aware export
├── cli.py                # installed `quantumbci` workbench command
├── claims.py             # claim classes + falsification contracts
├── spectral.py           # complex FFT + correct ideal-QFT measurement semantics
├── states.py             # density operators, purity, entropy, coherence-like observables
├── open_system.py        # transparent Lindblad dynamics
├── contextuality.py      # non-commuting operators and order effects
├── kalman.py             # stable classical Kalman + QLSA suitability diagnostics
├── foundation.py         # frozen foundation-token → operator bridge
├── interpretability.py   # mechanism signatures, ablations, stability
├── experiments/          # manifest + deterministic orchestration contracts
└── integrations/
    ├── neuros.py         # neurOS runtime/foundation/evidence bridge
    └── neuros_mechint.py # operator interventions for shared causal evidence
```

The original `qfft_module.py` and `qkalman_module.py` remain compatibility surfaces, but their
scientific semantics are corrected: QFT measurement probabilities are not presented as a complex
FFT, a NumPy inverse is never labelled quantum-enhanced, retired Qiskit Aqua HHL code is not an
active backend, and experimental linear-system solvers must be explicit and resource-accounted.

## Main experiment program

The research ladder is intentionally adversarial:

- **E001 operator geometry:** exact density/covariance equivalence audit first, then identical token
  tensors, merged neurOS longitudinal authority, normalized/ordinary/log covariance, bilinear,
  pooled, PCA and intervention controls, followed by participant-level inference.
- **E002 open-system dynamics:** synthetic parameter recovery and equivalence-to-classical-dynamics
  audit first, then Lindblad-style latent dynamics against LDS/Kalman, VAR, damped oscillator,
  switching-state and nonlinear controls.
- **E003 contextual/order effects:** retrospective discovery is non-confirmatory; prospective AB/BA
  work requires preregistration, applicable ethics approval, and history-aware classical adversaries.
- **E004 quantum resource sandbox:** QPU work begins only from an observable that survives prior
  mathematical and classical gates.
- **E005 physical quantum mechanism screen:** requires an identified substrate, operational witness,
  discriminating perturbation, detection floor, strongest classical mimic and replication design.

The E001 manifest now contains a real executable mathematical stage:

```bash
python -m quantumbci.experiments.tasks \
  equivalence-audit E001 density-covariance \
  --output equivalence_audit.json
```

Later dataset/model stages remain fail-closed until their executors are implemented. QuantumBCI does
not fabricate placeholder artifacts to make a DAG look complete.

Machine-readable contracts live in `experiments/manifests/`:

```bash
quantumbci experiments validate experiments/manifests/E001_density_geometry.json
quantumbci experiments plan experiments/manifests/E001_density_geometry.json \
  --source-sha "$(git rev-parse HEAD)" \
  --output .quantumbci/plans/E001
```

A plan ID is deliberately weaker than a scientific run ID. Real studies additionally bind raw data,
processed data, representation fingerprints, and immutable evidence authority.

## Shared mechanistic evidence

QuantumBCI provides operator-specific interventions without cloning `neuros-mechint`:

- remove density off-diagonals, interpreted as cross-feature covariance deletion;
- permute the density basis while preserving its spectrum;
- mix continuously toward the maximally mixed state.

When `neuros-mechint` is installed, these can run through its native causal-evidence contracts.
QuantumBCI does not relabel that evidence or promote it to a physical-quantum claim.

## Validation philosophy

A result is interesting only if it survives four ledgers:

- **Equivalence:** is the object genuinely information-distinct from the strongest classical form?
- **Mathematical:** Hermiticity/PSD/trace, normalization, numerical stability, circuit semantics.
- **Predictive:** held-out subjects/sessions, calibration, transfer, data efficiency, compute.
- **Mechanistic:** parameter recovery, intervention prediction, identifiability, stability, matched
  alternatives and explicit falsifiers.

A negative scientific finding is allowed to be a successful software run. Failing a promotion gate
makes a downstream claim ineligible; it does not turn falsification into an infrastructure error.

CI qualifies Python 3.10–3.12, installed workbench/recipe/export/audit surfaces, wheel contents, and
the real neurOS bridge against an exact merged neurOS longitudinal-authority revision.

## Reading context

- Quantum cognition can use quantum probability without requiring quantum brain physics:
  Pothos & Busemeyer (2022), https://pubmed.ncbi.nlm.nih.gov/34546804/
- LaBraM (ICLR 2024): https://openreview.net/forum?id=QzTpTRVtrP
- EEGPT (NeurIPS 2024): https://github.com/BINE022/EEGPT
- 2026 EEG foundation-model benchmark: https://arxiv.org/abs/2601.17883
- Historical Qiskit linear-solver/HHL removal notes:
  https://quantum.cloud.ibm.com/docs/en/api/qiskit/release-notes/0.43

## Roadmap

- **v0.2:** claim ledger + mechanism kernel + corrected scientific semantics
- **v0.3:** experiment orchestration + neurOS runtime/evidence integration
- **v0.4:** usable local workbench + frozen-embedding benchmark
- **v0.5:** portable recipes + verifiable RO-Crate/BIDS-aware evidence
- **v0.6:** equivalence-first E001 + merged neurOS longitudinal authority + participant inference
- **v0.7:** Lindblad-vs-LDS/oscillator implementation after mathematical identifiability gates
- **v0.8:** preregistered contextual/order-effect experiment with classical adversaries
- **v0.9:** quantum-hardware/resource sandbox only for hypotheses surviving the prior ladder

## Legacy notebooks

`test_qffy.ipynb` and the empty `test_qkalman.ipynb` are retained for provenance. Reproducible CLI,
Python APIs and artifact ledgers are the supported path; notebooks should remain teaching or
exploration surfaces rather than the sole source of research logic.
