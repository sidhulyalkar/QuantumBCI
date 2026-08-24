# neurOS × QuantumBCI

QuantumBCI is a **research extension of neurOS**, not a competing neural runtime. The projects have
complementary responsibilities:

- **neurOS** owns neural-data contracts, sources/replay/timing, recording interoperability,
  foundation-model interoperability, grouped/longitudinal evidence authority, model ladders and
  broad mechanistic evidence tooling.
- **QuantumBCI** owns quantum/quantum-inspired mechanism hypotheses, mathematical-equivalence gates,
  claim ceilings, operator/open-system/contextual primitives, matched falsification controls and
  quantum-resource accounting.

The dependency direction is intentionally one-way:

```text
neurOS stable contracts
       ^
       | optional dependency / plugin host
       |
QuantumBCI research extension
```

neurOS does not depend on QuantumBCI.

## Why reuse neurOS

The neurOS contracts solve several problems that QuantumBCI should not clone:

- `SignalFrame` / `StreamDescriptor` provide a stable neural-data ABI.
- recording and replay preserve runtime/provenance semantics.
- `neuros-foundation` provides fail-closed model interoperability and real-world evaluation
  contracts.
- merged `LongitudinalCaseAuthority` freezes actual source, calibration and final-evaluation sample
  membership together with processed-data identity and chronology.
- the longitudinal model ladder compares CSP/LDA, EEGNet, EEG-Conformer, frozen transfer and
  SourceWeigher on that same authority.
- `neuros-mechint` provides intervention/evidence abstractions that can carry promoted QuantumBCI
  mechanisms without declaring them biologically causal by fiat.

QuantumBCI should add **mechanism-specific value**, not another copy of those systems.

## Co-development installation

```bash
pip install -e ../neurOS-v1/packages/neuros-core
pip install -e ../neurOS-v1/packages/neuros-models
pip install -e ../neurOS-v1/packages/neuros-foundation
pip install -e .
```

For the heavier causal-evidence bridge:

```bash
pip install -e ../neurOS-v1/packages/neuros-mechint
```

QuantumBCI also exposes `neuros` and `neuros-mechint` optional extras for released installations.

The v0.6 cross-repository CI pins the exact merged neurOS longitudinal-authority revision rather than
tracking a moving branch.

## 1. Use QuantumBCI as a neurOS transform

QuantumBCI registers `quantumbci-density` in the standard `neuros.transforms` entry-point group:

```yaml
streams:
  - id: eeg
    source:
      plugin: mock
      options:
        sampling_rate: 250.0
        channels: 8
    transforms:
      - plugin: smoothing
        options:
          window_size: 3
      - plugin: quantumbci-density
        options:
          sample_axis: -1
          output: observables
```

`sample_axis=-1` corresponds to a common neurOS `(channels, samples)` EEG chunk. The transform
preserves `SignalFrame` identity when supplied and records an explicit
`quantumbci_claim_class=quantum_inspired` metadata field.

### Output modes

`output="vector"` emits the complete Hermitian operator as `d^2` real coordinates.

`output="observables"` emits:

1. purity;
2. von Neumann entropy;
3. L1 off-diagonal mass.

The third quantity is historically named `l1_coherence` in the operator API, but on the current
EEG density constructor it is a **basis-dependent function of cross-feature second-moment/covariance
terms**. It must not be interpreted as evidence for microscopic quantum coherence in neural tissue.

The transform never labels its output as a physical quantum state.

## 2. Use neurOS foundation models inside QuantumBCI

QuantumBCI can consume any runnable neurOS foundation adapter without embedding that model ecosystem
into its core package:

```python
from quantumbci.foundation import density_states_from_embeddings
from quantumbci.integrations.neuros import NeurOSFoundationEncoder

encoder = NeurOSFoundationEncoder.from_registry("some-runnable-model")
embeddings = encoder.encode(eeg, sample_rate_hz=250.0)
rho = density_states_from_embeddings(embeddings)
```

The bridge inherits neurOS fail-closed behavior. A catalog entry without a runnable upstream adapter
raises rather than generating placeholder embeddings.

For adapters whose encode method expects sampling rate under a model-specific keyword:

```python
encoder = NeurOSFoundationEncoder.from_registry(
    "some-model",
    sample_rate_kw="sfreq",
)
```

For E001, use a **real token-level representation**. Do not manufacture a token axis around a pooled
embedding merely to make an operator constructor run. If a sibling model exposes only one vector per
trial, compare its predictive result on the shared evidence authority but do not pretend the
representation geometry is identical.

## 3. Consume merged longitudinal authority in E001

The preferred v0.6 path is to consume a neurOS `LongitudinalCaseAuthority`, not to recreate its split
inside QuantumBCI.

A neurOS study first creates and serializes the authority. QuantumBCI then receives:

- the same `GroupedEvaluationData` used to create that authority;
- the serialized/restored `LongitudinalCaseAuthority`;
- one token representation aligned one-to-one with the processed data rows;
- upstream/raw dataset fingerprint;
- exact QuantumBCI source revision;
- exact neurOS source revision.

```python
from quantumbci.longitudinal import run_longitudinal_e001_case

result = run_longitudinal_e001_case(
    grouped_data,
    authority,
    token_embeddings,
    representation_id="labrAM@checkpoint-sha:layer-8-tokens",
    budgets_per_class=(0, 1, 2, 5, 10),
    upstream_dataset_fingerprint=RAW_DATA_FINGERPRINT,
    quantumbci_source_sha=QUANTUMBCI_SHA,
    neuros_source_sha=NEUROS_SHA,
)
```

The first operation is `authority.restore(grouped_data)`. neurOS therefore revalidates processed
neural bytes, sample identity, chronology, historical source groups, target calibration pools,
immutable final evaluation samples, partition fingerprint and calibration fingerprint before
QuantumBCI fits anything.

For each budget QuantumBCI uses:

```text
train = source history + authority.calibration_indices(budget)
test  = the same immutable authority.evaluation_indices
```

No future session enters source history and no calibration example enters final evaluation.

## 4. Scientific identity across both repositories

A v0.6 longitudinal case fingerprint binds:

1. upstream/raw dataset fingerprint;
2. neurOS processed-data SHA-256;
3. neurOS authority fingerprint;
4. neurOS partition fingerprint;
5. neurOS calibration-split fingerprint;
6. exact token representation SHA-256;
7. representation identifier/model/checkpoint description;
8. calibration-budget frontier;
9. benchmark settings;
10. exact QuantumBCI source revision;
11. exact neurOS source revision.

Changing any of these changes the QuantumBCI study fingerprint.

`bind_neuros_evidence(...)` remains useful at experiment-plan level, but the longitudinal runner adds
representation bytes and exact case authority to the final empirical identity.

## 5. Equivalence before predictive claims

The current density constructor

```text
rho = X^H X / Tr(X^H X)
```

is exactly a trace-normalized Hermitian second moment after the same optional centering. E001
therefore includes `normalized_covariance` as a mandatory exact classical equivalent.

The correct interpretation is:

```text
same neurOS case authority
        |
        v
same token tensor
   |            |
   v            v
density      exact normalized covariance
   |            |
   +------ equivalence gate ------+
                 |
                 v
compare only genuinely different
inductive biases / observables / dynamics
```

A successful equivalence finding blocks a representation-information novelty claim. It does not make
the run a failure.

See [Mathematical equivalence gates](MATHEMATICAL_EQUIVALENCE.md).

## 6. Participant-level inference

For promotion-oriented longitudinal results, repeated held-out sessions from one participant are
first averaged within participant. Bootstrap resampling then samples participants, not trials or EEG
windows.

`paired_participant_bootstrap(...)` fails closed when participant metadata is missing, fewer than two
participants are present, duplicate case rows appear, or participant membership differs across
calibration budgets.

This complements neurOS's frozen sample authority with an explicit independent inference unit.

## 7. neurOS-mechint as the shared intervention layer

Do not copy neurOS-mechint into QuantumBCI. Translate promoted mechanism interventions into its
native experiment/evidence contracts:

| QuantumBCI intervention | Shared evidence interpretation |
| --- | --- |
| remove density off-diagonals | cross-feature covariance/representation ablation |
| scramble density eigenbasis | basis counterfactual preserving spectrum |
| suppress one Hamiltonian coupling | parameter/component ablation |
| perturb dephasing/collapse rate | dose-response intervention |
| force contextual operators to commute | mechanism substitution |

A successful intervention may support the stated quantum-inspired model. It does not by itself
promote a physical-quantum claim.

## 8. ORION as a later representation lane

ORION can become another token source after ordinary/raw/foundation controls are understood:

```text
neurOS evidence authority
   -> ORION tokens
   -> QuantumBCI equivalence audit
   -> operator/dynamical mechanism
   -> matched classical controls
   -> neuros-mechint interventions
```

This ordering keeps attribution clean. If an ORION representation is already strong, QuantumBCI
receives credit only for incremental mechanism value beyond that representation.

## Flagship ecosystem workflow

A useful cross-project tutorial should eventually perform:

1. load/record neural data with neurOS;
2. freeze `LongitudinalCaseAuthority`;
3. run the neurOS specialist/frozen/SourceWeigher model ladder;
4. export a genuine token representation on the same rows;
5. run QuantumBCI mathematical equivalence auditing;
6. run E001 or another surviving non-equivalent mechanism;
7. perform participant-level paired inference;
8. route promoted interventions through neurOS-mechint;
9. emit one portable evidence object containing both repository revisions and all data,
   representation and authority fingerprints.

That is the intended ecosystem boundary: **neurOS is the reliable neural evidence substrate;
QuantumBCI is the adversarial mechanism laboratory.**
