# neurOS × QuantumBCI

QuantumBCI is designed to be a **research extension of neurOS**, not a competing neural runtime.
The projects have complementary responsibilities:

- **neurOS** owns neural-data contracts, device/dataset sources, replay, timing, runtime graphs,
  foundation-model interoperability, grouped evaluation, and broad mechanistic evidence tooling.
- **QuantumBCI** owns quantum/quantum-inspired mechanism hypotheses, claim ceilings, density/open-system/
  contextual primitives, falsification gates, and quantum-resource accounting.

The dependency direction is intentionally one-way:

```text
neurOS stable contracts
       ^
       | optional dependency / plugin host
       |
QuantumBCI research extension
```

neurOS does not depend on QuantumBCI. Installing QuantumBCI makes its research transforms available to
neurOS through standard Python entry points.

## Why reuse neurOS

The neurOS v2 contracts already solve several problems that QuantumBCI experiments require:

- `SignalFrame` and `StreamDescriptor` provide a stable neural-data ABI with timing, quality, and provenance.
- canonical recording/replay preserves exact runtime semantics and per-frame integrity.
- `neuros-foundation` provides a fail-closed foundation-model registry, representation probes, real-world
  evidence sources, deployment-unit-aware partitions, and longitudinal calibration contracts.
- `neuros-mechint` provides intervention/evidence abstractions that can later carry QuantumBCI mechanism
  interventions without declaring them biologically causal by fiat.

QuantumBCI should therefore add **mechanism-specific value**, not clone these facilities.

## Co-development installation

During active development, install neurOS from the sibling workspace so the exact source tree is visible:

```bash
# from a checkout containing both repositories
pip install -e ../neurOS-v1/packages/neuros-core
pip install -e ../neurOS-v1/packages/neuros-models
pip install -e ../neurOS-v1/packages/neuros-foundation
pip install -e .
```

For the heavier causal-evidence bridge:

```bash
pip install -e ../neurOS-v1/packages/neuros-mechint
```

The QuantumBCI package metadata also exposes `neuros` and `neuros-mechint` optional extras for released
installations.

## 1. Use QuantumBCI as a neurOS transform

QuantumBCI registers the `quantumbci-density` transform in neurOS's `neuros.transforms` plugin group.
A neurOS configuration can therefore contain:

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

`sample_axis=-1` corresponds to the common neurOS EEG chunk layout `(channels, samples)`. The transform
preserves a `SignalFrame` when one is supplied and annotates metadata with:

- `representation=quantumbci_density_vector` or `quantumbci_density_observables`;
- `quantumbci_claim_class=quantum_inspired`;
- density dimension and centering/sample-axis settings.

The transform never labels the output as a physical quantum state.

### Output modes

`output="vector"` emits the complete Hermitian density representation as exactly `d^2` real features:
real diagonal entries plus real and imaginary upper-triangular entries.

`output="observables"` emits three inspectable values:

1. purity;
2. von Neumann entropy;
3. L1 coherence.

The full-vector mode is appropriate for matched predictive benchmarks. The observable mode is useful for
runtime monitoring, mechanistic plots, and deliberately low-capacity probes.

## 2. Use neurOS foundation models inside QuantumBCI

QuantumBCI can consume any runnable neurOS foundation adapter without embedding that model ecosystem into
its core package:

```python
from quantumbci.foundation import density_states_from_embeddings
from quantumbci.integrations.neuros import NeurOSFoundationEncoder

encoder = NeurOSFoundationEncoder.from_registry("neuros-neurofmx")
embeddings = encoder.encode(eeg, sample_rate_hz=250.0)
rho = density_states_from_embeddings(embeddings)
```

The bridge intentionally inherits neurOS's fail-closed behavior. A catalog entry without a runnable
upstream adapter raises rather than generating placeholder embeddings.

For adapters whose encode method expects sampling rate under a model-specific name:

```python
encoder = NeurOSFoundationEncoder.from_registry(
    "some-model",
    sample_rate_kw="sfreq",
)
```

## 3. Reuse neurOS evidence boundaries in E001

E001 should use neurOS `GroupedEvaluationData` and its partition APIs rather than inventing a parallel
split implementation. A typical longitudinal study becomes:

```python
from neuros.foundation_models import (
    chronological_partition,
    make_nested_calibration_split,
)
from quantumbci.experiments import build_plan, load_manifest
from quantumbci.integrations.neuros import bind_neuros_evidence

manifest = load_manifest("experiments/manifests/E001_density_geometry.json")
plan = build_plan(manifest, source_sha=QUANTUMBCI_SHA)

partition = chronological_partition(
    grouped_data,
    split_unit="session",
    held_out_value="session_3",
)
calibration = make_nested_calibration_split(
    partition,
    evaluation_fraction=0.5,
    seed=0,
)

binding = bind_neuros_evidence(
    plan,
    dataset_fingerprint=UPSTREAM_RAW_DATA_SHA256,
    partition=partition,
    calibration_split=calibration,
    neuros_source_sha=NEUROS_SHA,
)
print(binding.scientific_run_id)
```

The scientific run identity now changes if any of these change:

- QuantumBCI manifest/source revision;
- upstream/raw dataset fingerprint;
- neurOS train/test partition assignment;
- neurOS nested calibration/evaluation assignment;
- neurOS source revision;
- installed neurOS package versions.

This is intentionally stronger than using only a seed or a split name.

## 4. neurOS-mechint should become the shared evidence layer

Do **not** copy neurOS-mechint into QuantumBCI. Instead, the next integration should define small adapters
that translate QuantumBCI mechanism interventions into `neuros-mechint` experiments:

| QuantumBCI intervention | neurOS-mechint interpretation |
| --- | --- |
| remove density off-diagonals | ablation intervention on an explicit representation surface |
| scramble density eigenbasis | counterfactual representation intervention |
| suppress one Hamiltonian coupling | parameter/component ablation |
| perturb dephasing/collapse rate | dose-response intervention |
| force contextual operators to commute | causal mechanism substitution |

This gives QuantumBCI access to held-out intervention evidence, replication, dose response, factorial
comparisons, and artifact provenance while giving neurOS-mechint a distinctive new scientific use case.

## 5. ORION is a later representation lane

ORION should remain downstream of the first E001 controls. Once the density benchmark is stable on raw
and frozen-foundation representations, ORION neural tokens can become an additional representation source:

```text
neurOS SignalFrame
   -> ORION tokens
   -> QuantumBCI density/open-system state
   -> matched classical controls
   -> neuros-mechint interventions
```

That experiment would ask whether quantum-structured geometry adds value **on top of a neurOS-native neural
representation**, rather than crediting QuantumBCI for representation quality learned elsewhere.

## Promotion strategy for both projects

The combined path should be the flagship tutorial for each repository:

1. record or load neural data with neurOS;
2. freeze leakage-resistant neurOS evidence partitions;
3. obtain specialist/foundation/ORION representations;
4. test a QuantumBCI mechanism against matched classical controls;
5. run interventions through neurOS-mechint;
6. emit one evidence pack containing both source revisions and all fingerprints.

That creates a genuine ecosystem story: neurOS becomes the reliable neural research substrate, while
QuantumBCI becomes a compelling mechanism laboratory that demonstrates why those substrate contracts matter.
