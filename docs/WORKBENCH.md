# QuantumBCI Research Workbench

QuantumBCI has two layers:

1. a dependency-light mechanism library for density states, open-system dynamics,
   contextual models, QFT semantics and classical controls;
2. a local-first **research workbench** that turns those pieces into reproducible
   runs with configs, explicit splits, artifacts and inspection commands.

The workbench is intentionally usable without neurOS or Qiskit. Optional neurOS
packages add runtime/evidence authority; optional quantum packages add circuit
backends. Neither is required to understand or test the core mechanism code.

## Five-minute path

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

quantumbci init
quantumbci doctor
quantumbci smoke
quantumbci runs list
```

`quantumbci smoke` creates a complete local run under `.quantumbci/runs/` and prints
the path to the artifact bundle. Open its `report.html` in any browser.

The smoke study is deliberately synthetic. Its labels are encoded in cross-feature
correlation, so density geometry should recover the signal while an off-diagonal
ablation should damage it. This is an end-to-end mechanism recovery check, not an
empirical neuroscience result.

## Configuration

`quantumbci init` writes:

```json
{
  "schema_version": 1,
  "artifact_root": ".quantumbci/runs",
  "default_seed": 0,
  "source_sha": "working-tree"
}
```

For a serious study, replace `source_sha` with the exact Git commit being executed.
Paths in a config file are resolved relative to the config file itself.

You can keep multiple configs for different projects or storage roots:

```bash
quantumbci doctor --config configs/my-study.json
quantumbci smoke --config configs/my-study.json
quantumbci runs list --config configs/my-study.json
```

## Commands

### Environment readiness

```bash
quantumbci doctor
quantumbci doctor --json
```

The doctor reports Python/NumPy, the artifact root, Qiskit/Aer availability and the
installed versions of `neuros-core`, `neuros-foundation` and `neuros-mechint`.
Optional packages being absent is not an error; the report tells you which capability
surface is actually available.

### Synthetic end-to-end study

```bash
quantumbci smoke
quantumbci smoke --seed 7
quantumbci smoke --output-root /tmp/qbci-runs --json
```

Each run owns:

```text
run.json
study_manifest.json
metrics.json
mechanism.json
predictions.jsonl
artifact_hashes.json
report.md
report.html
```

The run record separately carries software completion state, claim class, evidence
tier, source identity, deterministic scientific fingerprint, seed and summary metrics.
A completed smoke run is always `synthetic_sanity`; it cannot be promoted to physical
quantum evidence.

### Benchmark frozen embeddings without writing Python

If an encoder already produced NumPy arrays, run the matched comparison directly:

```bash
quantumbci benchmark embeddings.npy labels.npy \
  --train-indices train_indices.npy \
  --test-indices test_indices.npy \
  --split-name subject-exclusive-v1 \
  --output density_result.json
```

`embeddings.npy` must have shape `examples × tokens × features`. QuantumBCI deliberately
requires explicit train and test index arrays rather than silently creating a random
split. Add `--include-predictions --json` when you want the full machine-readable result.

### Inspect local runs

```bash
quantumbci runs list
quantumbci runs show <RUN_ID>
quantumbci runs show <RUN_ID> --json
```

The filesystem is the source of truth. There is no hidden database required to recover
a run.

### Experiment contracts

From a source checkout:

```bash
quantumbci experiments list
quantumbci experiments validate experiments/manifests/E001_density_geometry.json

quantumbci experiments plan \
  experiments/manifests/E001_density_geometry.json \
  --source-sha "$(git rev-parse HEAD)" \
  --output .quantumbci/plans/E001
```

Planning does not execute an experiment. It validates the DAG, binds the manifest to a
source revision and materializes a portable plan ledger. Real-data execution then adds
the dataset and split authority before a scientific run ID is assigned.

## Use your own frozen embeddings from Python

The same path is available as a dependency-light API:

```python
import numpy as np

from quantumbci.benchmarking import IndexSplit, benchmark_density_embeddings

# embeddings: examples x tokens x features
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

The helper fits the same low-capacity readout family to the full density
representation, a diagonal-only density control and a pooled mean/std classical
control. It then evaluates the **already fitted density readout** after deleting density
off-diagonals. The ablation is therefore an intervention, not a separately refit model.

The function deliberately never creates a random train/test split for you. For promoted
work, use neurOS or another immutable evidence authority and adapt it with:

```python
split = IndexSplit.from_partition(neuros_partition)
```

## neurOS workflow

With neurOS installed, the intended full path is:

```text
neurOS data/replay + evidence authority
        ↓
frozen neurOS/foundation embeddings
        ↓
QuantumBCI matched representation benchmark
        ↓
SourceWeigher adaptation adversary
        ↓
QuantumBCI mechanism interventions
        ↓
neuros-mechint evidence / replication
```

Install released compatible packages with:

```bash
pip install -e '.[neuros]'
```

During active co-development, use the exact pinned sibling neurOS packages described in
[`NEUROS_INTEGRATION.md`](NEUROS_INTEGRATION.md).

## Evidence discipline

The convenience layer never changes the scientific claim taxonomy:

- synthetic smoke tests qualify software and mechanism recovery;
- public neural datasets can support predictive and quantum-inspired mechanism claims;
- quantum hardware experiments require a resource ledger;
- physical quantum neural claims require independent operational evidence about the
  physical substrate.

Usability should make rigorous experiments easier to run, not make weak evidence easier
to overstate.
