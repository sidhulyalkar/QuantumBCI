# Public research workflows

QuantumBCI is intended to be useful outside a single repository, lab, model family, or neural dataset. The public workflow therefore separates **data authority**, **representation production**, **mechanism testing**, and **evidence sharing**.

## The portable workflow

```text
BIDS / MNE / MOABB / neurOS / custom dataset
                  |
                  v
      frozen representation tensor
   (LaBraM, EEGPT, neurOS, ORION, custom)
                  |
                  v
       quantumbci recipe validate
                  |
                  v
          quantumbci recipe run
                  |
                  v
 density + matched controls + intervention
                  |
                  v
   local immutable evidence/run bundle
          |                  |
          v                  v
   RO-Crate export       BIDS-aware export
          |                  |
          +--------+---------+
                   v
        another lab / archive / paper
```

A recipe is deliberately small. It does not own dataset downloading or foundation-model training. It records the frozen interface between those systems and the QuantumBCI mechanism benchmark.

## 1. Create a recipe

```bash
quantumbci recipe init study.json
```

The template contains:

```json
{
  "schema_version": 1,
  "id": "my-density-study",
  "title": "Density geometry on frozen neural embeddings",
  "claim_class": "quantum_inspired",
  "evidence_tier": "exploratory",
  "source_dataset": "replace-with-dataset-id-or-URL",
  "source_model": "replace-with-model-id-and-revision",
  "data": {
    "embeddings": "embeddings.npy",
    "labels": "labels.npy",
    "train_indices": "train_indices.npy",
    "test_indices": "test_indices.npy",
    "split_name": "subject-exclusive-v1"
  },
  "benchmark": {
    "ridge": 0.001
  }
}
```

Paths are resolved relative to the recipe file. The four inputs are SHA-256 fingerprinted during validation and run identity construction.

QuantumBCI does not create a random split for a recipe. The train/test index files are part of the scientific contract.

## 2. Validate before running

```bash
quantumbci recipe validate study.json
quantumbci recipe validate study.json --json
```

Validation checks that inputs exist, the claim ceiling remains `quantum_inspired`, the benchmark parameters are valid, and all input files can be fingerprinted.

This is also a useful handoff check when another lab sends you a recipe plus separately distributed embeddings: compare the input SHA-256 values before executing anything.

## 3. Run into the normal evidence registry

```bash
quantumbci recipe run study.json --config quantumbci.json
```

The resulting run contains:

```text
run.json
recipe.json
inputs.json
metrics.json
predictions.jsonl
artifact_hashes.json
report.md
report.html
```

`inputs.json` records file hashes and array shapes. `artifact_hashes.json` protects the generated evidence files. `run.json` carries the final scientific fingerprint and claim/evidence metadata.

The recipe runner intentionally does **not** copy the source embeddings into the run bundle. Neural embeddings may be large, licensed, restricted, or derived from participant data. The evidence bundle records hashes and provenance so the source tensors can be distributed through the appropriate data channel.

## 4. Verify a run later

```bash
quantumbci runs verify <RUN_ID>
```

Verification recomputes every hash in `artifact_hashes.json`. Missing or modified files make the run invalid.

Both public export paths require a valid artifact ledger. This prevents a report from being edited after the fact and then shared under the original scientific fingerprint.

## 5. Export a Research Object

```bash
quantumbci runs export <RUN_ID> \
  --format ro-crate \
  --output shared/my-study
```

Add `--archive` to also create a zip package.

The export follows the minimal RO-Crate 1.3 structure:

```text
my-study/
├── ro-crate-metadata.json
├── ro-crate-preview.html
└── data/
    └── <QuantumBCI run artifacts>
```

The metadata document uses the RO-Crate 1.3 context and describes the root Dataset, QuantumBCI software, and each evidence file. The HTML preview gives humans a zero-server landing page.

QuantumBCI does not claim external RO-Crate validator certification. Publication pipelines should run the validator required by their repository or institution.

## 6. Place evidence beside a BIDS dataset

```bash
quantumbci runs export <RUN_ID> \
  --format bids \
  --output /path/to/bids-dataset \
  --bids-version <YOUR_DATASET_BIDS_VERSION> \
  --source-dataset-url https://example.org/dataset
```

This creates:

```text
<bids-root>/
└── derivatives/
    └── quantumbci/
        ├── dataset_description.json
        └── evidence/
            └── <RUN_ID>/
                └── <QuantumBCI evidence bundle>
```

The derivative-level `dataset_description.json` records `DatasetType=derivative`, `GeneratedBy=QuantumBCI`, the explicit BIDS version supplied by the researcher, and an optional source-dataset URL.

The generic QuantumBCI evidence files are **not** claimed to implement a modality-specific standardized BIDS derivative datatype. This is a BIDS-aware provenance/discovery container, designed to coexist safely with standardized EEG/iEEG derivatives.

## Why BIDS, MOABB, and RO-Crate fit this project

BIDS derivatives are designed to preserve enough metadata for processed outputs to be understood and reused in later processing. A derivatives dataset also has a dataset-level description and may point back to source datasets. That makes BIDS a natural home for neural-data provenance without forcing QuantumBCI to own raw EEG storage.

MOABB already separates datasets/paradigms from evaluations/pipelines and provides cross-session and cross-subject benchmarking across a large open EEG catalog. QuantumBCI should consume those evidence authorities or their frozen outputs rather than duplicate the benchmark ecosystem.

RO-Crate addresses a different boundary: transporting a research object, its files, software/provenance context, and a human-readable preview. That makes it a natural publication/handoff format for a completed QuantumBCI evidence run.

## Recommended world-facing integrations

### Neuroscience laboratories

Use BIDS/MNE or neurOS for raw neural data and participant/session identity. Produce frozen tensors once, then test QuantumBCI mechanisms with fixed evidence authority.

### Foundation-model developers

Publish a small frozen representation benchmark pack for a model revision. QuantumBCI recipes can then test whether density/open-system/contextual mechanisms reveal structure not captured by ordinary probes.

### BCI benchmark maintainers

Treat a QuantumBCI mechanism as another representation/pipeline family inside a fixed MOABB evaluation, not as a bespoke dataset split. This makes comparisons interpretable.

### Mechanistic-interpretability researchers

Use QuantumBCI to define representation-level interventions, then route promoted hypotheses into `neuros-mechint` or another causal-evidence framework.

### Quantum-computing researchers

Start only from an observable that survives the classical/quantum-inspired evidence ladder. Export the full upstream evidence object alongside circuit/resource accounting so the quantum mapping cannot become detached from the neural result that motivated it.

### Reproducibility and archival workflows

Archive RO-Crates rather than screenshots or notebook state. The run fingerprint, source tensor hashes, artifact hashes, and machine-readable metrics remain available even when the original interactive environment is gone.

## A useful public benchmark pattern

A strong community contribution is not simply “our model achieved X%.” A reusable QuantumBCI contribution should ideally publish:

1. dataset identifier/version and legal access instructions;
2. preprocessing contract;
3. model/checkpoint revision;
4. frozen representation hashes;
5. immutable train/calibration/evaluation authority;
6. QuantumBCI recipe;
7. matched controls;
8. intervention results;
9. participant-level or other appropriate inference unit statistics;
10. exported evidence Research Object.

That bundle lets another group reproduce, falsify, or extend the mechanism without adopting the original lab's entire software stack.
