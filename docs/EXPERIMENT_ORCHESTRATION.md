# Experiment orchestration architecture

Last reviewed: 2026-08-24

QuantumBCI experiments should behave like scientific build systems. A run is valid only when its inputs, split policy, source revision, mechanism definition, controls, metrics, and produced artifacts are all addressable after the fact.

## 1. The orchestration contract

Every experiment is a directed acyclic graph (DAG) of stages. Stages declare dependencies, resource hints, a tokenized command, and artifacts they own. The orchestrator validates the DAG before any work starts and produces a deterministic run plan.

A run identity should be derived from:

`SHA256(canonical_manifest + source_commit + dataset_fingerprint + split_registry)`

The first 12-16 hex characters are human-friendly; the full digest remains in the run metadata.

No stage may infer train/test membership from file placement or notebook state. `split_registry.json` is an input artifact and subject/session groups are immutable for a run family.

## 2. Required artifact ledger

Each completed run should eventually materialize:

```text
runs/<experiment>/<run_id>/
├── manifest.json
├── source.json
├── environment.json
├── dataset_fingerprint.json
├── split_registry.json
├── stage_status.json
├── predictions.*
├── metrics.json
├── mechanism_observables.json
├── evidence_ledger.json
├── resource_ledger.json
└── report.md
```

Large arrays and checkpoints may live in object storage; the ledger stores content hashes and immutable URIs rather than copying them into Git.

## 3. Execution tiers

### Tier A: local / GitHub Actions

Use for schema validation, synthetic parameter recovery smoke tests, unit tests, tiny deterministic fixtures, and manifest planning. CI must not download protected clinical data or multi-GB checkpoints.

### Tier B: GPU extraction

Foundation-model inference is an expensive but cacheable stage. Extract embeddings once per `(dataset fingerprint, preprocessing contract, encoder checkpoint, source revision)` and make all representation experiments consume the same immutable cache. This prevents one representation from quietly receiving different preprocessing.

### Tier C: CPU fan-out

Classical controls, readouts, bootstrap resampling, ablations, and many low-dimensional dynamics fits should fan out as CPU jobs. These stages are embarrassingly parallel and should be scheduled as arrays on Slurm/Kubernetes/cloud batch when the experiment grows.

### Tier D: QPU

QPU work is a terminal gated lane, never an exploratory default. Follow the current Qiskit pattern: map the promoted observable, optimize for the target backend, execute, then analyze. The backend calibration snapshot and transpiled circuit statistics are evidence artifacts.

## 4. Recommended executor interface

Keep the core manifest engine platform-neutral. Executors should consume the same planned stage object:

- `LocalExecutor`: subprocess execution, developer smoke tests.
- `GitHubActionsExecutor`: CI-only synthetic and validation lanes.
- `SlurmExecutor`: GPU extraction and CPU arrays on academic clusters.
- `KubernetesExecutor`: containerized research infrastructure when needed.
- `QiskitExecutor`: simulator/QPU stages only after E004 gates pass.

Do not make Weights & Biases, MLflow, a cloud vendor, or a scheduler part of the scientific data model. They can mirror events, while the canonical evidence ledger remains portable.

## 5. Cache and invalidation rules

Embedding cache keys must include encoder checkpoint hash and preprocessing parameters. Representation cache keys must additionally include mechanism hyperparameters. Readout caches must include split registry, seed, hyperparameter budget, and representation artifact hash.

A changed subject split invalidates everything downstream. A changed readout must not force re-extraction of frozen embeddings. A report-only change does not invalidate model artifacts.

## 6. Failure semantics

The orchestrator should fail closed:

- missing dependency artifact -> block stage;
- checksum mismatch -> invalidate stage and descendants;
- unknown dataset license/consent scope -> block data stage;
- subject leakage detector failure -> invalidate run;
- non-finite metrics or non-physical density-state invariant -> fail stage;
- failed scientific gate -> downstream promoted stages become `not_eligible`, not `failed`.

This distinction matters. A scientifically negative experiment is a successful run with a negative gate decision.

## 7. Statistical orchestration

Model selection happens only inside training subjects. Test subjects stay untouched until the final evaluation stage. Bootstrapping is over the independent unit of generalization, normally subject, not over arbitrarily many correlated EEG windows.

Every leaderboard-like scalar should be accompanied by the paired subject-level distribution, confidence interval, and strongest matched control. Mechanism observables get their own stability report across resamples and seeds.

## 8. Promotion ladder

```text
mathematical validity
        ↓
synthetic identifiability
        ↓
within-dataset held-out prediction
        ↓
cross-subject / few-shot robustness
        ↓
mechanism intervention + stability
        ↓
cross-dataset replication
        ↓
quantum algorithm resource study (optional)
        ↓
physical-quantum experiment (separate evidence class)
```

A later rung can never retroactively repair a failed earlier rung.

## 9. Near-term implementation sequence

1. Merge manifest/DAG validator and deterministic plan renderer.
2. Implement data-contract + split-registry tasks with synthetic fixtures first.
3. Add MNE/BIDS dataset adapters without embedding dataset-specific split logic in models.
4. Add encoder adapters and immutable embedding caches.
5. Implement E001 fully and lock the statistical report schema.
6. Reuse E001 trajectories/caches for E002.
7. Keep E003 prospective collection behind preregistration/ethics gates.
8. Implement E004 only after a prior experiment has a promoted observable.
