# E001 on Kumar2024

This is QuantumBCI's first package-owned real-dataset execution path. It combines the public MOABB
Kumar2024 motor-imagery dataset with the merged neurOS longitudinal evidence authority and the
v0.6 equivalence-first E001 benchmark.

## Scientific question

This study does **not** ask whether the current density constructor contains more information than
covariance. That question is already settled for the current constructor:

```text
rho = X^H X / Tr(X^H X)
```

After optional centering, this is exactly a trace-normalized Hermitian second moment. The density
representation is therefore information-equivalent to the corresponding normalized covariance.

Kumar2024 is used for the more useful empirical questions:

1. can QuantumBCI preserve exact source-history, calibration and final-evaluation authority on real
   longitudinal EEG?
2. how do normalization, covariance geometry, pooled statistics, PCA and off-diagonal interventions
   behave under cross-day drift and increasing target calibration?
3. do those patterns remain visible across the original GR and PAR participant cohorts?
4. can every result be traced from selected raw source bytes through neurOS processed-data identity,
   authority, representation bytes, code revisions and participant-level inference?
5. can the resulting evidence object be independently checksum-verified and exported as RO-Crate
   without redistributing raw participant EEG?

A negative or equivalence-preserving result is a successful scientific run.

## Dataset contract

The pinned neurOS/MOABB interpretation is:

- dataset id: `moabb-kumar2024`;
- 18 participants;
- six separate-day sessions, represented as `0` through `5` by MOABB;
- left-hand versus right-hand motor imagery;
- MOABB subjects 1-9 are the original GR cohort;
- MOABB subjects 10-18 are the original PAR cohort;
- default analysis band: 8-30 Hz;
- native sampling rate: 512 Hz unless an explicit resample rate is supplied.

Current MOABB loads only the bar-feedback runs. Racing-game files distributed in the same archive
are not part of this declared study surface.

## Raw-source fingerprint

Kumar2024 is distributed as one Zenodo ZIP. Current MOABB `data_path(subject)` returns the same
extracted dataset root for every selected subject. Subject-specific selection happens inside the
loader.

QuantumBCI mirrors that loader boundary rather than hashing the whole archive once per participant.
The fingerprint adapter:

1. verifies every selected MOABB subject resolves to one shared extracted root;
2. applies the current MOABB subject mapping:
   - subjects 1-9 -> raw subjects 1-9;
   - subjects 10-18 -> raw subjects 11-19;
3. selects only subject-specific GDF files under:

```text
Offline/<GR|PAR>/<subject>/**/*.gdf
Online/<GR|PAR>/<subject>/**/*.gdf
```

4. explicitly excludes:

```text
Race/**
```

5. hashes each unique selected file once with SHA-256;
6. derives per-subject content fingerprints;
7. derives one aggregate fingerprint for the selected study cohort;
8. records stable relative file names and byte counts, never machine-specific absolute paths.

Changing a selected Offline/Online GDF file changes the appropriate participant and aggregate
fingerprints. Changing an unused `Race/**` file does not.

This upstream fingerprint is intentionally separate from neurOS `processed_data_sha256`, partition
fingerprints and calibration-split fingerprints.

## Exact neurOS authority compatibility

QuantumBCI reuses the same prior-session protocol as the merged neurOS model ladder.

For each subject and target session, the split seed is:

```text
first_32_bits(SHA256("<base>|moabb-kumar2024|<subject>|<target-session>"))
```

with base seed `2026` by default.

The case authority is:

```text
source history   = every chronologically prior session
target session   = one held-out session
calibration pool = deterministic class-stratified subset of the target session
final evaluation = deterministic immutable remainder of the target session
```

The default evaluation fraction is `0.5`. Every calibration budget shares the exact same final
evaluation examples.

The case ID follows the same neurOS form:

```text
moabb-kumar2024/subject-<N>/session-<S>/split-<stable-seed>
```

`LongitudinalCaseAuthority.restore(data)` is called before the benchmark consumes the case. That
revalidates processed neural bytes and the frozen evidence assignment.

## Representation under test

MOABB epochs arrive as:

```text
examples x EEG channels x time
```

The first QuantumBCI lane uses the real temporal samples as tokens:

```text
time tokens x EEG-channel features
```

No artificial token axis is created around a pooled decoder output.

This makes the first real study an EEG covariance/operator-geometry and evidence-authority study.
Foundation-model token lanes can later consume the same sample authority without changing the
scientific split.

## Prepared feature contract

Real EEG epochs are wide, so recomputing every representation transform at every calibration budget
would be both expensive and scientifically ambiguous. v0.6.1 makes the adaptation boundary explicit.

For one participant tensor, QuantumBCI prepares the budget-independent E001 features once:

- density;
- exact normalized covariance;
- ordinary covariance;
- log-covariance geometry;
- bilinear second moment;
- pooled mean/std;
- diagonal density;
- fixed off-diagonal-deletion feature surface.

For each target-session case, the flattened PCA control is fit **once on chronological source
history only**. It is then frozen for every target-calibration budget.

The adaptation contract is:

```json
{
  "static_feature_scope": "prepared_once_per_participant_tensor",
  "pca_fit_scope": "source_history_only",
  "target_calibration_changes": "readout_only",
  "final_evaluation_in_representation_fit": false
}
```

Target calibration may update the matched low-capacity readout. It does not refit PCA or any other
representation transform. Final evaluation examples never enter representation fitting.

This contract is written into:

- `run.json`;
- `study_manifest.json`;
- `representation_index.json`;
- `evidence_ledger.json`;
- `report.md`.

The artifact ledger is recomputed after those fields are written.

## E001 controls

Each calibration point evaluates the same prepared tensor surface with:

- density operator;
- exact trace-normalized covariance;
- ordinary covariance;
- log-covariance geometry;
- bilinear second moment;
- pooled mean/std;
- source-history-frozen flattened PCA;
- diagonal density;
- fixed-readout off-diagonal deletion.

Density and exact normalized covariance must remain prediction-identical. If that invariant breaks,
the benchmark fails rather than silently changing the scientific interpretation.

Off-diagonal deletion is interpreted as removal of cross-feature covariance structure. It is not a
microscopic quantum-coherence witness.

## Full scientific identity

Each case binds:

1. aggregate selected raw-source fingerprint;
2. neurOS processed-data SHA-256;
3. neurOS authority fingerprint;
4. neurOS partition fingerprint;
5. neurOS calibration-split fingerprint;
6. exact time-by-channel representation SHA-256;
7. representation and preprocessing identity;
8. canonical calibration frontier;
9. benchmark regularization parameters;
10. PCA fit scope and dimension;
11. exact QuantumBCI source revision;
12. exact neurOS source revision.

The study-level fingerprint additionally binds the complete set of case authority and case-study
fingerprints.

## Participant-level inference

The independent inference unit is the participant, not the EEG window or trial.

For every calibration budget and control:

1. repeated held-out-session deltas are averaged within participant;
2. the participant set must be identical across budgets;
3. duplicate case rows fail closed;
4. participants are bootstrap-resampled with replacement;
5. the paired mean density-minus-control delta and 95% bootstrap interval are recorded.

The density-minus-normalized-covariance delta is expected to be exactly zero because those two
representations are information-equivalent under the present constructor.

## Local execution

Install the real EEG profile:

```bash
pip install -e '.[real-eeg]'
```

Run the smallest GR/PAR-crossing checkpoint:

```bash
quantumbci-kumar2024 \
  --subjects 1,10 \
  --held-out-sessions 5 \
  --budgets 0,1,2,5,10 \
  --quantumbci-source-sha "$(git rev-parse HEAD)" \
  --neuros-source-sha <PINNED_NEUROS_SHA> \
  --output .quantumbci/studies/E001-kumar2024
```

The source-checkout wrapper remains available at:

```text
scripts/evidence/run_kumar2024_e001.py
```

## GitHub Actions separation

### `E001 Kumar2024 contract`

Runs on pull requests and main pushes, but **never downloads Kumar2024**. It installs the exact
pinned merged neurOS authority and qualifies:

- the installed `quantumbci-kumar2024` command;
- routing through the canonical cached executor;
- synthetic six-session Kumar-style data under real neurOS authority classes;
- source-history-frozen PCA semantics;
- density/normalized-covariance identity;
- participant bootstrap;
- closed-world bundle verification;
- RO-Crate export;
- wheel inclusion of every real-study module.

### `E001 Kumar2024 real study`

This workflow is `workflow_dispatch` only. Opening a pull request cannot trigger a public dataset
download.

It exposes three explicit scopes:

- `smoke`: default subjects `1,10`, target session `5`;
- `cohort`: all 18 participants, target session `5`;
- `full-longitudinal`: all 18 participants, target sessions `1-5`.

The recommended empirical ladder is:

```text
smoke -> inspect evidence -> cohort -> inspect cohort effects -> full-longitudinal
```

The cohort checkpoint is valuable even after the caching optimization because it provides a clean
18-participant cross-cohort result before multiplying the number of longitudinal target cases.

The real workflow verifies the finished evidence directory, creates an RO-Crate archive and uploads
only derived study artifacts. Raw EEG is not included in the GitHub artifact.

## Evidence bundle

A successful study writes:

```text
run.json
study_manifest.json
source_revisions.json
dataset_fingerprint.json
neuros_authority.json
representation_index.json
case_results.json
results.csv
predictions.jsonl
bootstrap_metrics.json
evidence_ledger.json
report.md
artifact_hashes.json
```

`artifact_hashes.json` closes the directory under QuantumBCI's closed-world verifier. Missing,
modified or unexpected top-level files invalidate the bundle.

Because the directory contains a normal `run.json`, it can be exported directly with
`export_run_ro_crate(...)`.

## Interpretation ceiling

Even a completed 18-participant × five-target-session study remains **quantum-inspired, offline,
real-dataset evidence**.

It cannot establish:

- additional information in the current density constructor beyond normalized covariance;
- microscopic neural quantum coherence;
- entanglement;
- quantum computation in neural tissue;
- closed-loop BCI superiority;
- clinical efficacy.

The purpose of this study is to establish a trustworthy empirical base for the next genuinely
non-equivalent operator, dynamical or contextual mechanism.
