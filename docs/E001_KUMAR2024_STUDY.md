# E001 on Kumar2024

This is the first real-dataset execution path for QuantumBCI's equivalence-first E001 program.
It uses the merged neurOS longitudinal evidence authority and the public MOABB Kumar2024 motor-
imagery dataset.

## What this study asks

The study **does not** ask whether the current density constructor contains more information than
covariance. v0.6 already establishes that

```text
rho = X^H X / Tr(X^H X)
```

is exactly a trace-normalized Hermitian second moment.

Instead, Kumar2024 is used to answer a stricter and more useful set of questions:

1. can QuantumBCI preserve exact longitudinal source/calibration/evaluation authority on real EEG?
2. how do trace normalization, covariance geometry, pooled statistics, PCA and operator
   interventions behave under cross-day drift and increasing target calibration?
3. do the same conclusions hold for both original Kumar2024 training cohorts (GR and PAR)?
4. can every result be traced from original downloaded bytes through processed-data authority,
   representation bytes, source revisions and participant-level inference?
5. can the final evidence object be independently verified and exported without redistributing raw
   participant data inside the QuantumBCI artifact?

A result can be scientifically useful even when the density-information novelty gate remains false.

## Dataset boundary

The neurOS Kumar2024 specification fixes the MOABB interpretation used here:

- dataset id: `moabb-kumar2024`;
- 18 participants;
- six separate-day sessions, ordered `0` through `5` in MOABB metadata;
- left-hand versus right-hand motor imagery;
- subjects 1-9 preserve the original `GR` cohort label;
- subjects 10-18 preserve the original `PAR` cohort label;
- 8-30 Hz analysis band by default.

MOABB includes the bar-feedback runs and excludes the car-racing runs from the online sessions.
The source dataset is distributed through Zenodo and is recorded by neurOS as CC BY 4.0.

## Exact compatibility with the neurOS model ladder

QuantumBCI deliberately reproduces the sample-authority semantics of the merged neurOS model-ladder
runner rather than creating a look-alike split.

For one subject and target session, the split seed is

```text
first_32_bits(SHA256("<base>|moabb-kumar2024|<subject>|<target-session>"))
```

with base seed `2026` by default.

The case then uses:

```text
source history   = every chronologically prior session
target session   = one held-out session
calibration pool = deterministic class-stratified subset of target session
final evaluation = deterministic, immutable remainder of target session
```

The default evaluation fraction is `0.5`, and every calibration budget uses the same final
evaluation examples.

The resulting `LongitudinalCaseAuthority` case ID follows the same form as neurOS:

```text
moabb-kumar2024/subject-<N>/session-<S>/split-<stable-seed>
```

Given identical MOABB processing parameters and source data, this allows a QuantumBCI case to share
the same authority fingerprint as the neurOS specialist/frozen/SourceWeigher ladder.

## Representation under test

The first lane intentionally avoids pretending a pooled neural-decoder embedding has tokens.
MOABB epochs arrive as:

```text
samples x EEG channels x time
```

QuantumBCI transposes each epoch to:

```text
time tokens x EEG-channel features
```

and runs E001 on that real token surface.

This makes the first real study primarily an **EEG covariance/operator geometry and evidence-
authority validation study**. Foundation-model token lanes can be added later without changing the
sample authority.

## Controls

Each calibration point uses the v0.6 adversarial E001 suite on exactly the same epoch tensor:

- density operator;
- exact trace-normalized covariance;
- ordinary covariance;
- log-covariance geometry;
- bilinear second moment;
- pooled mean/std;
- train-only flattened PCA;
- diagonal density;
- fixed-readout off-diagonal deletion.

Density and trace-normalized covariance must remain prediction-identical. Off-diagonal deletion is
interpreted as a cross-feature covariance intervention, not microscopic quantum coherence.

## Raw-data fingerprint

Before MOABB preprocessing is consumed as evidence, the study calls the dataset's public
`data_path(subject)` API for every selected participant and hashes the original local source files.

The raw-source manifest stores only:

- a stable relative/basename label;
- byte count;
- SHA-256;
- per-subject content fingerprint;
- aggregate selected-dataset fingerprint.

Absolute local paths are not serialized. Moving byte-identical source data to another machine does
not change the scientific dataset fingerprint.

This raw-source fingerprint is separate from neurOS's downstream `processed_data_sha256`, partition
fingerprint and calibration-split fingerprint.

## Full scientific identity

Every case binds:

1. aggregate raw-source content fingerprint;
2. neurOS processed-data SHA-256;
3. neurOS authority fingerprint;
4. neurOS partition fingerprint;
5. neurOS calibration-split fingerprint;
6. exact time-by-channel representation SHA-256;
7. representation/preprocessing identifier;
8. calibration frontier and benchmark configuration;
9. exact QuantumBCI source revision;
10. exact neurOS source revision.

The study-level fingerprint additionally binds the complete set of case authority and case-study
fingerprints.

## Participant-level inference

The independent unit is the participant, not the EEG epoch.

For every calibration budget and control:

1. density-minus-control deltas from repeated target sessions are averaged within participant;
2. the participant set must be identical across the calibration frontier;
3. participants are bootstrap-resampled with replacement;
4. the observed paired mean, 95% bootstrap interval and bootstrap probability above zero are
   recorded.

The normalized-covariance delta should be identically zero because that control is information-
equivalent to the present density constructor.

## Local execution

Install QuantumBCI plus the real EEG evidence profile, or co-develop against a pinned neurOS
workspace:

```bash
pip install -e '.[real-eeg]'
```

Then run a two-participant final-session study:

```bash
quantumbci-kumar2024 \
  --subjects 1,10 \
  --held-out-sessions 5 \
  --budgets 0,1,2,5,10 \
  --quantumbci-source-sha "$(git rev-parse HEAD)" \
  --neuros-source-sha <PINNED_NEUROS_SHA> \
  --output .quantumbci/studies/E001-kumar2024
```

Run every longitudinal target session after the cohort checkpoint has qualified:

```bash
quantumbci-kumar2024 \
  --subjects 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18 \
  --all-target-sessions \
  --budgets 0,1,2,5,10 \
  --quantumbci-source-sha "$(git rev-parse HEAD)" \
  --neuros-source-sha <PINNED_NEUROS_SHA> \
  --output .quantumbci/studies/E001-kumar2024-full
```

The source-checkout compatibility wrapper remains available at
`scripts/evidence/run_kumar2024_e001.py`.

## GitHub Actions execution profiles

The workflow **E001 Kumar2024 real study** is `workflow_dispatch` only. Opening a pull request never
downloads Kumar2024.

It provides three scopes:

### `smoke`

Defaults to subjects `1,10` and target session `5`. This crosses the GR/PAR cohort boundary while
keeping the first real download and compute surface small.

### `cohort`

Runs all 18 participants on target session `5`. This is the recommended first complete empirical
checkpoint because it provides a full participant-level cohort comparison without multiplying the
expensive train-only PCA control over every target session.

### `full-longitudinal`

Runs all 18 participants over target sessions `1-5`. This is the complete deployment-history
frontier and should be run after the cohort artifact is inspected and accepted.

All three profiles pin neurOS to the same merged authority revision, verify the finished bundle,
create an RO-Crate archive and upload only derived evidence. Raw EEG files are not included in the
GitHub artifact.

## Evidence bundle

A successful run writes:

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

Because the directory also contains a normal `run.json`, it can be exported directly as an
RO-Crate with `export_run_ro_crate(...)`.

## Interpretation ceiling

Even the full 18-participant × five-target-session study remains **quantum-inspired, offline,
real-dataset evidence**.

It cannot establish:

- extra information in the current density constructor beyond normalized covariance;
- microscopic neural quantum coherence;
- entanglement;
- quantum computation in brain tissue;
- closed-loop BCI superiority;
- clinical efficacy.

The most valuable outcome of this study is a trustworthy empirical base from which a genuinely
non-equivalent operator, dynamical or contextual mechanism can be tested next.
