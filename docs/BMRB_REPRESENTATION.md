# BMRB-Representation

`BMRB_REPRESENTATION_V1` asks whether the same authority-bound neural mechanism contrast is conserved when the frozen representation changes.

The benchmark is intentionally narrower than a claim that one representation is "better" or that a foundation model discovers quantum structure. Its core question is:

> Does a declared candidate effect, including the consequence of ablating that candidate, recur across the same participants and held-out cases in raw neural and independently frozen learned representation spaces after matched classical controls?

## Scientific separation

BMRB-Representation keeps two questions separate:

1. **Conservation**: does the effect recur across representation spaces?
2. **Information novelty**: does the candidate survive the strongest matched classical representation control?

A density representation can therefore be perfectly conserved across raw EEG, LaBraM and EEGPT while still failing the adversary gate if it remains mathematically equivalent to normalized covariance.

That outcome is scientifically useful. It says the second-moment structure is conserved, not that a uniquely quantum representation has been discovered.

## Evidence authority

Every lane must be built from `LongitudinalE001CaseResult` records under the same neurOS authority. BMRB-Representation v1 requires exact equality of:

```text
participant
occasion / held-out session
case_id
calibration_per_class
authority_fingerprint
```

across all lanes.

The benchmark does not silently intersect mismatched participant sets. Missing pairs change the estimand and fail closed.

## Create a raw-neural lane

The existing Kumar2024 study already runs raw time-by-channel E001 representations under merged neurOS longitudinal authority. For a reusable BMRB lane, collect the resulting `LongitudinalE001CaseResult` objects and write:

```python
from quantumbci import write_e001_representation_lane_bundle

write_e001_representation_lane_bundle(
    raw_cases,
    "lanes/raw",
    study_id="kumar2024-raw",
    representation_family="raw_neural",
)
```

The output is closed-world and integrity checked:

```text
lanes/raw/
  run.json
  study_manifest.json
  case_results.json
  report.md
  artifact_hashes.json
```

## Create a frozen foundation-model lane

QuantumBCI does not download or train a foundation model implicitly. The representation must be frozen first.

A neurOS adapter can be wrapped with `NeurOSFoundationEncoder`, then applied independently to epochs:

```python
from quantumbci import encode_frozen_epochs
from quantumbci.integrations.neuros import NeurOSFoundationEncoder

encoder = NeurOSFoundationEncoder.from_registry("<model-id>")
embeddings = encode_frozen_epochs(
    epochs,
    encoder,
    sample_rate_hz=sample_rate_hz,
)
```

`encode_frozen_epochs` invokes only the encoder's `encode` operation. It does not call `fit`, `adapt` or `train`.

Use the exact same neurOS `LongitudinalCaseAuthority` that was used for the raw lane:

```python
from quantumbci import run_longitudinal_e001_case

case = run_longitudinal_e001_case(
    data,
    authority,
    embeddings,
    representation_id="labram:<exact-model-revision>",
    budgets_per_class=(0, 1, 2, 5, 10),
    upstream_dataset_fingerprint=raw_dataset_fingerprint,
    quantumbci_source_sha=quantumbci_sha,
    neuros_source_sha=neuros_sha,
)
```

Then serialize the lane with pinned model identity:

```python
write_e001_representation_lane_bundle(
    labram_cases,
    "lanes/labram",
    study_id="kumar2024-labram",
    representation_family="foundation_model",
    model_id="LaBraM",
    model_revision="<exact revision>",
)
```

A `foundation_model` lane without both `model_id` and `model_revision` is rejected by the aggregate benchmark.

## BMRB manifest

Create a manifest that declares the mechanism, reference representation and preregistered thresholds:

```json
{
  "schema_version": 1,
  "study_id": "kumar2024-cross-representation",
  "mechanism_id": "cross_feature_second_moment",
  "participant_key": "subject",
  "policy": {
    "policy_id": "rep-conservation-v1",
    "preregistered": true,
    "reference_representation_id": "raw",
    "min_participants": 3,
    "min_representations": 3,
    "min_representation_families": 2,
    "min_reference_positive_fraction": 0.8,
    "min_all_lane_positive_fraction": 0.8,
    "min_all_lane_ablation_positive_fraction": 0.8,
    "min_direction_match_fraction": 0.8,
    "min_ablation_direction_match_fraction": 0.8,
    "min_information_novel_representation_fraction": 1.0
  },
  "lanes": [
    {"lane_id": "raw", "artifact_dir": "lanes/raw"},
    {"lane_id": "labram", "artifact_dir": "lanes/labram"},
    {"lane_id": "eegpt", "artifact_dir": "lanes/eegpt"}
  ]
}
```

Run:

```bash
quantumbci-bmrb representation bmrb-representation.json \
  --output-dir bmrb-representation
```

Outputs:

```text
bmrb-representation/
  bmrb_representation.json
  report.html
```

The JSON artifact has both a source fingerprint for the declared scientific identity and a complete artifact fingerprint for report integrity.

## Participant-balanced conservation

For each lane, repeated cases and calibration budgets are first averaged within participant. Participants are then weighted equally.

The benchmark reports:

```text
reference_positive_fraction
all_lane_positive_fraction
all_lane_ablation_positive_fraction
direction_match_fraction
ablation_direction_match_fraction
information_novel_representation_fraction
pairwise_reference_correlations
```

Reference correlations are descriptive only. They are unavailable for fewer than three participants or degenerate participant effects and are not used as hidden promotion thresholds.

## BMRB ladder

BMRB-Representation maps onto the same monotonic evidence ladder as BMRB-Dynamics:

```text
0 descriptive
  exact paired representation authority

1 predictive
  candidate beats strongest declared control under held-out authority

2 adversary_surviving
  representation lanes are information-novel relative to matched classical controls

3 source_stability
  candidate and ablation directions are conserved across representation changes

4 repeated_case
  conservation replicates across independent participants and representation families

5 causal_mechanistic
  not established by cross-representation recurrence

6 physical_quantum
  not applicable to this quantum-inspired benchmark
```

A later representation-conservation result cannot pass around an earlier equivalence failure. If density remains covariance-equivalent, the adversary tier fails and later conservation tiers remain visible as `characterized` evidence.

## What to run first

A useful first real study is:

```text
Kumar2024 raw time-by-channel
    vs
LaBraM frozen embeddings
    vs
EEGPT frozen embeddings
```

using the exact same participants, held-out sessions, final evaluation examples and calibration budgets.

Then expand one variable at a time:

```text
Kumar2024
  -> Ma2020
  -> Wang2026
  -> external task family
```

and representation families:

```text
raw
  -> EEGNet / EEG-Conformer learned supervised representations
  -> BENDR
  -> LaBraM
  -> EEGPT
  -> other independently justified frozen encoders
```

The goal is not to collect model names. The goal is to determine which mechanistic quantities survive representation changes, which collapse into classical equivalences, and which remain stable enough to justify a later intervention study.
