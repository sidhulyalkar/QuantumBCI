# E002 trajectory evidence authority

v0.8 introduces the evidence boundary that must exist before QuantumBCI fits any real neural
dynamics model.

A normal train/test split is not sufficient for a continuous trajectory study. Two model lanes can
use the same rows and still see different evidence if they disagree about temporal adjacency,
window overlap, representation fitting, missing windows, or which transitions cross a deployment
boundary.

QuantumBCI therefore separates two authorities:

```text
neurOS / upstream neural authority
  exact processed neural samples
  source / calibration / final-evaluation membership
  processed-data fingerprint
                 |
                 v
QuantumBCI TrajectoryEvidenceAuthority
  exact state-tensor bytes
  trajectory identity
  window start + stop times
  fixed window duration + stride
  representation-fit subset
  temporal purge gaps
  legal within-role transition graph
                 |
                 v
all E002 model lanes
```

The trajectory authority does not replace neurOS. It can bind the upstream neurOS authority
fingerprint while adding the temporal semantics a dynamics benchmark needs.

## Why this matters

Several subtle leaks are easy to introduce in continuous neural data:

- overlapping windows can land on opposite sides of a train/evaluation boundary;
- adjacent windows can share most of their raw samples even when their row indices differ;
- a PCA, encoder, normalization, or latent-dimension selector can be refit using target/evaluation
  examples;
- one model can silently bridge a gap that another model treats as a sequence break;
- a model can train on a transition whose left endpoint is calibration and right endpoint is final
  evaluation;
- two methods can independently regenerate numerically different latent tensors while claiming to
  use the same representation.

`TrajectoryEvidenceAuthority` turns these choices into frozen evidence rather than implementation
conventions.

## v1 scope

The first version deliberately supports only:

```text
fixed window duration
fixed start-time stride
missing data policy = reject
```

Irregular trajectories are not approximated automatically. They need an explicit contract for
maximum legal gaps, numerical integration timing, missingness, and interpolation. Until that lands,
`time_step_policy` must be `fixed`.

A gap larger than the declared stride is allowed and creates a trajectory block boundary. A
start-time delta smaller than the declared stride fails closed because it reveals a denser temporal
lattice than the authority claims.

## Portable descriptor

A trajectory study can be handed to QuantumBCI with one JSON descriptor and NumPy arrays. Example:

```json
{
  "schema_version": 1,
  "dataset_id": "my-eeg-dataset",
  "case_id": "subject-07/recording-03",
  "latent_dimension": 3,
  "time_step_policy": "fixed",
  "expected_window_seconds": 2.0,
  "expected_step_seconds": 0.5,
  "step_tolerance_seconds": 1e-6,
  "purge_seconds": 2.0,
  "upstream_authority_fingerprint": "neuros-authority-fingerprint",
  "source_revisions": {
    "quantumbci": "<GIT_SHA>",
    "neuros": "<GIT_SHA>",
    "encoder": "<MODEL_OR_CODE_REVISION>"
  },
  "data": {
    "states": "states.npy",
    "trajectory_ids": "trajectory_ids.npy",
    "start_times_s": "start_times_s.npy",
    "stop_times_s": "stop_times_s.npy",
    "valid_mask": "valid_mask.npy"
  },
  "split": {
    "fit_indices": [0, 1, 2, 3],
    "calibration_indices": [5, 6],
    "evaluation_indices": [10, 11, 12],
    "representation_fit_indices": [0, 1, 2]
  }
}
```

The public JSON Schema is packaged at:

```text
quantumbci/schemas/trajectory-contract-v1.schema.json
```

File paths are resolved relative to the descriptor. They are **not** part of scientific identity.
Renaming `states.npy` while preserving byte-identical contents and the same authority semantics does
not change the authority fingerprint.

## Data identity

`TrajectoryEvidenceData.data_sha256` binds:

- exact numeric state tensor bytes and dtype/shape;
- trajectory IDs in row order;
- start times;
- stop times;
- valid-window mask;
- declared data metadata.

Changing one state value, timestamp, trajectory ID, validity flag, or metadata field changes the
content fingerprint.

The state tensor must be two-dimensional:

```text
windows × state_features
```

For the first qubit-state E002 lane the expected state dimension is three Bloch coordinates, but the
authority itself is generic over feature dimension.

## Evidence roles

The authority freezes four index sets:

- `fit_indices`: source/history windows available for dynamical fitting;
- `calibration_indices`: target adaptation windows, if any;
- `evaluation_indices`: immutable final evidence windows;
- `representation_fit_indices`: the only windows allowed to fit PCA, an encoder adapter, scaling,
  latent dimensionality, or another learned representation transformation.

`representation_fit_indices` must be a subset of `fit_indices`.

The three evidence roles are mutually disjoint. Their index ordering is canonicalized before
fingerprinting, so `[3, 1, 2]` and `[1, 2, 3]` describe the same evidence set.

## Purged temporal boundaries

Index disjointness is not enough when windows overlap in time.

For windows from different evidence roles that share a trajectory ID, QuantumBCI computes their
edge-to-edge temporal separation. A negative separation means the windows overlap. Every pair of
roles must satisfy:

```text
minimum temporal separation >= purge_seconds
```

This catches a classic sliding-window leak where the fit window ending at 10 seconds and the final
window starting at 9 seconds share raw data despite having different row IDs.

## Legal transition graph

Models do not receive arbitrary adjacent rows. The authority constructs legal transition pairs
inside each role.

For fixed-step v1, a pair `(i, j)` is legal only when:

- both windows have the same trajectory ID;
- both belong to the same evidence role;
- `start_j - start_i` matches `expected_step_seconds` within tolerance.

Larger gaps break the graph. Transitions across fit/calibration/evaluation boundaries are never
exposed.

The authority must contain at least one legal fit transition and one legal final-evaluation
transition.

## Executable E002 stage

After the synthetic identifiability gate passes, materialize the temporal authority:

```bash
python -m quantumbci.experiments.tasks \
  trajectory-contract E002 \
  --input trajectory_contract.json \
  --output trajectory_index.json
```

The output records:

- data SHA-256;
- authority fingerprint;
- all frozen evidence indices;
- representation-fit subset;
- window/step/purge policy;
- upstream authority binding;
- source revisions;
- legal transition counts for fit/calibration/evaluation;
- a declaration that every later model lane must use this same tensor and authority.

This is an **authority artifact**, not a model result. Passing it gives permission to fit models. It
does not provide predictive evidence and cannot promote a physical quantum claim.

## Same-tensor rule

The eventual E002 model ladder must consume the exact same:

```text
trajectory authority fingerprint
state tensor SHA-256
legal transition graph
```

for:

- constrained Lindblad-family fitting;
- the exact affine/Bloch representation;
- conventional LDS/Kalman;
- VAR or damped-oscillator controls;
- switching-state controls;
- nonlinear controls when justified.

A lane may not regenerate a look-alike latent tensor and call it matched evidence.

## Fail-closed examples

The authority rejects:

- duplicate indices;
- overlapping evidence roles;
- representation fitting on calibration/evaluation windows;
- invalid or missing selected windows;
- duplicate temporal coordinates;
- wrong window duration;
- time gaps smaller than the declared fixed stride;
- insufficient purge gaps;
- tensor/timestamp mutation after authority creation;
- unsupported irregular timing;
- cases with no legal fit or final-evaluation transitions.

These failures are evidence-contract errors, not numerical inconveniences.

## Next implementation boundary

With temporal authority in place, the next E002 implementation should be the **matched dynamics
model-fit API**. It should accept a validated trajectory authority rather than raw arrays and should
produce model artifacts that explicitly bind:

```text
authority_fingerprint
state_tensor_sha256
fit_transition_ids
evaluation_transition_ids
model/config/source revision
```

The first useful comparison should stay small:

1. unconstrained affine dynamics;
2. gauge-fixed canonical Lindblad-family projection/fit;
3. conventional regularized linear dynamics / Kalman-style state model;
4. damped oscillator where appropriate.

Switching and nonlinear models can follow after the matched evidence and scoring surface is proven.
