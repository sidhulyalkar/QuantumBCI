# BMRB gate-level operating diagnostics

The ordinary BMRB operating-characteristics artifact answers a necessary but incomplete
question: **did the benchmark reach the intended overall scientific decision?**

`BMRB_KNOWN_TRUTH_GATE_DIAGNOSTICS_V1` is a sibling methods artifact that asks the next
question: **which scientific gate produced that decision, and was that gate behaving correctly
under declared known truth?**

The diagnostic artifact uses the exact same `BMRBOperatingStudyPolicy`, frozen grid, seed
partition, participant counts, calibration authority, and production `run_validation_replicate`
path as the operating-curves study. It does not alter the existing operating artifact schema.

## Canonical gate order

The v1 diagnostic path records four confirmatory gates in their scientific decision order:

1. `effect`
2. `adversary`
3. `conservation`
4. `coverage`

For current v1 known-truth DGMs, a negative scenario declares one intended failing component.
That gate is treated as expected FAIL and the remaining gates are expected PASS. Known-positive
scenarios expect all four gates to pass.

This is an explicit modeling assumption, not an inferred truth label. If future DGMs are
intended to fail multiple gates, the diagnostic schema should be revised rather than silently
pretending they are single-failure cases.

## Per-cell evidence

For every frozen operating-grid cell, the artifact records:

- overall scientific pass count/rate;
- expected pass/fail truth for each gate;
- observed gate pass count/rate;
- Wilson interval for each gate pass rate;
- gate confusion counts (`TP`, `TN`, `FP`, `FN`);
- first-failing-gate counts across Monte Carlo replicates;
- first-failure localization rate against the declared DGM;
- the exact cell parameters and frozen base seed.

A scientific PASS must equal the number of replicates with **no** failing gate. The verifier
reconstructs and enforces that invariant.

## Aggregate gate metrics

Across the complete frozen grid, each gate reports:

- expected-PASS support;
- expected-FAIL support;
- pass sensitivity;
- failure specificity;
- false-pass rate;
- false-fail rate.

A rate is `null` when its truth class has no support. This matters today for `coverage`: the
current default DGM family contains coverage-positive cases but no dedicated coverage-negative
scenario. Therefore coverage pass sensitivity can be estimated, while coverage specificity and
false-pass rate must remain unestimated.

**Missing truth support is not evidence of perfect performance.**

A future coverage-negative DGM should deliberately violate the declared participant,
representation, or representation-family coverage requirement while preserving valid pairing
and the other scientific gates as far as possible.

## Decision-path decompositions

Two aggregate decompositions make global error rates more interpretable:

### False-promotion escape path

For an overall-null scenario that nevertheless passes, the artifact attributes the false
promotion to the DGM's **expected failing gate**. This answers which intended safeguard was
escaped.

### Known-positive loss path

For a known-positive scenario that fails, the artifact attributes the loss to the **first
observed failing gate**. This exposes which gate is becoming over-stringent as sample size,
noise, or heterogeneity changes.

These are diagnostic decompositions, not causal explanations of why a statistical error
occurred.

## Software-invalid evidence is outside the confusion matrix

Exact-pairing violations and other software-invalid evidence are not scientific gate failures.
They are excluded from gate confusion entirely and validated through separate fail-closed
contracts such as the structured missing-pair stress test.

The artifact therefore fixes:

```text
software_invalid_trials_in_gate_confusion = 0
```

and states the exclusion policy explicitly. An invalid evidence bundle must not inflate gate
specificity by being counted as a correctly rejected scientific null.

## Artifact integrity

Gate diagnostics have their own schema and fingerprint rather than extending the stable v1
operating-curves artifact. The read-side verifier:

- recomputes the root fingerprint;
- reconstructs the complete operating policy, grid, and seed partition;
- verifies policy/grid/seed fingerprints;
- reconstructs Cartesian cell order and base seeds;
- reconstructs expected gate truth from the declared DGM;
- recomputes per-cell confusion accounting and Wilson intervals;
- verifies first-failing-gate partitions and localization;
- reconstructs aggregate gate confusion and error-path decompositions;
- requires unsupported rates to remain `null`.

Fingerprints are integrity checks, not signatures. External provenance and preregistration still
matter for confirmatory use.

## Relationship to Stage B

This diagnostic layer should be run on the **development** operating authority before final
acceptance criteria are frozen. It is intended to guide the next DGM expansion over:

- responder fraction;
- session-count imbalance severity;
- participant count;
- measurement noise;
- heterogeneity;
- near-boundary effects;
- informative or structured missingness;
- dedicated coverage-negative conditions;
- stronger reduced comparators.

Only after those development diagnostics are understood should final numeric acceptance bounds
be justified, externally preregistered, and bound to the already-separated evaluation seed
authority.

## Claim boundary

Gate diagnostics validate decision-path behavior under declared synthetic DGMs. They do **not**
validate biological realism, establish a neural causal mechanism, or authorize a
physical-quantum interpretation.
