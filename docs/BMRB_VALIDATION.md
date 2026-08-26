# BMRB known-ground-truth validation

QuantumBCI v0.19 adds a validation layer for the benchmark itself.

The central question is no longer only whether a candidate neural mechanism survives BMRB. Before a biological result is trusted, we also need evidence that **BMRB reaches the right decision when the answer is known in advance**.

This is a software-and-method validation program, not a simulation of biological truth.

## Why this exists

A benchmark can look rigorous while still having bad operating characteristics. It may reject true mechanisms, admit predictive shortcuts, silently average away reversals, tolerate broken pairing, or report the wrong reason for a failure.

`BMRB_KNOWN_TRUTH_VALIDATION_V1` therefore drives the production confirmatory representation evaluator with declared participant-level data-generating patterns and measures both the final decision and failure localization.

The validation code deliberately calls `evaluate_confirmatory_representation` directly. It does not maintain a second copy of BMRB gate logic.

## ADEMP structure

### Aims

Measure whether BMRB:

- rejects a true effect null;
- rejects an information-equivalent candidate even when it predicts well;
- recovers a shared non-equivalent mechanism;
- rejects a predictive shortcut that has no ablation dependence;
- rejects a representation-specific signature that does not conserve;
- keeps a predeclared primary calibration budget separate from a reversed secondary budget;
- rejects incomplete cross-representation pairing instead of silently pooling it.

### Data-generating mechanisms

The first grid contains six stochastic scenarios plus one structural corruption test.

| Scenario | Known truth | Intended decision |
| --- | --- | --- |
| `effect-null` | candidate adds no predictive effect | fail effect criteria |
| `equivalence-null` | candidate predicts but is declared information-equivalent | fail adversary survival |
| `shared-mechanism-positive` | positive effect and ablation dependence conserve across representations | pass scientific criteria |
| `predictive-shortcut` | predictive effect exists without functional dependence | fail conservation |
| `representation-specific` | effect exists only in the reference representation | fail conservation / scientific criteria |
| `calibration-reversal` | primary budget is positive while a secondary budget reverses sign | pass using only the primary estimand |
| missing-pair corruption | one exact-paired observation is deleted | reject the analysis input |

Participant heterogeneity and measurement noise are generated independently under deterministic seeds. The suite uses two exactly paired representation families and participant-level inference.

### Estimands

For each stochastic scenario the suite records:

- scientific pass rate;
- decision error rate relative to declared truth;
- expected failure-component localization rate;
- reference-lane effect bias;
- bootstrap interval coverage of the declared reference effect.

The current qualification contract gates on decision behavior and failure localization. Interval coverage is reported now so it can become a calibrated performance target once the simulation grid is expanded.

### Methods compared

v0.19 validates BMRB itself. It does not yet claim that BMRB dominates alternative benchmark rules. The next validation expansion should compare BMRB against deliberately weaker baselines such as raw accuracy thresholding, unpaired pooling, and one-number representation similarity rules.

### Performance measures

The installed software contract currently requires:

- effect-null false-positive rate <= 0.10;
- known-positive recovery >= 0.90;
- adversarial decision error <= 0.10;
- expected failure localization >= 0.90;
- calibration-reversal recovery >= 0.90;
- missing exact pairs must be rejected.

These are **software-validation thresholds**, chosen to qualify the deterministic synthetic grid. They are not universal biological significance, power, or promotion thresholds.

## Run it

```bash
quantumbci-bmrb-validate --require-qualified
```

For a faster smoke run:

```bash
quantumbci-bmrb-validate \
  --replicates 4 \
  --bootstrap-resamples 100 \
  --output bmrb-validation.json \
  --require-qualified
```

The JSON output contains scenario summaries and every replicate-level decision so downstream analyses can inspect operating characteristics rather than relying on one aggregate score.

## Claim boundary

A successful known-truth validation supports the statement that the implemented BMRB decision machinery behaves correctly on the declared synthetic adversary grid.

It does **not** establish that:

- the synthetic data are biologically realistic;
- a candidate mechanism is true in EEG;
- a representation is causally necessary in the brain;
- a quantum-inspired parameterization implies a physical quantum substrate;
- the current validation grid is exhaustive.

The validation suite is an adversary for the benchmark, not a certificate of biological truth.

## Next expansion

The highest-value additions after v0.19 are:

1. explicit high-heterogeneity and missing/noisy-session operating curves;
2. effect-size and sample-size sweeps for empirical power/type-I surfaces;
3. naive benchmark baselines for comparative calibration;
4. non-invertible and invertible representation transforms with known mechanism conservation;
5. dataset-level hierarchical simulation for multi-study replication;
6. multiplicity scenarios spanning many candidate mechanisms, layers, and metrics;
7. causal/interventional known-truth simulations for the BMRB causal ladder.

Those additions turn validation from a contract test into a genuine methods-study simulation program.
