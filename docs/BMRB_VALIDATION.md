# BMRB known-ground-truth validation

QuantumBCI v0.19 adds a validation layer for the benchmark itself.

The central question is no longer only whether a candidate neural mechanism survives BMRB. Before a biological result is trusted, we also need evidence that **BMRB reaches the right decision when the answer is known in advance**.

This is a software-and-method validation program, not a simulation of biological truth.

## Why this exists

A benchmark can look rigorous while still have bad operating characteristics. It may reject true mechanisms, admit predictive shortcuts, silently average away reversals, tolerate broken pairing, or report the wrong reason for a failure.

`BMRB_KNOWN_TRUTH_VALIDATION_V1` therefore drives the production confirmatory representation evaluator with declared participant-level data-generating patterns and measures both the final decision and failure localization.

The validation code deliberately calls `evaluate_confirmatory_representation` directly. It does not maintain a second copy of BMRB gate logic.

v0.19 also includes `BMRB_KNOWN_TRUTH_STRESS_V1`, an extended stress layer that compares BMRB with deliberately weak decision rules. Those rules are diagnostic negative controls, not proposed scientific baselines.

## ADEMP structure

### Aims

Measure whether BMRB:

- rejects a true effect null;
- rejects an information-equivalent candidate even when it predicts well;
- recovers a shared non-equivalent mechanism;
- rejects a predictive shortcut that has no ablation dependence;
- rejects a representation-specific signature that does not conserve;
- rejects scientifically insufficient representation-family coverage without misclassifying valid evidence as software-invalid;
- keeps a predeclared primary calibration budget separate from a reversed secondary budget;
- conserves a declared mechanism across an invertible coordinate change;
- remains well behaved under participant heterogeneity;
- aggregates repeated noisy sessions at the participant level;
- rejects incomplete cross-representation pairing instead of silently pooling it.

### Core data-generating mechanisms

The compact qualification grid contains seven stochastic scenarios plus one structural corruption test.

| Scenario | Known truth | Intended decision |
| --- | --- | --- |
| `effect-null` | candidate adds no predictive effect | fail effect criteria |
| `equivalence-null` | candidate predicts but is declared information-equivalent | fail adversary survival |
| `shared-mechanism-positive` | positive effect and ablation dependence conserve across representations | pass scientific criteria |
| `predictive-shortcut` | predictive effect exists without functional dependence | fail conservation |
| `representation-specific` | effect exists only in the reference representation | fail conservation / scientific criteria |
| `coverage-family-deficit` | two valid exactly paired representation families are supplied under a policy requiring three | fail coverage only |
| `calibration-reversal` | primary budget is positive while a secondary budget reverses sign | pass using only the primary estimand |
| missing-pair corruption | one exact-paired observation is deleted | reject the analysis input |

The coverage-negative DGM is intentionally different from the missing-pair corruption. Its evidence is structurally valid and exactly paired. Effect, adversary-survival, and conservation evidence are constructed to pass. The policy simply requires one more independent representation family than the study supplies, so `coverage` should be the sole scientific failure.

Participant heterogeneity and measurement noise are generated independently under deterministic seeds. The suite uses exactly paired representation families and participant-level inference.

### Extended stress mechanisms

The `--extended` surface adds six harder attacks:

| Scenario | Stress question |
| --- | --- |
| `equivalence-null-naive-trap` | does an effect-only rule falsely accept a candidate that BMRB rejects as information-equivalent? |
| `predictive-shortcut-naive-trap` | does an effect-only rule falsely accept predictive value without functional dependence? |
| `calibration-reversal-naive-trap` | does averaging primary and secondary budgets erase a valid predeclared primary effect? |
| `invertible-coordinate-positive` | can the declared effect and ablation dependence survive a coordinate-family change? |
| `heterogeneous-shared-positive` | does participant heterogeneity preserve the correct positive decision when the mechanism is shared? |
| `noisy-repeated-sessions-positive` | are three noisy sessions per participant summarized without converting sessions into independent participants? |

The repeated-session case intentionally produces more case rows while requiring the evaluator's inference-unit count to remain the number of participants.

### Estimands

For each core stochastic scenario the suite records:

- scientific pass rate;
- decision error rate relative to declared truth;
- expected failure-component localization rate;
- reference-lane effect bias;
- bootstrap interval coverage of the declared reference effect.

The extended suite additionally records:

- BMRB scientific pass rate;
- pass rate of a naive primary-effect-only rule;
- pass rate of a naive rule that averages effects across calibration budgets.

The current qualification contract gates on decision behavior and failure localization. Interval coverage is reported now so it can become a calibrated performance target once the simulation grid is expanded.

### Methods compared

The production BMRB confirmatory evaluator is the scientific method under validation.

Two deliberately weak rules serve as negative controls:

1. **Primary-effect-only:** accept when mean candidate advantage exceeds the threshold, ignoring equivalence/adversary and functional-dependence gates.
2. **Budget-averaged effect:** average candidate advantage across every calibration budget before thresholding, violating the predeclared-primary-estimand contract.

The intended outcome is not merely that BMRB passes its own fixtures. The weak rules must visibly fail on cases specifically constructed to exploit their missing safeguards. v0.19 does not claim these two toy rules are the strongest competing scientific frameworks.

### Performance measures

The core software contract currently requires:

- effect-null false-positive rate <= 0.10;
- known-positive recovery >= 0.90;
- adversarial decision error <= 0.10, now including the valid coverage-family deficit;
- expected failure localization >= 0.90;
- calibration-reversal recovery >= 0.90;
- missing exact pairs must be rejected.

The extended contract additionally requires:

- BMRB decision error <= 0.10 across every stress scenario;
- the naive effect-only rule falsely accepts the equivalence and shortcut traps at least 90% of the time;
- the naive budget-averaged rule falsely rejects the calibration-reversal trap at least 90% of the time;
- BMRB recovers invertible-coordinate, heterogeneous, and repeated-session positives at least 90% of the time.

These are **software-validation thresholds**, chosen to qualify the deterministic synthetic grid. They are not universal biological significance, power, or promotion thresholds.

## Run it

Core validation:

```bash
quantumbci-bmrb-validate --require-qualified
```

Extended validation:

```bash
quantumbci-bmrb-validate \
  --extended \
  --replicates 4 \
  --participants 8 \
  --bootstrap-resamples 100 \
  --output bmrb-validation.json \
  --require-qualified
```

The JSON output contains scenario summaries and every replicate-level decision so downstream analyses can inspect operating characteristics rather than relying on one aggregate score. With `--extended`, the root artifact also contains a `stress_suite` object.

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

With all four confirmatory gates now receiving both expected-PASS and expected-FAIL support in the development diagnostics, the next highest-value additions are:

1. stronger comparative methods beyond the two deliberately weak negative controls;
2. non-invertible representation transforms and nuisance-specificity simulations;
3. multiplicity scenarios spanning candidate mechanisms, layers, tasks, and metrics;
4. dataset-level hierarchical simulation for multi-study replication;
5. causal/interventional known-truth simulations for the BMRB causal ladder;
6. publication-quality operating-curve and gate-diagnostic figures from the frozen development authority.

Those additions should remain sequential, and the already separated final-evaluation seed partition should stay unopened until the complete acceptance plan is scientifically justified and preregistered.
