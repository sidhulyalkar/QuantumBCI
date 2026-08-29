# BMRB operating-characteristics studies

QuantumBCI v0.19 established a compact known-truth qualification suite. That suite answers a software question: **does the implemented BMRB decision machinery behave correctly on a small declared adversary set?**

The operating-characteristics layer asks a broader methods question:

> Across predeclared sample-size, effect, heterogeneity, and noise regimes, how often does BMRB recover known positives, reject known nulls, localize the correct failing gate, and provide calibrated participant-level uncertainty?

This remains synthetic benchmark validation. It is not evidence that a synthetic data-generating mechanism is biologically realistic.

## Why this is a separate layer

A deterministic CI grid and a publication-grade simulation study should not quietly become the same object.

The CI suite is intentionally small and fast enough to qualify software changes. A methods study needs many independent Monte Carlo replications, parameter curves, uncertainty around operating rates, and a frozen evaluation authority that is not repeatedly inspected during development.

`quantumbci.bmrb_validation_operating` therefore calls the existing production `run_validation_replicate(...)` function rather than implementing a second decision pipeline.

## Frozen simulation authority

A `BMRBOperatingStudyPolicy` binds:

- study ID;
- exact QuantumBCI source revision;
- `development` or `evaluation` partition;
- complete parameter grid and its fingerprint;
- Monte Carlo replicates per cell;
- participant-bootstrap resamples;
- primary calibration budget;
- deterministic seed-partition contract and fingerprint.

Changing any of these decisions changes the policy fingerprint.

### Development and final evaluation are disjoint

`SimulationSeedPartition` reserves separate arithmetic seed regions for development and final evaluation. Within a study, each grid cell receives a deterministic base seed and each Monte Carlo replicate receives a deterministic offset.

The policy verifies that the largest development seed permitted by the frozen grid remains below the first evaluation seed. This is simulation authority, not cryptographic secrecy. Its purpose is to make development/evaluation reuse detectable and reviewable.

A researcher should use the development partition to debug DGMs, choose a scientifically justified grid, and preregister acceptable operating behavior. The final evaluation partition should then be run after those decisions are frozen.

## Parameter grid

`OperatingCurveGrid` currently varies:

- known-truth scenario;
- independent participant count;
- candidate/ablation effect scale;
- between-participant heterogeneity scale;
- measurement-noise scale.

The recommended development grid is deliberately substantial:

```text
participants:          4, 8, 16, 32
effect scale:          0.50, 0.75, 1.00, 1.25
heterogeneity scale:   0.50, 1.00, 2.00
measurement noise:     0.50, 1.00, 2.00
scenarios:             7 registered core known-truth DGMs
```

The seventh DGM, `coverage-family-deficit`, preserves valid exact pairing and strong evidence on the preceding gates while requiring three representation families and supplying two. That gives the coverage gate explicit expected-FAIL support without conflating scientific insufficiency with malformed evidence.

The recommended development design is therefore **1008 cells** before Monte Carlo replication. It is a development recommendation, not a universal final-study grid or biological power calculation.

The exact final grid should be committed or externally registered before its final evaluation partition is executed.

## Per-cell operating evidence

Every cell reports:

- expected scientific outcome and expected failure component;
- participant count and DGM scaling parameters;
- frozen base seed;
- number of Monte Carlo replications;
- observed BMRB pass count and pass rate;
- decision-error rate relative to declared truth;
- Monte Carlo standard error of the pass rate;
- 95% Wilson interval for the pass probability;
- expected-failure localization rate;
- reference-effect mean bias and RMSE;
- participant-bootstrap interval coverage.

Simulation replication is distinct from participant bootstrap resampling. Increasing participant bootstrap resamples does not increase the number of independent Monte Carlo simulation replicates.

## Aggregate evidence

The aggregate artifact reports descriptive operating summaries such as:

- false-promotion rate across declared null cells;
- known-positive recovery rate;
- mean cell decision-error rate;
- mean failure-localization rate;
- mean reference-effect interval coverage;
- the worst observed grid cell by decision error.

The v1 operating artifact deliberately serializes:

```text
qualification_defined = false
```

No universal acceptable false-promotion, power, or coverage threshold is introduced by the library. A confirmatory methods study must define and justify its own acceptance criteria **before** final evaluation.

## Example

```python
from quantumbci.bmrb_validation_operating import (
    BMRBOperatingStudyPolicy,
    recommended_development_grid,
    run_bmrb_operating_characteristics,
    write_bmrb_operating_characteristics,
)

policy = BMRBOperatingStudyPolicy(
    study_id="bmrb-operating-development-v1",
    source_sha="<exact-quantumbci-commit>",
    partition="development",
    grid=recommended_development_grid(),
    replicates_per_cell=200,
    bootstrap_resamples=300,
)

result = run_bmrb_operating_characteristics(policy)
write_bmrb_operating_characteristics(result, "bmrb-operating-development.json")
```

Do not use the 1008-cell example casually in ordinary CI. `qualification_smoke_grid()` exists for small deterministic software tests.

## What is still missing for a full methods paper

All four confirmatory gates now have dedicated expected-PASS and expected-FAIL truth support in the registered development DGM family. The next high-value expansions are therefore different kinds of stress rather than another basic gate fixture:

- near-boundary effect grids chosen from a frozen design rationale rather than post hoc inspection;
- more Monte Carlo replications with explicit precision targets;
- stronger reduced comparator methods under matched information sets;
- multiplicity strategies across many mechanisms or evidence axes;
- study/dataset-level hierarchical replication;
- causal/interventional known-truth DGMs;
- publication-quality operating-curve and gate-diagnostic figures and tables.

Those should be added sequentially so a change in the data-generating mechanism remains distinguishable from a change in the BMRB decision machinery.

## Claim boundary

Good operating characteristics mean the implemented benchmark behaves as intended under the declared synthetic worlds. They do **not** establish that those worlds resemble neural biology, that a real EEG mechanism is causal, or that any physical quantum substrate exists.
