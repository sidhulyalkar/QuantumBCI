# BMRB correlated candidate-search stress

Candidate count is not the same thing as independent search opportunity. Closely related layers, hyperparameters, preprocessing pipelines, or representations can be highly redundant while still creating many labels that a researcher might inspect.

`quantumbci.bmrb_multiplicity_correlated` makes that distinction explicit.

## Block-correlation model

The stress keeps the searched family at a fixed number of candidate labels while varying how many genuinely independent latent evidence draws generated those labels.

For the default 20-candidate family, the development surface uses:

```text
candidate labels:       20  20  20  20
independent draws:       1   2   5  20
```

Candidates assigned to the same latent draw reuse the exact production-validation seed. They are therefore perfect redundant variants under the synthetic benchmark. The first `k` candidate labels introduce draws `0..k-1`; remaining labels cycle over those existing draws.

This construction is deliberately discrete and auditable. It avoids claiming that one arbitrary continuous correlation coefficient captures the geometry of real model-search spaces.

## Scientific DGM

Every independent draw is passed through production `run_validation_replicate(...)` using the same near-boundary known null as the original winner-picking stress:

```text
BMRB effect threshold:       0.050
true reference effect:       0.049
true alternate-lane effect:  0.049
```

The scientific truth therefore remains on the null side of the frozen validation boundary. No BMRB threshold, p-value rule, or gate is changed.

## Exact invariants

The nested block design gives several exact checks rather than approximate expectations:

1. **One effective draw equals the primary result.** If all 20 labels are copies of one evidence draw, `any survivor` and the predeclared primary must be identical.
2. **Primary invariance.** Draw zero is always the primary candidate's draw, with the same seed under every correlation condition. Expanding the effective search space cannot alter the primary result.
3. **Nested naive survival.** Independent-draw sets are nested. Once a replicate contains a survivor at `k` effective draws, adding more independent draws cannot make that same replicate lose its naive `any survivor` status.
4. **No authority transfer.** Non-primary survivors remain reportable but cannot acquire promotion authority after inspection.

The important quantity is therefore not merely candidate count. It is the number and dependence structure of genuinely distinct search opportunities.

## Why this matters

A search over 20 nearly identical layers may have much less winner-picking inflation than a search over 20 genuinely different mechanisms, yet both can be described informally as “we tried 20 models.” The block-correlated surface forces that ambiguity into the benchmark artifact.

This is especially relevant to neural-mechanism work where candidates often arise from:

- adjacent network layers;
- neighboring checkpoints;
- minor preprocessing variants;
- closely related representation transforms;
- hyperparameter sweeps;
- multiple task definitions derived from the same participants.

The next refinement should attack adaptive search, where the later candidates themselves depend on earlier observed results.

## Authority boundary

This stress remains development simulation authority. It does not execute the frozen final-evaluation partition, validate biological truth, establish neural causal necessity, or authorize a physical-quantum interpretation.
