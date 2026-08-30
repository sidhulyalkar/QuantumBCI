# BMRB adaptive candidate-search authority

A frozen candidate list does not eliminate researcher degrees of freedom if the rule for deciding **what to inspect next** can change after seeing the current result.

`quantumbci.bmrb_adaptive_search` freezes that inspection policy. `quantumbci.bmrb_adaptive_search_stress` attacks it under the same near-boundary known-null DGM used by the multiplicity work.

## Scientific separation

Adaptive search is treated as a discovery/inspection process, not as a new promotion rule.

The v1 contract keeps three things distinct:

1. **candidate authority**: one complete candidate family and one promotion-authoritative primary are frozen by `BMRBMultiplicityPlan`;
2. **inspection authority**: the adaptive search start point, routing metric, routing cutoff, route strides, collision rule, maximum evaluations, and stopping rule are frozen by `BMRBAdaptiveSearchPlan`;
3. **promotion authority**: complete candidate evidence is still passed to the multiplicity layer, so an adaptive stop cannot erase failures, omit uninspected candidates, or transfer promotion authority to the first interesting survivor.

This means a search can be adaptive without the confirmatory record becoming adaptive.

## v1 outcome-routed protocol

The search always begins at the multiplicity plan's predeclared primary candidate.

After a failed candidate, the next preferred candidate depends on its observed reference effect:

```text
reference_observed_effect >= routing_effect_cutoff
    -> above_cutoff_stride
else
    -> below_cutoff_stride
```

If that preferred location has already been inspected, the collision rule selects the first unvisited candidate encountered circularly from the preferred index.

Search stops at the first scientific survivor or at `max_evaluations`, whichever comes first.

All of these choices are fingerprinted. Changing the cutoff, route strides, budget, candidate family, or stopping semantics changes the adaptive-plan fingerprint.

## Why require distinct route strides?

If the above- and below-cutoff strides are identical, the route does not actually depend on evidence. v1 rejects that configuration rather than labelling a fixed sequence as adaptive.

## Complete evidence remains mandatory

`run_adaptive_search(...)` requires evidence for **every candidate in the frozen multiplicity family**, including candidates the simulated adaptive analyst would never reach before stopping.

This is intentional. The transcript answers:

> What would an outcome-dependent analyst inspect, and when would they stop?

The multiplicity layer separately answers:

> Given the complete frozen family, is the predeclared primary scientifically promotion-eligible?

The adaptive transcript cannot substitute its inspected subset for the confirmatory evidence set.

## Known-null stress

`BMRB_ADAPTIVE_CANDIDATE_SEARCH_STRESS_V1` generates a complete 20-candidate universe through production `run_validation_replicate(...)` with:

```text
BMRB effect threshold:       0.050
true reference effect:       0.049
true alternate-lane effect:  0.049
```

The default adaptive plan uses:

```text
start:                frozen primary
routing cutoff:       0.050
above-cutoff stride:  1
below-cutoff stride:  3
maximum evaluations:  20
stop:                 first scientific survivor
```

Because the maximum budget covers the complete universe and the collision rule never revisits a candidate, adaptive `any survivor` must equal exhaustive `any survivor`. The difference is the trajectory and stopping time, not which latent candidate universe exists.

The key contrast remains:

```text
naive adaptive survivor
vs.
authorized primary promotion
```

A non-primary survivor may end the naive search early, but it does not inherit promotion authority.

## What v1 does not solve

This is deliberately narrower than a general sequential-testing framework. It does not yet provide:

- alpha-spending or always-valid p-values;
- optional-stopping-valid confidence sequences;
- reinforcement-learning or Bayesian optimization search policies;
- adaptive creation of new candidate hypotheses outside the frozen universe;
- multi-family hierarchical search;
- cross-dataset sequential replication authority.

Those should receive explicit method IDs, known-truth simulation, and claim-boundary tests rather than being smuggled into v1.

## Claim boundary

This adaptive-search layer does not validate biological truth. It validates reporting and authority semantics under a declared synthetic search process. It does not establish neural causal necessity, execute the sealed final-evaluation partition, or authorize a physical-quantum interpretation.
