# BMRB study-level operating characteristics

`BMRB_STUDY_KNOWN_TRUTH_OPERATING_CURVES_V1` validates the higher-level replication and
sensitivity machinery under declared cross-study truth. It is a software-validation
program, not evidence that a neural mechanism is biologically real.

## Why another operating layer exists

Participant-level BMRB validation answers whether one study's confirmatory machinery
rejects known nulls, recovers known positives, and localizes adversarial failures. It does
not answer whether several finished studies are combined correctly.

The cross-study layer has different failure modes:

- hundreds of participants in one dataset can masquerade as hundreds of replications;
- a failed primary can be quietly replaced by successful later studies;
- a just-sufficient broad PASS can be narrated as redundant support;
- a directional context reversal can be hidden by a broad pass count;
- positive replication margin can be mistaken for low heterogeneity;
- duplicate truth labels can silently overweight one condition in aggregate metrics;
- nominally independent studies can accidentally share simulation randomness.

The study-level operating program attacks those failures directly.

## Hierarchical execution path

Every simulated outer replicate follows the production hierarchy:

1. A declared truth label selects a participant-level BMRB DGM.
2. `run_validation_replicate(...)` generates and evaluates one independent study.
3. Its bounded result becomes exactly one `BMRBStudyEvidence` object.
4. The frozen family is evaluated with `evaluate_study_replication(...)`.
5. The completed replication decision is inspected by `assess_study_sensitivity(...)`.
6. Only study-level decisions and diagnostics enter the outer operating summary.

Participant rows are never pooled across studies. Participant count affects precision
inside each study, but it never becomes the number of replication votes.

## Declared cross-study truths

v1 freezes eight **distinct** software scenarios. Their replication minima are
scenario-specific test fixtures, not universal biological thresholds.

| Scenario | Study truths | Intended interpretation |
| --- | --- | --- |
| `homogeneous-positive-3` | positive, positive, positive | broad positive with zero replication margin, therefore a fragility warning despite directional agreement |
| `homogeneous-null-3` | null, null, null | broad null without a heterogeneity warning |
| `homogeneous-positive-4` | positive ×4 | redundant broad positive without a warning |
| `homogeneous-null-4` | null ×4 | broad null without a heterogeneity warning |
| `primary-only-positive-4` | positive, null, null, null | preserve context-specific primary evidence, reject broad claim |
| `primary-fail-replications-positive-4` | null, positive, positive, positive | later studies cannot replace the frozen primary |
| `fragile-one-conflict-4` | positive, positive, positive, reversal | broad PASS with zero margin and directional conflict |
| `redundant-one-conflict-5` | positive, positive, positive, positive, reversal | broad PASS with positive replication margin that still warrants a heterogeneity warning |

The final two scenarios deliberately separate **replication-count fragility** from
**cross-study heterogeneity**. A warning can arise because one successful replication is
indispensable, because study effects conflict, or both. Positive redundancy margin alone
is therefore not treated as evidence of homogeneous support.

The `reversal` truth is a declared adverse context whose reference effect reverses sign and
fails the participant-level effect criterion. It is not interpreted as proof that the
mechanism is universally false.

## Seed authority

`StudySimulationSeedPartition` assigns a unique deterministic seed to every
`(partition, cell, outer replicate, study)` tuple. Separate strides are capacity checked so
study seeds cannot overlap replicate seeds and replicate seeds cannot overlap cell seeds.
The authority also freezes a maximum cell capacity and verifies that the **entire**
development and evaluation seed spaces are disjoint across that capacity, rather than
checking only their base offsets.

The **evaluation partition is not executable in v1**. It is represented in policy content
so a future adjudicated release can bind it, but the public operating runner fails closed
if asked to execute it. CI and development use only the development partition.

## Grid axes

The recommended development grid varies:

- eight declared cross-study scenarios;
- participants per study: 4, 8, 16;
- within-study participant heterogeneity scale: 0.5, 1, 2;
- measurement-noise scale: 0.5, 1, 2;
- cross-study positive-effect variation scale: 0, 1, 2.

That is **648 cells**. Study count is encoded by the declared scenario family in v1 rather
than by dynamically changing the replication rule after looking at results.

CI uses a separate deterministic eight-cell smoke grid with zero synthetic noise and one
outer replicate. CI therefore tests software contracts without consuming the development
operating program as if it were a scientific result.

## Reported estimands

Each cell reports:

- broad replication pass rate and Wilson interval;
- decision-error rate relative to declared cross-study truth;
- context-specific-evidence semantics match rate;
- sensitivity-warning match rate;
- primary-role protection rate;
- fragile-claim detection rate;
- mean successful-replication margin;
- mean study-level effect range.

The aggregate result separates false promotion on declared broad negatives from recovery
on declared broad positives. Each declared scenario contributes one condition family;
duplicated aliases are not used to reweight aggregate behavior. `qualification_defined`
remains **false**. v1 does not invent a pass/fail acceptance threshold after seeing
development curves.

## Sensitivity is still not a promotion gate

The operating program exercises the merged study-sensitivity layer, but it does not change
its authority. A fragile broad replication PASS remains the same replication PASS while
carrying a sensitivity warning. A future promotion-authoritative heterogeneity rule would
require a new method identifier, external preregistration, and its own known-truth
operating-characteristic validation.

This is intentionally not a random-effects or hierarchical meta-analysis.

## Claim boundary

These simulations validate deterministic software behavior under declared truths. They do
not validate biological truth, establish a universal replication threshold, prove
population or task universality, or authorize a physical-quantum mechanism claim. The
final evaluation partition remains sealed and unexecuted.
