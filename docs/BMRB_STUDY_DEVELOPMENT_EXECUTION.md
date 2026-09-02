# BMRB study-level development execution

The study-level BMRB operating program is scientifically defined and its complete
recommended development surface has now been executed, recomposed, independently verified,
and persisted as a reviewable evidence capsule.

This execution layer keeps the computation **development-only**, deterministic, resumable,
and inspectable without changing the qualified scientific policy or result schema.

## Why shard the development program

The recommended study-level grid contains 648 cells:

- 8 declared cross-study truth families;
- participant counts 4, 8, and 16;
- three within-study heterogeneity scales;
- three measurement-noise scales;
- three cross-study positive-effect variation scales.

With 8 outer Monte Carlo replicates per cell and 3 to 5 independent studies per scenario,
the full grid invokes the production participant-level validation path about **20,088
times**, before accounting for participant bootstrap work inside each validation call.

A one-process runner makes a late machine failure expensive and discourages actually
running the evidence program. Sharding is therefore an execution concern, not a new
scientific method.

## Scientific identity is unchanged

`bmrb_study_operating_shards` reuses:

- the exact `BMRBStudyOperatingPolicy`;
- the exact canonical grid order;
- the existing global cell index;
- the existing nested seed authority;
- production `run_study_operating_replicate(...)`;
- the existing cell summarization arithmetic;
- the existing `BMRBStudyOperatingResult` schema;
- the promotion-grade read-side verifier.

A cell run in shard 7 therefore receives the same `(cell, replicate, study)` seed as the
same cell in the monolithic v1 runner. Shard boundaries have no scientific meaning.

## Partial shards are never scientific results

Every shard explicitly records:

- `complete_operating_result = false`;
- `qualification_defined = false`;
- `evaluation_partition_executed = false`;
- `physical_quantum_promotion_eligible = false`.

A shard may be cached, copied, retried, or inspected for execution diagnostics. It must not
be quoted as the operating result of the frozen policy.

Only `merge_bmrb_study_operating_shards(...)` can create the ordinary complete result, and
it fails closed unless:

1. every shard binds the exact same complete policy;
2. there are no overlapping cell indices;
3. there are no missing cell indices;
4. there are no out-of-grid cell indices;
5. the merged cells appear in the canonical frozen order;
6. the resulting ordinary operating artifact passes the existing promotion-grade verifier.

This also means a convenient subset, a successful subset, or a subset that finished first
cannot silently become the scientific development result.

## Duplicate weighting is rejected at the execution boundary

The original v1 grid constructor predates promotion-grade artifact reuse and permits
repeated numeric axis values. The read-side verifier already rejects those repetitions
because duplicate axes would silently give some conditions extra aggregate weight.

The sharded execution layer adopts the stricter rule up front: scenario, participant,
heterogeneity, noise, and cross-study-effect axes must be unique before a publication-grade
shard plan can be created.

The original v1 result identity is left intact. This is a stricter execution boundary, not
a retroactive rewrite of qualified v1 artifacts.

## Complete development execution

The complete recommended grid was executed in GitHub Actions run `33467852855` against
exact qualified science source:

`681ea12c436fce121ba74de6f877a8267e94dd3f`

All 24 deterministic 27-cell shards completed successfully. The original merge job
successfully reconstructed the complete 648-cell result before its descriptive reporter
failed by treating the read-side verifier's `None` return value as a result object. The
reporter failure occurred after scientific recomposition and did not invalidate the shard
family.

The exact 24 source shards were subsequently recovered without rerunning simulation. The
race-safe recovery run `33596653260`:

1. downloaded the original 24 shard artifacts from run `33467852855`;
2. reconstructed the complete result using the qualified merge implementation;
3. passed `verify_bmrb_study_operating_mapping(...)`;
4. independently checked the capsule SHA-256 manifest;
5. uploaded the verified capsule;
6. committed the exact verified bytes to `evidence/bmrb-study-development-v1/`.

The complete scientific artifact fingerprint is:

`53e18166c3bbf071e929d13d79e1eef09d9046d9a99d49536d728a4c0ff36879`

Independent recovery attempts over the same original shard family produced the same
policy, aggregate mapping, 648 canonical cell identities, cell values, and artifact
fingerprint.

## What the complete surface says

The persisted capsule reports the following applicability-aware coarse development
summaries:

- pure-null broad promotion rate: **0.000000**;
- homogeneous-positive broad recovery rate: **0.994599**;
- contextual or failed-primary broad promotion rate: **0.000000**;
- conflicted-positive broad recovery rate: **0.993827**;
- failed-primary role protection in its applicable adversary: **1.000000**;
- fragile-conflict detection in its applicable adversary: **1.000000**.

These are not final acceptance thresholds and are not precision estimates. With only eight
outer replicates per cell, rates move in increments of 0.125. For example, observing 0/8
events still leaves a wide two-sided 95% Wilson interval whose upper bound exceeds 0.3.
The development surface is therefore useful for mapping gross failure regimes, not for
claiming tight tail probabilities.

## The sensitivity diagnostic exposed a real semantic weakness

Broad replication authority behaved cleanly under the homogeneous null scenarios, with
zero broad promotions in the complete development surface. The v1 sensitivity warning,
however, is poorly calibrated near a true zero effect:

- `homogeneous-null-3` sensitivity-warning truth-match rate is approximately **0.2485**;
- `homogeneous-null-4` sensitivity-warning truth-match rate is approximately **0.5154**.

This pattern is expected from the current rule because directional agreement is measured
relative to the observed sign of the primary study effect. When the true effect is zero,
that sign is noise. A future method should use explicit null/practical-null applicability
or magnitude gating rather than tuning the v1 threshold after seeing these results.

Sensitivity v1 therefore remains diagnostic and non-promotion-authoritative.

## Applicability-aware reporting is required

The original v1 aggregate fields remain inside the scientific artifact for schema fidelity,
but they must not be overinterpreted:

- `mean_false_promotion_rate` combines scientifically different non-broad-truth scenarios
  and is not a classical Type-I error estimand;
- `mean_known_positive_recovery_rate` combines homogeneous and deliberately conflicted
  positive scenarios and is not a single classical power estimand;
- `primary_role_protection_rate` and `fragile_claim_detection_rate` use `1.0` in many
  scenarios where the metric is not applicable;
- `cross_study_effect_scale` is inert for all-null scenarios and perturbs candidate effect
  only for positive-labelled studies, so a global marginal would be misleading.

The persisted development analysis therefore reports scenario-conditional summaries and
only evaluates the special-purpose diagnostics in scenarios where they are defined.

## Development-only CLI

The CLI intentionally exposes no evaluation-partition flag.

Create a deterministic plan:

```bash
python -m quantumbci.bmrb_study_operating_shard_cli plan \
  --study-id bmrb-study-development-v1 \
  --source-sha <exact-source-sha> \
  --cells-per-shard 32 \
  --output artifacts/study-operating/plan.json
```

Run one shard:

```bash
python -m quantumbci.bmrb_study_operating_shard_cli run \
  --study-id bmrb-study-development-v1 \
  --source-sha <exact-source-sha> \
  --start-cell 0 \
  --stop-cell 32 \
  --output artifacts/study-operating/shard-000-032.json
```

Merge only after the complete shard family exists:

```bash
python -m quantumbci.bmrb_study_operating_shard_cli merge \
  --study-id bmrb-study-development-v1 \
  --source-sha <exact-source-sha> \
  --output artifacts/study-operating/development-result.json \
  artifacts/study-operating/shard-*.json
```

The source SHA, grid selection, replicate count, and bootstrap count must be identical for
every invocation. Policy drift causes the merge to fail.

The `smoke` grid exists only for deterministic qualification. The recommended grid is the
actual completed development surface.

## What this evidence does not do

It does not:

- execute final evaluation;
- choose or justify final numeric acceptance thresholds;
- convert development evidence into qualification automatically;
- make sensitivity promotion-authoritative;
- perform random-effects or hierarchical meta-analysis;
- validate biological truth;
- authorize a physical-quantum interpretation.

The synthetic final-evaluation partition remains **procedurally held out and unexecuted**.
The v1 deterministic evaluation seed authority is public, so this is a policy-enforced
holdout rather than cryptographic blinding. No evaluation cell was executed in producing
this development evidence.

## Next evidence step

The complete v1 surface has done its job: it validated the broad hierarchy under clear
truths and exposed where the benchmark itself is too coarse or semantically weak.

The next synthetic method should receive a new method ID and focus on:

- near-boundary candidate effects rather than only clear null/strong-positive regimes;
- independent ablation-necessity margins and heterogeneity;
- explicit metric applicability rather than encoding not-applicable as 1.0;
- magnitude-gated or not-applicable sensitivity behavior near practical nulls;
- stratified error estimands rather than mixed aggregate false-promotion/recovery scores;
- substantially higher replicate counts in targeted cells for precision operating curves;
- externally held seed entropy if cryptographic blinding is desired.

At the same time, project effort should increasingly move to the outcome-blind Kumar2024
cohort authority, external preregistration, and then the first real BMRB mechanism study.
The development surface should not become an excuse to postpone empirical neuroscience.
