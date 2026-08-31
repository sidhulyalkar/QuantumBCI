# BMRB study-level development execution

The study-level BMRB operating program is now scientifically defined, but the recommended
development surface is too large to treat as a single fragile process.

This execution layer makes that **development-only** computation deterministic, resumable,
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

The `smoke` grid exists only for deterministic qualification. The recommended grid remains
the actual development surface.

## What this tranche does not do

It does not:

- execute or expose final evaluation seeds;
- choose acceptance thresholds;
- convert development evidence into qualification automatically;
- make sensitivity promotion-authoritative;
- perform random-effects or hierarchical meta-analysis;
- validate biological truth;
- authorize a physical-quantum interpretation.

The final study-level evaluation partition remains sealed and unexecuted.

## Next evidence step

Once this execution layer is qualified, the highest-value scientific action is to run and
inspect the **complete recommended development grid**, not add another governance object.
The development result should be used to understand failure surfaces, runtime, low-N
behavior, noise/heterogeneity interactions, primary-role protection, and sensitivity-warning
behavior before any externally justified final acceptance thresholds are frozen.

After the benchmark-development surface is characterized, the project should increasingly
shift effort toward the first real Kumar2024 BMRB study and independent replication rather
than continuing to accumulate software authority without empirical evidence.
