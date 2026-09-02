# Kumar2024 cohort authority freeze

QuantumBCI now has an outcome-blind structural authority for the complete public Kumar2024 cohort. The purpose of this tranche is to freeze **what data and evidence partitions a future registered E001 study would use before any mechanism outcome is computed**.

It is not an E001 result and it is not evidence that any candidate mechanism succeeds.

## Exact source authority

The successful full-cohort freeze was produced by GitHub Actions run `33596812399` from tooling head `232b0105b02bb21dee7a22c902fcc6805ef714a5` while checking out:

- QuantumBCI scientific source `681ea12c436fce121ba74de6f877a8267e94dd3f`;
- neurOS authority source `ffa28ed552dc75158b673fdcd70729b1c9c69b47`.

The run completed every fail-closed step, including the source-code guard against mechanism executors, all 18 participant authority constructions, complete-cohort validation, outcome-field scanning, and artifact upload.

The uploaded source artifact is `Kumar2024-full-authority-freeze-681ea12c436fce121ba74de6f877a8267e94dd3f`, artifact ID `9835103441`, with ZIP digest `sha256:2a5c014ee4f7fcfe2b05d8d208717a597185652a870bff2fef740e7196e828d5`.

## Frozen cohort and longitudinal geometry

The authority binds exactly:

- MOABB Kumar2024 subjects `1..18`;
- original protocol groups GR = subjects `1..9` and PAR = subjects `10..18`;
- sessions `0,1,2,3,4,5` in chronological order;
- session `5` as the held-out occasion;
- sessions `0..4` as source history;
- 8–30 Hz preprocessing with native sampling preserved;
- 22 EEG channels;
- 2561 samples per epoch;
- evaluation fraction `0.5` inside the held-out occasion;
- one exact deterministic split seed and authority fingerprint per participant.

The real cleaned dataset geometry is preserved rather than rounded into an idealized 60-trial session. Across participants, total processed epochs range from 391 to 400. Held-out calibration pools range from 28 to 30 trials, while evaluation sets range from 29 to 30 trials. For every participant, source-history, calibration, and evaluation indices are pairwise disjoint and together cover every processed epoch exactly once.

## Raw-source identity is explicit

The freeze fingerprints the exact Kumar2024 bar-feedback GDF corpus that MOABB can consume for the selected participants:

- 360 unique GDF files;
- 4,193,818,400 total bytes;
- raw dataset fingerprint `c91c6dca34be880e688359e210686c1823461ad93923f71e947bb3d0725d6c8b`.

`Race/**` is explicitly excluded because it is outside the declared MOABB bar-feedback path.

A subtle but important upstream mapping is also frozen. MOABB subjects `1..9` map directly to raw subjects `1..9`, while MOABB subjects `10..18` map to raw subjects `11..19`. The authority therefore does not assume that MOABB participant IDs are interchangeable with raw folder numbers.

## Cohort authority identity

The complete 18-participant evidence assignment has cohort authority fingerprint:

`36cdfdf42e5ac375999d4defa02554cf4d2d04472ed6c06a08c389b5ad02b81c`

The persisted repository capsule additionally has its own canonical capsule fingerprint. That capsule embeds:

1. the full cohort freeze summary;
2. all 18 complete longitudinal authority mappings, including exact indices;
3. the complete raw-source fingerprint and all 360 file hashes;
4. the original producing workflow's SHA-256 file manifest;
5. source workflow/artifact provenance;
6. an explicit claim boundary.

The read-side verifier reconstructs the producing workflow's pretty-JSON bytes from the embedded mappings and requires the resulting 20 file hashes to match the original Action manifest exactly. It also recomputes all canonical internal fingerprints and validates the partition semantics independently.

## What was inspected

This is **design-stage structural inspection**. The workflow necessarily inspected:

- source file names, byte content and hashes;
- participant/session identity;
- processed tensor geometry;
- class labels only for constructing the predeclared stratified calibration/evaluation split;
- the exact resulting index assignments.

Because class labels are used to build the split, this should not be described as an untouched-data exercise. The correct claim is narrower: **no mechanism outcome was computed or inspected**.

## What remains outcome-blind

The producing workflow and persisted capsule explicitly record:

- `e001_executed = false`;
- `predictions_computed = false`;
- `mechanism_effects_computed = false`;
- `control_comparisons_computed = false`;
- `confirmatory_outcomes_observed = false`;
- `biological_mechanism_established = false`;
- `physical_quantum_promotion_eligible = false`.

The verifier rejects outcome-shaped fields such as candidate/control metrics, effect deltas, p-values, bootstrap confidence bounds, predictions, accuracy, or AUC even if an attacker recomputes the outer capsule fingerprint.

## Why freeze the exact evaluation indices now

A future confirmatory study should not decide which session-5 trials count as calibration versus evaluation after seeing E001 behavior. Freezing the indices now converts that choice from an analyst degree of freedom into preregistration authority.

This does **not** make the evaluation samples secret. The repository is public and the split authority is inspectable. The protection is procedural and auditable: the evidence assignment is fixed before mechanism outcomes exist.

If future work requires cryptographic blinding rather than a public preregistered assignment, it needs a new method with externally held entropy and a precommitted hash. That should not be retroactively claimed for this v1 authority.

## What is still missing before a real confirmatory E001 execution

The cohort/data authority is now concrete, but the **decision authority is not yet fully preregistered**. Before running mechanism outcomes across these participants, a separate registered plan should freeze at least:

- the exact candidate mechanism and ablation/control family;
- the primary estimand and participant-level summary;
- the minimum practically meaningful effect, if one is used;
- whether promotion is based on observed effect, a lower confidence bound, a non-inferiority/superiority margin, or another predeclared rule;
- multiplicity across candidate/control comparisons;
- missing/failed participant handling;
- GR/PAR subgroup reporting versus promotion authority;
- participant-level versus study-level inference boundaries;
- bootstrap/resampling authority and seeds;
- what constitutes a technical failure rather than scientific null evidence;
- the exact registered output schema and claim language.

The current freeze should be an input to that preregistration. It should not be mistaken for the preregistration itself.

## Scientific ceiling

This artifact can establish that an empirical test was attached to a predeclared public dataset, cohort, chronology, and evidence partition. It cannot establish biological causality by itself, universal BCI generalization, a physical-quantum neural substrate, or the truth of any mechanism family.

The next high-value step is therefore **decision-rule preregistration**, not E001 execution.
