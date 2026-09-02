# Kumar2024 E001 confirmatory decision authority v1

## Purpose

The complete Kumar2024 cohort/evidence authority is frozen and merged, but a frozen dataset split is not a frozen scientific decision.

This authority defines **how a future confirmatory E001 result may be interpreted before any confirmatory mechanism outcome is computed or inspected**. It is deliberately separate from the existing E001 execution code.

Method ID:

`kumar2024_e001_confirmatory_decision_plan_v1`

Preregistration seal role:

`kumar2024_e001_confirmatory_preregistration_seal_v1`

This method does not execute E001 and does not authorize execution.

## Exact data/evidence authority

The decision plan binds the authority merged in PR #56:

- authority capsule fingerprint: `1013358b419436a3a9592c8a48eec2372701b1977e7ced06f4c25cfd4ebae29d`
- cohort authority fingerprint: `36cdfdf42e5ac375999d4defa02554cf4d2d04472ed6c06a08c389b5ad02b81c`
- raw dataset fingerprint: `c91c6dca34be880e688359e210686c1823461ad93923f71e947bb3d0725d6c8b`
- frozen QuantumBCI science source: `681ea12c436fce121ba74de6f877a8267e94dd3f`
- frozen neurOS authority source: `ffa28ed552dc75158b673fdcd70729b1c9c69b47`
- subjects: exactly `1..18`
- held-out session: exactly `5`
- evaluation fraction: `0.5`
- all 18 authority, partition, calibration-split, and processed-data fingerprints.

The read side recomputes the original cohort fingerprint from the serialized 18-case authority identities. Replacing a participant authority and then recomputing the outer decision-plan fingerprint is therefore rejected.

## What is the primary scientific question?

For the current density representation, v1 asks a narrow question:

> Does predictive performance depend on the off-diagonal cross-covariance structure represented by the density feature construction?

The primary estimand is participant-balanced:

`participant_mean_density_minus_offdiagonal_ablation`

The production participant bootstrap first aggregates correlated cases within participant and then resamples participants. The inference unit is fixed to `subject`.

A future registered plan must explicitly supply:

- one primary calibration budget per class;
- a non-negative minimum practically relevant effect;
- participant-bootstrap resample count;
- bootstrap seed;
- rationale for those choices.

There are **no production defaults for the scientific minimum effect** in this authority.

The v1 primary decision statistic is the 95% participant-bootstrap CI lower bound. A primary pass requires that lower bound to be at least the explicitly registered minimum effect.

This is intentionally stricter than declaring success because a point estimate is positive. `bootstrap_probability_positive` remains descriptive and is not promotion-authoritative in v1.

## Why off-diagonal ablation is primary

The existing density constructor is exactly equivalent to a trace-normalized covariance representation. Therefore a generic statement that “density beats classical controls” is not a coherent information-novelty hypothesis for the current constructor.

The off-diagonal ablation instead asks whether cross-feature covariance structure matters for the predictive result. A favorable answer supports dependence on that structure. It does **not** establish that the structure is uniquely quantum, biologically causal, or information-novel relative to covariance.

## Closed production control family

The authority freezes the exact production E001 names:

- candidate: `density`
- exact equivalence control: `normalized_covariance`
- `covariance`
- `log_covariance`
- `bilinear_second_moment`
- `pooled_mean_std`
- `pca_flattened`
- `diagonal_density`
- ablation: `offdiagonal_ablation`

The classical family is closed before confirmatory execution. The strongest classical control may still be reported as a descriptive adversary selected by maximum balanced accuracy within this predeclared family, but that data-dependent selection is **not** a second hidden promotion rule.

Multiplicity authority is `one_family_one_primary_v1`: the sole confirmatory primary hypothesis is `density_vs_offdiagonal_ablation`. Additional controls are adversarial/descriptive unless a future method preregisters additional confirmatory hypotheses and corresponding multiplicity control.

## Hard information-novelty ceiling

Production `evaluate_density_information_gate(...)` identifies mathematical equivalence between the current density constructor and normalized covariance and requires identical predictions for the exact-equivalence control.

Therefore v1 freezes:

- `information_novelty_promotion_eligible = false`
- `current_density_representation_information_novel = false`

No choice of effect threshold, bootstrap seed, calibration budget, or favorable E001 result may override this invariant.

A successful off-diagonal-ablation criterion could support language such as:

> Participant-level E001 performance depends on cross-covariance structure under the preregistered Kumar2024 benchmark.

It cannot support:

> The density representation contains information unavailable to covariance.

and cannot support a physical-quantum neural-substrate claim.

## Complete-cohort semantics

The plan fixes:

- complete 18-subject cohort required;
- no silent participant intersection;
- missing or invalid evidence is **not** a scientific null;
- a technical failure requires stop-and-adjudicate before further unblinding.

This prevents a failed participant from quietly disappearing from the confirmatory denominator or being counted as evidence against the mechanism.

A future executor must verify exact equality against the persisted authority capsule before it can adjudicate results.

## GR/PAR subgroup semantics

The original protocol groups are frozen:

- GR: subjects `1..9`
- PAR: subjects `10..18`

They are diagnostic-only in v1. They cannot become an alternate promotion path after inspecting subgroup results.

If a later study wants subgroup-specific confirmatory claims, that requires a new preregistered method with explicit multiplicity and minimum-evidence rules.

## External preregistration

`Kumar2024E001PreregistrationSeal` binds a `PreregistrationEvidence` record to the exact decision-plan fingerprint.

The external registration must provide its real:

- immutable registration URI;
- timestamp;
- registered document SHA-256;
- exact registered policy fingerprint.

QuantumBCI does not fabricate or contact an external registry. Software-fixture URI/hash values used in unit tests are not scientific registrations.

The seal still serializes:

- `evaluation_executed = false`
- `confirmatory_outcomes_observed = false`
- `execution_authorized = false`
- `information_novelty_promotion_eligible = false`
- `biological_mechanism_established = false`
- `physical_quantum_promotion_eligible = false`

So external registration of this decision plan is necessary but not sufficient to run the cohort.

## Why execution remains a separate tranche

The current `run_kumar2024_study(...)` execution path can reconstruct an authority from configuration. A publication-grade confirmatory executor should instead prove that every restored participant split, processed-data identity, source fingerprint, participant membership, and evaluation index is exactly the one in the merged authority capsule.

That executor/adjudicator does not exist in this tranche.

A future method should:

1. consume the exact persisted authority capsule;
2. consume an externally registered decision seal;
3. bind an exact qualified executor source SHA;
4. refuse any authority/split/control/budget mismatch;
5. execute each participant once under the registered policy;
6. construct participant-level bootstrap evidence using the registered resample count and seed;
7. adjudicate only the predeclared primary criterion;
8. preserve diagnostic controls/subgroups separately;
9. emit a read-side-verifiable result artifact;
10. keep the information-novelty and physical-quantum ceilings intact.

Only that later, independently qualified method could make `execution_authorized` true.

## Threshold-selection boundary

This repository intentionally does not choose a production `minimum_effect` in this tranche.

The value should be justified without Kumar confirmatory outcomes, for example from a predeclared practical-performance rationale, prior external evidence, measurement-resolution considerations, or a separately qualified known-truth operating analysis. It must not be selected because it makes the held-out Kumar result pass.

Likewise, the primary calibration budget and bootstrap resample count should be justified before execution. Existing software defaults or test fixtures are not scientific justification by themselves.

## Claim boundary

This authority is a software/scientific-decision contract. It does not:

- execute Kumar2024 E001;
- inspect a confirmatory prediction or effect;
- choose a production scientific threshold;
- establish biological causality;
- establish representation information novelty for the current covariance-equivalent density constructor;
- establish universal BCI generalization;
- authorize a physical-quantum claim.

The correct next milestone after this authority is qualified is external threshold/rationale selection and preregistration, followed by a **separate** authority-bound executor/adjudicator qualification. Real confirmatory E001 remains blocked until those gates are complete.
