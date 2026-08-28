# BMRB Final-Evaluation Seal

The Stage B operating-characteristics layer separates development and evaluation Monte Carlo authority. This document defines the next boundary: **how to freeze the final synthetic evaluation decision rule before any evaluation seed is observed**.

## Why a seal exists

A development study is allowed to inform scientific design. A final evaluation is supposed to test that design. If thresholds, endpoints, scenario selection, or multiplicity rules are edited after final-evaluation results are visible, the evaluation no longer provides an honest independent check.

QuantumBCI therefore separates two objects:

1. `BMRBOperatingAcceptancePlan`: the complete machine-readable decision plan.
2. `BMRBEvaluationSeal`: that exact plan plus externally timestamped preregistration evidence whose registered policy hash equals the plan fingerprint.

The seal is an integrity and provenance contract. It is **not** a result and it does not run the evaluation partition.

## Required sequence

1. Run and verify the **development** operating-characteristics study.
2. Preserve the development artifact and its policy fingerprint.
3. Inspect development evidence and justify the final evaluation design scientifically.
4. Construct an exact `BMRBOperatingStudyPolicy` with `partition="evaluation"`. Do not execute it.
5. Declare explicit acceptance criteria, their numeric bounds, and a rationale for every bound.
6. Declare the multiplicity/reporting policy. Endpoints may not be silently dropped after evaluation.
7. Build `BMRBOperatingAcceptancePlan` and record `plan.plan_fingerprint`.
8. Put the plan fingerprint and full scientific rationale in an external immutable/timestamped registration document.
9. Construct `PreregistrationEvidence` whose `registered_policy_sha256` equals the plan fingerprint.
10. Construct and archive `BMRBEvaluationSeal`.
11. Independently verify the external registration timestamp/URI and the local seal.
12. Only then may a separate future execution lane consume the sealed evaluation policy.

The current seal module intentionally contains no `run_bmrb_operating_characteristics` call and no final-evaluation command.

## Thresholds are not library defaults

QuantumBCI does not know a universally correct false-promotion rate, recovery rate, coverage target, or gate-localization threshold. The appropriate values depend on the scientific claim, simulation design, Monte Carlo precision, and consequences of false promotion.

For that reason, `OperatingAcceptanceCriterion` requires explicit caller-supplied bounds and rationale. There are no scientific numeric defaults.

The plan does require that a final BMRB validation address certain **structural endpoints** from the methods program:

- aggregate false-promotion behavior;
- aggregate known-positive recovery;
- explicit pass behavior for the exact-equivalence null;
- explicit pass behavior for the predictive-shortcut null;
- explicit pass behavior for the shared-mechanism positive case.

Those requirements prevent a final evaluation from quietly omitting the cases most relevant to the benchmark's falsification claims. They do not choose the acceptable numbers.

## Development evidence binding

The plan binds:

- `development_evidence_ref`;
- the development operating artifact fingerprint;
- the development operating policy fingerprint;
- the exact future evaluation policy, including its grid, source revision, replicate count, calibration budget, bootstrap resampling, and evaluation seed-partition authority;
- all criteria and bounds;
- multiplicity/reporting policy;
- scientific rationale.

Changing any of these changes the plan fingerprint and invalidates an existing preregistration binding.

## External preregistration

`PreregistrationEvidence` records:

- registration URI;
- timezone-aware registration timestamp;
- SHA-256 of the complete registered document;
- SHA-256/fingerprint of the exact registered machine-readable plan;
- optional registry name.

QuantumBCI verifies that the supplied registration record binds the current plan. It does **not** contact OSF or another registry and cannot prove that an external service's timestamp is authentic. Registry authenticity and chronology must be independently checked during review/publication.

## Tamper behavior

The serialized seal is canonical and fingerprinted. Read-side verification rejects, among other things:

- a stale outer artifact fingerprint;
- stale or noncanonical nested evaluation-policy fingerprints;
- post-hoc threshold changes;
- a development policy disguised as final evaluation;
- missing core null/positive endpoints;
- criteria targeting scenarios outside the frozen grid;
- an external preregistration hash that does not match the exact plan;
- any artifact claiming `evaluation_executed=true` while presenting itself as a preregistration seal.

Fingerprints are integrity checks, not cryptographic signatures. External registry verification remains necessary.

## What this does not establish

A correctly sealed evaluation can make the later synthetic methods result more credible. It does **not** by itself establish that:

- the data-generating mechanisms are biologically realistic;
- a candidate mechanism is necessary in real neural systems;
- a representation is uniquely mechanistic;
- any physical-quantum substrate has been demonstrated.

Those claim ceilings remain unchanged.

## Next methods work

Before the actual frozen evaluation is executed, issue #22 still calls for richer development evidence, especially responder-mixture heterogeneity, repeated-session dependence, structured missingness/invalid evidence, gate confusion/sensitivity/specificity, naive-comparator operating curves, runtime reporting, near-boundary nulls, and an explicit multiplicity strategy. The seal provides the machinery to freeze the final decision rule after that development work is complete.
