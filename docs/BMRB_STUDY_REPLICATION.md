# BMRB study-level replication authority

Broad neural-mechanism claims require a higher evidence level than participant-level
inference inside one dataset. This layer freezes that higher level explicitly.

## Scientific unit

The replication unit is the **independent study/dataset**, not the participant.

Participant-level uncertainty remains inside each confirmatory study analysis. The
replication evaluator consumes one bounded `BMRBStudyEvidence` object per frozen study.
It does not accept trial, window, session, or participant rows.

This prevents a large dataset from manufacturing extra replication votes merely because
it contains more participants.

## Frozen study family

`BMRBStudyReplicationPolicy` preregisters:

- one primary study at order zero;
- one or more independent replication studies;
- a unique dataset identity for every study slot;
- the mechanism identity being replicated;
- the minimum number of successful replication studies;
- equal study-level weighting;
- participant weighting as diagnostic only;
- a scientific rationale;
- an optional external preregistration binding the exact policy fingerprint.

v1 deliberately requires the primary study to pass. A large successful secondary study
cannot retrospectively become the primary after the frozen primary fails.

Every frozen study must be reported. Missing failed studies and post-hoc extra studies
are rejected rather than silently changing the replication denominator.

## Independent-source protection

Unique dataset labels are necessary but not sufficient. `evaluate_study_replication(...)`
also rejects two study slots that reuse the same `source_fingerprint`.

This blocks a simple replication-laundering attack in which the same underlying evidence
is copied into multiple nominally independent study IDs.

## Three separate decisions

The result keeps three concepts separate:

1. **Within-study scientific evidence**: each study's existing confirmatory gates.
2. **Replication criteria**: the frozen primary passes and enough frozen independent
   replication studies pass.
3. **Broad-claim authority**: the replication policy is externally preregistered and all
   supplied study results have confirmatory authority.

Broad promotion requires both the replication criteria and broad-claim authority.

A failed broad-replication decision does **not** erase a positive study. Positive study
IDs remain visible, and `context_specific_only` distinguishes a context-specific signal
from evidence that has earned a broad replication claim.

## Participant imbalance adversary

`BMRB_STUDY_REPLICATION_IMBALANCE_STRESS_V1` freezes the scientific pattern:

- primary study: PASS;
- independent replication: FAIL.

It then swaps sample sizes:

- 500 primary participants / 20 replication participants;
- 20 primary participants / 500 replication participants.

The authoritative cross-study decision is FAIL in both cases because the replication
study failed. The study-positive fraction remains exactly 1/2.

The participant-weighted descriptive fraction, however, flips from above 0.9 to below
0.1. A naive participant-majority story therefore reverses even though the study-level
scientific evidence has not changed.

That participant-weighted quantity is retained as an adversarial diagnostic, not as a
promotion gate.

## Heterogeneity stays visible

v1 reports each study-level reference effect through the evidence objects and summarizes:

- unweighted study-effect mean;
- minimum study effect;
- maximum study effect;
- effect range;
- successful and failed replication study IDs.

This is intentionally not a random-effects meta-analysis. With only a small number of
studies, silently estimating a between-study variance and treating it as settled
inference would add a new scientific method without its own validation.

A future meta-analytic method should receive a new method identifier, explicit estimand,
known-truth simulations, and operating-characteristic qualification.

## Adapter from confirmatory evidence

`BMRBStudyEvidence.from_confirmatory_result(...)` converts a finished
`ConfirmatoryRepresentationResult` into one study-level evidence object. The adapter
carries forward:

- study and mechanism identity;
- participant count as descriptive study metadata;
- within-study scientific PASS/FAIL;
- confirmatory authority and promotion eligibility;
- reference-lane study effect and confidence interval;
- source fingerprint.

The participant count is never converted into replication weight.

## Claim boundary

This study-replication layer does not validate biological truth. It validates software
semantics for keeping study-level replication authority separate from participant-level
sample size.

It does not establish a universal replication threshold, does not prove a mechanism is
shared across all populations or tasks, and does not make a physical-quantum claim.

The sealed final-evaluation partition remains unexecuted. Development fixtures and CI
qualification are not final biological evaluation.

## Next methods frontier

After this authority layer is qualified, the strongest follow-on work is:

- study-level known-truth operating curves with varying heterogeneity;
- explicit context/moderator strata rather than silent pooling;
- a separately versioned random-effects or hierarchical meta-analytic method;
- sensitivity to one influential study;
- leave-one-study-out stability;
- publication/availability bias adversaries;
- binding this machine-readable replication policy into a future version of the final
  evaluation seal without mutating the existing sealed schema in place.
