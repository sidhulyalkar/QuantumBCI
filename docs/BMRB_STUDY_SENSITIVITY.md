# BMRB study heterogeneity and influence sensitivity

Study-level replication answers whether a frozen primary plus enough independent
replications satisfy a declared broad-replication rule. It does not, by itself, tell us
whether that PASS is robust or precariously balanced on one study.

This v1 layer adds a **non-promotion-authoritative** sensitivity report.

## What it measures

Given a completed `BMRBStudyReplicationDecision`, the report computes:

- study-weighted direction agreement relative to the frozen primary study;
- minimum-to-maximum study effect range;
- leave-one-study-out unweighted effect means;
- maximum absolute leave-one-study-out mean shift;
- the most influential study by that shift;
- successful-replication margin above the frozen minimum;
- whether removing one successful replication would collapse the declared replication criterion.

Participant counts do not weight direction agreement or the leave-one-study-out effect
mean. The unit remains the independent study.

## Explicit sensitivity policy

`BMRBStudySensitivityPolicy` contains study-specific thresholds for:

- minimum direction agreement fraction;
- maximum study-effect range;
- maximum leave-one-study-out mean shift.

The policy can be externally preregistered and fingerprinted. These thresholds are not
universal biological constants.

Even a preregistered v1 sensitivity policy is **not promotion-authoritative**. It labels
fragility and heterogeneity but does not silently change the qualified replication
promotion decision.

## Replication-margin fragility

A broad replication PASS can occur with zero margin. For example, if a policy requires
two successful replications and exactly two pass, removing either successful replication
would make the criterion fail.

`single_successful_replication_removal_flips_claim` makes that dependency visible.

A PASS supported by three successful replications when only two were required has a
positive margin and is not vulnerable to loss of one successful replication under the
same frozen rule.

## Heterogeneity stress

`BMRB_STUDY_HETEROGENEITY_STRESS_V1` compares two four-study fixtures under the same
replication policy.

### Fragile PASS

- primary effect: 0.12, PASS;
- replication 1: 0.11, PASS;
- replication 2: 0.10, PASS;
- replication 3: -0.10, FAIL;
- minimum successful replications: 2.

The formal replication criterion passes, but the success margin is zero, one successful
replication removal collapses the rule, effect spread is large, and one conflicting
study strongly shifts the unweighted study mean.

### Redundant compact PASS

- primary effect: 0.12, PASS;
- replications: 0.11, 0.10, 0.09, all PASS;
- minimum successful replications: 2.

The replication criterion passes with positive margin, directions agree, effect spread
is compact, and no single successful replication removal collapses the rule.

The sensitivity layer distinguishes these two cases while leaving both underlying
replication promotion decisions unchanged.

## Why no random-effects model yet

This report intentionally does not estimate a between-study variance, pooled hierarchical
effect, or random-effects confidence interval. With few studies, such an estimator would
be a new inference method, not a harmless summary.

A future hierarchical/meta-analytic method should have:

- an explicit study-level estimand;
- a new method identifier;
- preregistered estimator choices;
- known-truth simulations over study count and true heterogeneity;
- bias, interval-coverage, type-I-error, power, and influential-study operating curves;
- missing-study/publication-bias adversaries.

## Claim boundary

This sensitivity layer does not validate biological truth. A clean sensitivity report
does not prove that a mechanism generalizes to every population, task, acquisition
system, or representation.

It does not create a physical-quantum claim, and it does not alter the sealed
final-evaluation partition. The final-evaluation partition remains unexecuted.
