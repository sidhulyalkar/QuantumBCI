# BMRB dependence and session-imbalance stress contract

This validation layer attacks a specific statistical failure mode that a row-oriented neural
analysis can easily hide: **participants are the inferential unit even when participants
contribute very different numbers of sessions or cases**.

The contract is synthetic software validation. It does not claim that the responder mixtures
below are biologically realistic, and it does not establish a neural or physical-quantum
mechanism.

## Why this exists

The existing BMRB stress suite already covers ordinary participant heterogeneity and repeated
noisy sessions. Those cases use balanced session counts. Balanced repeats cannot demonstrate
that a participant with many sessions is prevented from dominating the confirmatory estimand.

`BMRB_KNOWN_TRUTH_DEPENDENCE_STRESS_V1` therefore creates deliberately adversarial session
profiles in which a row-pooled effect summary points in the wrong scientific direction.

## Frozen attacks

### Majority-responder imbalanced positive

Seven of eight participants carry the declared positive mechanism. Each responder contributes
one session, while the single nonresponder contributes twenty sessions.

The known-truth expectation is:

- BMRB passes because participant-level support is strong;
- a naive row-weighted effect rule fails because the heavily sampled nonresponder dominates the
  row count;
- a simple participant-balanced diagnostic agrees with BMRB.

This is a **false-rejection trap** for row pooling.

### Minority-responder overweight trap

Two of eight participants carry a large positive effect and each contributes twenty sessions.
The six nonresponders contribute one session each.

The known-truth expectation is:

- BMRB fails and localizes the failure to the effect evidence;
- a naive row-weighted effect rule passes because the two responders dominate the row count;
- the participant-balanced diagnostic rejects.

This is a **false-promotion trap** for row pooling.

The participant-balanced diagnostic is not a replacement benchmark. It is a negative-control
calculation used to show which part of the row-pooled failure comes specifically from unequal
participant weighting.

## Structured missingness

The contract also removes selected latent-representation rows from otherwise paired repeated
sessions. The production confirmatory evaluator must reject the bundle as **software invalid**
because representation lanes are no longer exactly paired.

That outcome is intentionally distinct from a scientific negative:

- invalid pairing means the requested estimand cannot be evaluated under the declared contract;
- it must not be converted into a failed mechanism result;
- it must not be silently repaired by dropping unmatched cases from the other lane.

## Qualification scope

The CI qualification requires all of the following on the deterministic smoke scenarios:

- zero BMRB decision error;
- the majority-responder row-pooled rule exhibits its expected false rejection;
- the minority-responder row-pooled rule exhibits its expected false acceptance;
- the participant-balanced diagnostic agrees with the declared participant-level truth;
- the minority case localizes its BMRB failure to the effect component;
- structured missing representation rows are rejected as software invalid.

These are software-contract expectations for the frozen stress cases. They are not universal
scientific acceptance thresholds.

## Relationship to the operating-characteristics program

This module is a deterministic adversarial qualification layer, not the final Monte Carlo
methods grid. Its purpose is to prove that the production estimator behaves correctly before
the broader Stage B development study parameterizes:

- responder fractions and effect distributions;
- session-count imbalance severity;
- measurement noise;
- structured and informative missingness;
- participant count;
- near-boundary effect sizes;
- stronger matched comparator rules.

The final evaluation seed partition remains sealed. These dependence attacks belong to
development and qualification work until their broader parameter grid and acceptance rationale
are preregistered.

## Claim boundary

Passing this contract says that BMRB resists the declared row-weighting and exact-pairing
software attacks. It does **not** validate biological realism, establish causal neural truth, or
authorize a physical-quantum interpretation.
