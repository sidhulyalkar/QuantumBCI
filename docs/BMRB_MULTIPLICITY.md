# BMRB multiplicity and candidate-family authority

BMRB already separates effect, classical-adversary survival, conservation, and coverage. That is not enough if a researcher can run many candidate mechanisms, layers, tasks, or metrics and then promote whichever survivor looks best.

`quantumbci.bmrb_multiplicity` closes that loophole before adding any new p-value gate. `quantumbci.bmrb_multiplicity_stress` then attacks the authority with near-boundary known-null candidates passed through the production BMRB evaluator.

## Why v1 starts with authority rather than a correction formula

The confirmatory BMRB evaluator does **not** currently use a p-value threshold as its hidden promotion rule. Participant-level sign-flip p-values and bootstrap intervals are uncertainty evidence, while promotion is controlled by predeclared effect/adversary/conservation/coverage criteria.

Blindly attaching Holm, Bonferroni, FDR, or another multiple-testing procedure to those p-values would therefore change the scientific decision rule rather than merely make an existing rule safer.

The first multiplicity problem is more basic and easier to falsify:

> Was the complete candidate family declared before final evidence was inspected, and can a non-primary survivor acquire promotion authority after the fact?

v1 answers that with grouping, ordering, roles, closed-world result matching, a content fingerprint, and a development-only winner-picking stress.

## Candidate roles

Every candidate belongs to one declared family and has one role:

- `primary`: the single promotion-authoritative candidate for that family in v1;
- `secondary`: retained and reported as confirmatory/characterization evidence, but not promotion-authoritative under the v1 primary-only rule;
- `exploratory`: hypothesis-generating evidence that cannot be promoted post hoc.

Each family must contain exactly one primary candidate, and that candidate must occupy order zero. Candidate order is contiguous and frozen.

This is intentionally conservative. A future procedure that permits several primary hypotheses, alpha transfer, Holm correction, hierarchical FDR, or another multiplicity strategy should receive a new explicit method ID and its own operating-characteristic validation rather than silently broadening v1.

## Closed-world results

`apply_multiplicity_plan(...)` requires the observed result mapping to contain **every declared candidate exactly once and no undeclared candidate**.

That means:

- a failed candidate cannot disappear from the family after evaluation;
- a new candidate cannot be added because it looks promising;
- renaming/reordering roles changes the plan fingerprint;
- removing a candidate changes the plan fingerprint;
- a scientific PASS from a secondary/exploratory candidate is visible but does not become a promoted result.

The decision artifact explicitly reports both:

```text
naive_any_survivor
authorized_any_promotion
```

so a winner-picking trap can be visible rather than hidden inside a final headline.

## Known-truth candidate-search stress

`BMRB_CANDIDATE_SEARCH_MULTIPLICITY_STRESS_V1` uses the production `run_validation_replicate(...)` path rather than inventing a toy pass/fail simulator.

Every searched candidate is generated from the same declared near-boundary null DGM:

```text
BMRB effect threshold:       0.050
true reference effect:       0.049
true alternate-lane effect:  0.049
```

The mean effect is therefore on the null side of the frozen software-validation boundary. Moderate participant heterogeneity means an individual candidate can occasionally survive all gates by chance.

For each simulated search family the stress records:

- how many candidates scientifically passed;
- whether the predeclared primary passed;
- whether a naive `any survivor` search would report success;
- whether the multiplicity authority permits a promotion;
- how many non-primary survivors were retained but suppressed from promotion.

The crucial invariance test expands the same search from 2 to 20 candidates while preserving the primary candidate's exact simulation seed. Searching more candidates may increase the naive survivor count, but it **cannot change the primary candidate result or transfer promotion authority**.

The stress is development simulation authority only. It is not connected to the frozen final-evaluation seed partition.

## Example

```python
from quantumbci.bmrb_multiplicity import (
    BMRBMultiplicityCandidate,
    BMRBMultiplicityPlan,
    apply_multiplicity_plan,
)

plan = BMRBMultiplicityPlan(
    plan_id="mechanism-family-v1",
    family_order=("motor-imagery-mechanisms",),
    candidates=(
        BMRBMultiplicityCandidate(
            candidate_id="contextual-dynamics-primary",
            family_id="motor-imagery-mechanisms",
            role="primary",
            order=0,
            rationale="Primary non-equivalent mechanism frozen before final evidence.",
        ),
        BMRBMultiplicityCandidate(
            candidate_id="alternative-operator-exploratory",
            family_id="motor-imagery-mechanisms",
            role="exploratory",
            order=1,
            rationale="Retained for hypothesis generation only.",
        ),
    ),
    scientific_rationale="Prevent post-hoc winner selection across the candidate family.",
)

decision = apply_multiplicity_plan(
    plan,
    {
        "contextual-dynamics-primary": False,
        "alternative-operator-exploratory": True,
    },
)

assert decision.naive_any_survivor is True
assert decision.authorized_any_promotion is False
```

The exploratory PASS is not erased. It is simply not relabelled as the preregistered primary result.

## What this does not solve yet

v1 does not claim that one-primary-per-family is the optimal policy for every study. It does not yet provide:

- corrected multi-primary testing;
- alpha allocation or transfer;
- hierarchical familywise-error control;
- false-discovery-rate procedures;
- correlated candidate-search operating curves;
- study/dataset-level hierarchical replication;
- multiplicity across adaptive model/layer selection unless those choices are represented as candidates in the frozen family.

Those should be added as explicit methods and attacked with known-truth simulations before they receive promotion authority.

## Relationship to the final-evaluation seal

The existing operating-evaluation seal currently contains a human-readable `multiplicity_policy` field. The new candidate-family plan is the machine-readable scientific authority that should eventually replace or bind that prose in a versioned seal schema.

This development branch does **not** mutate the sealed final-evaluation partition or execute its seeds. The safe sequence is:

1. qualify candidate-family authority and winner-picking stress on development evidence;
2. extend the stress to correlated candidate families and adaptive layer/task searches;
3. define any corrected multi-primary method explicitly;
4. version and bind the chosen multiplicity authority into a future preregistered evaluation-seal schema;
5. only then consider opening the final evaluation partition.

## Claim boundary

Multiplicity authority controls reporting and promotion semantics under a declared candidate family. The synthetic stress estimates behavior under declared known-null candidate searches. Neither validates biological truth, establishes neural causal necessity, or authorizes a physical-quantum interpretation.
