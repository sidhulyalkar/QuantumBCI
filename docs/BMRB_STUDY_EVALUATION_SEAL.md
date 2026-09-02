# BMRB study-level evaluation seal

The study-level evaluation seal freezes the scientific authority required **before** a
future final cross-study operating evaluation can be run. It does not run that evaluation.

This is a separate method from the already-qualified participant-level BMRB evaluation
seal. The older seal remains intact. Study/dataset replication has different estimands and
failure modes, so its acceptance authority receives its own method ID and artifact role.

## Why a second seal is necessary

The merged study operating program now tests a hierarchy that did not exist when the
participant-level seal was designed:

1. participant-level known-truth evidence is generated inside one study;
2. exactly one bounded evidence object becomes one study vote;
3. a frozen primary plus independent replications determine broad replication;
4. non-authoritative sensitivity reports directional disagreement, influence and
   replication-margin fragility;
5. the outer operating program summarizes behavior over known cross-study truths.

A final evaluation of this hierarchy must not acquire its decision rule after those final
results are visible. The seal therefore binds the rule before evaluation.

## Development evidence is verified, not merely hashed

`verify_bmrb_study_operating_mapping(...)` checks a serialized development artifact before
it can enter a seal. It recomputes and validates:

- artifact and policy fingerprints;
- the exact benchmark and method IDs;
- grid identity and grid fingerprint;
- unique scenario and numeric grid axes, preventing duplicate-cell weighting;
- declared scenario truth semantics;
- replicate counts and attainable pass rates;
- decision-error identities;
- Wilson intervals;
- diagnostic fractions and effect-range validity;
- aggregate false-promotion, recovery, context and warning summaries;
- `qualification_defined = false`;
- `evaluation_partition_executed = false`;
- the physical-quantum claim ceiling.

Fingerprints are integrity checks, not signatures. The verifier therefore checks internal
scientific invariants in addition to fingerprint consistency.

## Exact development/evaluation policy pairing

A study evaluation plan carries both the verified development policy and the future
evaluation policy. After normalizing only the partition label and its derived policy
fingerprint, their scientific semantics must be identical.

That freezes, among other things:

- code/source revision identity;
- scenario grid and ordering;
- participant-count grid;
- within-study heterogeneity grid;
- measurement-noise grid;
- cross-study effect-variation grid;
- Monte Carlo replicate count;
- participant-level bootstrap count;
- sensitivity thresholds;
- seed-partition fingerprint.

The evaluation policy must use `partition="evaluation"`, while the public study operating
runner continues to reject that partition. Creating a seal therefore does not execute or
consume final evaluation evidence through the qualified runner.

## The v1 holdout is procedural, not cryptographic

The word "seal" describes a scientific-policy boundary in v1, not secret test-set entropy.
The deterministic `StudySimulationSeedPartition` parameters, including the evaluation
offset, are part of the public source and serialized authority. A researcher who deliberately
bypasses the qualified evaluation runner can therefore derive the deterministic evaluation
seed space from lower-level public components.

The v1 safeguards are still meaningful:

- the qualified runner refuses evaluation execution;
- development and evaluation seed spaces are disjoint;
- the exact future policy is fingerprinted before evaluation;
- external preregistration can bind the complete acceptance authority;
- any actual evaluation execution remains auditable as a procedural violation if it occurs
  before registration.

However, this must not be described as cryptographic blinding or as making synthetic final
outcomes technically inaccessible. The more precise language is **procedurally held out** or
**policy-enforced holdout**.

If a future benchmark requires cryptographic blinding, it should use a new method with
high-entropy evaluation seed material held outside the repository, a public hash commitment
made before development decisions are finalized, and a one-time reveal only after the exact
analysis/acceptance authority has been externally registered.

No study-level evaluation cell has been executed in the completed v1 development program.

## Full RNG authority is serialized by the seal

The merged v1 operating result stores a seed-partition fingerprint but not every seed
parameter. The seal closes that read-side gap without rewriting v1 artifacts: it serializes
the full `StudySimulationSeedPartition` authority and requires its fingerprint to match
both development and evaluation policies.

The authority binds development/evaluation offsets, cell/replicate/study strides, and
maximum cell/replicate/study capacities. The underlying constructor proves the complete
capacity-bounded development and evaluation seed spaces are disjoint.

Disjoint deterministic seed spaces prevent accidental development/evaluation overlap. They
do not by themselves create secret evaluation entropy.

## Explicit acceptance criteria, no default thresholds

`StudyOperatingAcceptanceCriterion` supports bounded aggregate and scenario-level metrics.
QuantumBCI supplies **no scientific numeric defaults**. Threshold values must be justified
and frozen before final evaluation.

The v1 schema currently requires criteria that include aggregate mean false-promotion and
aggregate mean known-positive recovery. The completed development surface showed that those
aggregates combine scientifically distinct scenario classes. They remain part of qualified
v1 serialization, but they should not be mistaken for classical Type-I error and power, and
new publication-grade acceptance semantics should receive a new method ID rather than
silently redefining v1 after observing development results.

At minimum, the qualified v1 plan requires explicit bounds for:

- aggregate mean false-promotion rate, with an upper bound;
- aggregate mean known-positive recovery, with a lower bound;
- homogeneous four-study positive recovery;
- homogeneous four-study null promotion;
- protection of the failed frozen primary against later positive replications;
- detection of a zero-margin directional conflict;
- sensitivity warning behavior when a five-study family has positive replication margin
  but still contains a directional reversal.

A future v2 should separate pure-null broad promotion, contextual leakage, failed-primary
replacement, homogeneous-positive recovery, conflict-tolerant recovery, and warning
calibration into explicit estimands rather than relying on mixed headline averages.

## Machine-readable hierarchy authority

`BMRBStudyHierarchyAuthority` binds:

- the exact ordered study-operating scenario contract fingerprint;
- `primary_plus_predeclared_replications_equal_study_vote_v1`;
- `primary_must_pass = true`;
- one independent study = one vote;
- participant weighting = diagnostic only;
- `study_effect_heterogeneity_leave_one_out_v1`;
- exact sensitivity thresholds;
- `sensitivity_promotion_authoritative = false`.

A sensitivity warning can therefore qualify a broad PASS as fragile or context-sensitive,
but it cannot silently become a new promotion veto inside v1.

The completed development surface reinforces that separation: the broad replication decision
behaved cleanly in pure-null controls while the direction-agreement warning was poorly
calibrated near true zero effects. That warning must remain non-authoritative in v1.

## Machine-readable multiplicity and adaptive-search authority

The older participant-level seal carried a free-text multiplicity field. The study-level
seal instead binds `BMRBStudySearchAuthority`.

It contains the exact canonical `BMRBMultiplicityPlan` and declares one of two adaptive
search states:

- `forbidden`: no adaptive discovery is permitted; or
- `predeclared_plan`: an exact canonical `BMRBAdaptiveSearchPlan` is bound.

When adaptive search is present, it must reference the same closed multiplicity family.
Regardless of discovery mode:

- the confirmatory evidence set is the complete closed multiplicity family;
- adaptive discovery never defines that confirmatory set;
- promotion authority stays with the multiplicity plan's predeclared primary.

This prevents outcome-dependent inspection from changing what counts as confirmatory
evidence after results are seen.

## External preregistration

`BMRBStudyEvaluationSeal` becomes valid only when an external `PreregistrationEvidence`
record binds the exact acceptance-plan fingerprint. The registration therefore commits to
all nested authorities together:

- verified development evidence identity;
- full RNG authority;
- development/evaluation policy pair;
- hierarchy and sensitivity authority;
- multiplicity and adaptive-search authority;
- every numeric acceptance criterion.

Changing any nested policy changes the plan fingerprint and invalidates the registration.

External preregistration is particularly important for v1 because the synthetic evaluation
seed authority is public. The scientific defense against researcher degrees of freedom is
therefore the timestamped frozen decision policy and transparent execution history, not a
claim that the deterministic test seeds are unknowable.

## What this does not authorize

The seal is an authorization object for a future **synthetic software benchmark**. It does
not:

- execute the final evaluation partition;
- make the final synthetic outcomes cryptographically inaccessible;
- validate biological truth;
- prove a neural mechanism is causal;
- establish universal replication or heterogeneity thresholds;
- create random-effects or hierarchical meta-analytic authority;
- turn sensitivity v1 into a promotion gate;
- allow adaptive discovery to select the confirmatory evidence set;
- authorize a physical-quantum substrate interpretation.

A future evaluation executor/adjudicator must be a separate method and should consume an
externally registered seal. Until that method exists and is separately qualified, the final
study-level evaluation partition remains procedurally held out and unexecuted.
