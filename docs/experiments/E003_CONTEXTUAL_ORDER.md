# E003: Contextuality and order effects

**Claim ceiling:** quantum-inspired contextual probability.

Recent work continues to explore non-commutative probability as a framework for contextual cognition, but this experiment explicitly separates that modelling choice from claims of microscopic quantum brain physics.

## Core question

For a task in which cue A followed by B is experimentally distinguishable from B followed by A, does a compact non-commuting operator model predict behavior/neural readouts better than classical models that are explicitly given the same history?

## Phase 1: retrospective discovery

Use an existing dataset only if it contains exact trial order/history metadata. Fit the contextual model and classical history controls. This phase selects plausible observables and effect scales; it is not confirmatory and cannot be counted as replication.

## Phase 2: prospective preregistration

Design a balanced within-subject AB/BA task with randomized block/order assignment and a task where contextual influence is behaviorally meaningful. Power the sample size by simulation using conservative effect distributions from Phase 1 rather than selecting a convenient N.

Before collection, freeze:

- behavioral primary endpoint;
- EEG time windows/features or encoder checkpoint;
- artifact rejection rule;
- exclusion criteria;
- model families and complexity budgets;
- primary order-effect contrast;
- stopping/recruitment rule;
- replication cohort or held-out confirmation design.

Human data collection is blocked until applicable ethics/IRB approval and consent/data-retention rules are recorded.

## Models

Quantum-inspired:

- two or more observables/projectors;
- learned non-commutativity constrained to a small state space;
- AB and BA likelihoods from explicit sequential updates.

Required controls:

- history-augmented logistic/GLM with A, B, order and interactions;
- HMM/state-space history model;
- capacity-matched small RNN/GRU if data volume allows;
- permutation/no-order null.

## EEG endpoints

Use ERP/spectral features first because they are inspectable. A validated E001 encoder can be secondary. Test whether the neural state after A predicts a different B response than the matched state after B predicts A, while controlling baseline, block, fatigue/time, and previous-trial history.

## Statistics and promotion

Primary comparison is held-out predictive log score with a complexity-aware secondary criterion. The behavioral/neural AB-vs-BA effect must replicate prospectively. A contextual model is promoted only if it improves held-out prediction after classical models receive explicit order/history variables and matched capacity.

Failure to beat a history-aware classical model falsifies the claim that non-commutative structure is needed for this task.

## Interpretation firewall

Even a strong replicated order effect supports only the usefulness of a contextual/non-commutative probability model. It cannot establish quantum coherence, entanglement, or quantum computation in neural tissue.

## Reference

- Emori, Khrennikov & Iriki (2026), non-commutative structures of brain and cognition: https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2026.1882287/full
