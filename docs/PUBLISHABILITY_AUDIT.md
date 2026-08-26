# QuantumBCI publishability audit

This document is a deliberately critical assessment of what would make QuantumBCI a
credible community research method rather than a technically elaborate demonstration.

## Proposed scientific identity

The strongest publishable contribution is **not** "quantum EEG" and is not a new EEG
accuracy leaderboard.

A stronger framing is:

> **BMRB is an adversarial, preregisterable benchmark for testing whether a candidate
> neural mechanism survives mathematical equivalence, matched classical alternatives,
> representation changes, repeated evidence, and intervention.**

Quantum-structured models are one candidate family that can be tested inside this
framework. Classical, mechanistic, and foundation-model hypotheses should be able to use
the same evidence machinery.

That framing makes negative results first-class. If a density construction is exactly
normalized covariance, a covariance-equivalence failure is a successful scientific
falsification rather than a disappointing benchmark score.

## What is already unusually strong

QuantumBCI already has several properties worth preserving:

- exact evidence and split authority rather than convenience random splits;
- content-addressed run and representation identities;
- mathematical-equivalence gates before empirical novelty claims;
- participant-level rather than window-level promotion inference;
- matched-information-set classical adversaries for dynamics;
- explicit causal/intervention evidence that cannot erase an earlier falsifier;
- strict physical-quantum claim ceilings;
- closed-world artifact verification;
- RO-Crate and conservative BIDS-aware export;
- pinned model/repository revisions for optional integrations;
- installed CLI and wheel qualification across Python versions.

These are stronger scientific-engineering foundations than simply adding more models.

## Publication blockers found in the v0.17 audit

### 1. Preregistration was self-declared

BMRB v1 could carry `preregistered: true`, but a boolean does not establish that the
policy existed before the result was inspected.

v0.18 adds an external preregistration evidence contract containing:

- external registration URI;
- timezone-aware registration timestamp;
- SHA-256 of the complete registered document;
- SHA-256 fingerprint of the exact machine-readable decision policy.

QuantumBCI does not pretend to authenticate an external registry offline. The run binds
the supplied registry reference and hashes so reviewers can independently verify the
external record.

### 2. Calibration budgets were averaged into one primary effect

Averaging 0-shot, 1-shot, 5-shot, and 10-shot effects can hide a budget-dependent reversal
and produces an ambiguous confirmatory estimand.

The v0.18 confirmatory protocol requires one **primary calibration budget** before
analysis. Every other budget remains in a secondary calibration frontier and is never
averaged into the primary effect.

### 3. The strongest classical control was selected on final evaluation

E001 v1 reports the best observed classical control on the final evaluation set. That is
a useful descriptive stress test, but selecting the comparator after observing final-test
performance is not a clean confirmatory analysis.

The v0.18 confirmatory protocol instead requires `primary_classical_control` in the
registered policy. For the current density constructor, the natural primary control is
`normalized_covariance` because it is the exact mathematical equivalent. A
`best-observed-on-evaluation` control may still be reported descriptively, but it cannot
become the confirmatory comparator after the fact.

### 4. Uncertainty reporting was too thin

Direction fractions are useful but insufficient for a paper. v0.18 adds participant-level
bootstrap confidence intervals and deterministic two-sided sign-flip tests. The
participant remains the inference unit. These uncertainty summaries do not become hidden
p-value gates; the registered decision rule remains authoritative.

### 5. Release metadata had drifted

`CITATION.cff` had remained at v0.14 while the package had reached v0.17. v0.18 synchronizes
citation metadata and expands package metadata, project URLs, research keywords, and
classifiers.

## Important gaps that remain after v0.18

v0.18 improves confirmatory correctness, but it does **not** make a paper complete.

### Validate BMRB itself before trusting BMRB on biology

A methods paper needs ground-truth simulations where the answer is known. At minimum,
construct an ADEMP-style simulation program:

1. **Aims**: estimate false-positive, false-negative, effect-recovery, and gate-calibration
   behavior.
2. **Data-generating mechanisms**:
   - exact classical-equivalence null;
   - true candidate mechanism;
   - confounded predictive shortcut;
   - representation-specific mechanism;
   - shared mechanism under invertible coordinate changes;
   - participant heterogeneity;
   - missing/noisy sessions;
   - calibration-budget reversals.
3. **Estimands**: participant mean effect, ablation effect, cross-representation
   conservation, adversary survival, and promotion ceiling.
4. **Methods**: BMRB plus simpler baselines and naive similarity/accuracy rules.
5. **Performance measures**: type-I error, power, interval coverage, effect bias,
   calibration, and failure-mode localization.

A benchmark that cannot reject known nulls or recover known mechanisms should not be
interpreted on EEG.

### Representation evidence must become multidimensional

Sign agreement and participant correlation are a useful floor, not a complete theory of
representation. Recent neuroscience work distinguishes at least sensitivity, specificity,
invariance, and downstream functional use.

Future BMRB-Representation should report separate evidence axes rather than one composite
"similarity" score:

- **geometry**: e.g. carefully interpreted CKA/Procrustes/neighborhood structure;
- **functional predictivity**: forward and, where defensible, reverse prediction;
- **specificity**: candidate information relative to nuisance/task alternatives;
- **invariance**: robustness to declared nuisance transforms;
- **functional dependence**: erasure/ablation/intervention consequences.

No single CKA, RSA, probing accuracy, or correlation should be promoted as "the same
mechanism".

Relevant context:

- Pohl et al., *Clarifying the conceptual dimensions of representation in neuroscience*,
  Nature Reviews Neuroscience (2026), DOI: 10.1038/s41583-026-01030-8.
- Tang et al., *What Do EEG Foundation Models Capture from Human Brain Signals?* (2026),
  arXiv:2605.11410, which separates what a model learns, what it uses, and how much is
  explained using probing and cross-covariance subspace erasure.

### Foundation-model evaluation needs multiple transfer regimes

Frozen embeddings alone are not a complete EEG-FM comparison. Current broad benchmarking
finds that linear probing is frequently insufficient and specialist models trained from
scratch remain competitive.

A real paper should compare separately labelled regimes under their own evidence authority:

- frozen / zero-shot representation;
- linear probe;
- parameter-efficient adapter, if supported;
- full fine-tuning, if powered;
- specialist scratch-trained controls such as CSP/LDA, EEGNet, and EEG-Conformer;
- random or untrained representation controls.

Checkpoint, layer, pooling, normalization, channel mapping, sample rate, crop/window,
transfer regime, and every learned adaptation surface must be frozen before final
evaluation.

Relevant context:

- Liu et al., *EEG Foundation Models: Progresses, Benchmarking, and Open Problems* (2026),
  arXiv:2601.17883.

### Model provenance must bind bytes, not only names

A model ID and revision string are not enough for archival reproducibility. Promotion-grade
foundation-model lanes should eventually bind:

- upstream repository commit;
- checkpoint SHA-256;
- model/config SHA-256;
- preprocessing/config SHA-256;
- exact layer and token/pooling extraction;
- input physical units and normalization;
- channel order/mapping;
- resampling/filter/crop/window policy;
- transfer regime and adaptation authority.

### Cross-dataset inference needs another hierarchical level

Participants from several datasets should not simply be pooled as exchangeable rows. Once
Kumar2024, Ma2020, Wang2026, and an external task family are present, BMRB needs a
study/dataset-level replication or meta-analytic layer. Dataset/task should become a higher
level of evidence, with heterogeneity made visible.

### Multiplicity must be explicit

Layer sweeps, model families, tasks, calibration budgets, metrics, and mechanism candidates
create researcher degrees of freedom. A paper needs either:

- a small preregistered primary family;
- hierarchical testing;
- or explicit multiplicity correction for confirmatory secondary tests.

Exploratory matrices can remain broad if clearly labelled exploratory.

## First empirical paper program

A defensible sequence is:

### Stage A: benchmark validation

Run known-positive and known-null simulations before real EEG promotion.

### Stage B: Kumar2024 mechanism study

Use exact neurOS authority and compare:

- raw time-by-channel EEG;
- CSP/Riemannian or another strong classical representation;
- EEGNet / EEG-Conformer learned specialist representations;
- LaBraM frozen and at least one adapted transfer regime;
- EEGPT frozen and at least one adapted transfer regime;
- random/untrained controls.

The current density mechanism is expected to fail information novelty because it is
normalized-covariance equivalent. That negative result should be reported clearly.

### Stage C: non-equivalent candidate

Only after the equivalence baseline is understood should the study promote a genuinely
non-equivalent operator, contextual, dynamical, or intervention-predictive mechanism.

### Stage D: independent replication

Replicate across Ma2020, Wang2026, then one task family outside the original motor-imagery
setting.

## Candidate paper thesis

A strong methods paper could test the thesis:

> Predictive performance and representational similarity are insufficient evidence that
> two neural/model systems instantiate the same mechanism; adversarial equivalence,
> participant-level replication, and perturbational dependence provide distinct and more
> falsifiable evidence.

Example hypotheses:

1. predictive accuracy does not imply mechanism conservation;
2. some attractive mechanism signatures collapse under exact classical equivalence;
3. representation geometry and functional dependence can disagree;
4. mechanism/ablation conservation is more diagnostic than geometry alone;
5. foundation-model families can share task-relevant information while relying on
   different representational subspaces.

## Publication tracks

### Scientific / methods manuscript

The immediate priority should be the BMRB methodology plus benchmark validation and real
EEG evidence. Appropriate venues depend on empirical breadth and maturity; the method
should be judged on falsifiability and community utility, not on the word "quantum".

### Research-software manuscript

A software venue such as JOSS becomes more compelling after:

- sustained public development history;
- at least one real research result produced with the package;
- evidence that someone other than the author can install and use it;
- stable public API/schema documentation;
- release tags/archival DOI;
- contribution/adoption history.

The software paper should complement, not substitute for, a scientific validation paper.

## Claim-language guide

Prefer:

- "candidate mechanism" before causal evidence;
- "representation-compatible" or "conserved effect" before functional evidence;
- "quantum-inspired parameterization" when mathematics is quantum-derived but biology is
  not independently quantum-witnessed;
- "classical-equivalent" whenever exact equivalence exists;
- "characterized" when evidence exists but no confirmatory authority is present.

Avoid:

- "the brain uses X" from decoding/probing alone;
- "quantum information in EEG" from density/covariance notation;
- "causal mechanism" from representation ablation alone;
- "preregistered PASS" without an externally inspectable preregistration record.

## Definition of publishable readiness

QuantumBCI should be considered ready for a serious methods submission only when all are
true:

- BMRB has known-ground-truth positive/null validation;
- the primary real study is preregistered externally;
- primary control/budget/model/layer/pooling choices are frozen before final evaluation;
- participant-level effect sizes and uncertainty are reported;
- strong specialist and foundation-model baselines are present;
- at least one non-equivalent candidate mechanism is tested;
- results replicate across more than one dataset or task authority;
- every released result can be reconstructed from content-addressed evidence objects;
- package installation, examples, tests, and release artifacts pass from a clean environment;
- scientific and physical-quantum claim ceilings remain explicit.
