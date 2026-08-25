# E002 v0.14: trajectory-block stability evidence

QuantumBCI v0.14 moves E002 beyond the question “which model predicts better on one frozen split?” and asks a harder question:

> **Does the proposed mechanism survive plausible perturbations of the source evidence while final evaluation remains untouched?**

This is a necessary step toward mechanism research, but it is not sufficient evidence for a biological or physical-quantum mechanism.

## Why stability belongs after predictive adversaries

The E002 predictive ladder is intentionally adversarial:

```text
canonical Lindblad-style family
        ↕ exact affine equivalence audit
unconstrained affine generator
        ↓
observed-state persistence / AR / full VAR
        ↓
probabilistic identity-observation Kalman control
        ↓
two-regime switching-state control
        ↓
calibrated nonlinear RFF residual control
        ↓
trajectory-block bootstrap stability
        ↓
intervention-direction evidence
```

A mechanism should not be promoted merely because it fits one dataset partition. After the strongest planned predictive classical controls are present, v0.14 measures whether the fitted quantities are stable to source-data resampling.

**Stability is a falsifier, not a certificate of ontology.**

If a parameter changes sign, explodes in interval width, or appears only under a narrow model selection outcome, that instability is evidence against a strong mechanistic interpretation. If it remains stable, the next question is still causal: does an intervention move the system in the direction the mechanism predicts?

## Bootstrap unit: trajectories, not rows

Time samples are not exchangeable observations. Row-wise bootstrap would destroy temporal dependence and can manufacture falsely narrow uncertainty.

v0.14 therefore uses:

```text
role_stratified_trajectory_block_bootstrap_v1
```

For each replicate:

1. source-fit trajectory blocks are sampled with replacement;
2. calibration trajectory blocks are sampled independently with replacement;
3. all windows belonging to a sampled role-local trajectory block are copied together;
4. final-evaluation trajectory values are copied exactly once;
5. the complete fitted E002 source model is rebuilt;
6. final evaluation is scored read-only.

The bootstrap does **not** resample evaluation trajectories. It asks how source uncertainty changes the model while keeping the scientific test set fixed.

## Evidence roles remain distinct

The bootstrap never collapses fit, calibration, and evaluation into one pool.

- **Fit:** representation/model fitting only.
- **Calibration:** nonlinear complexity selection only.
- **Evaluation:** fixed read-only scoring only.

Fit and calibration blocks are sampled independently. New bootstrap-local trajectory IDs prevent accidental cross-role adjacency.

## What is recomputed

Every successful replicate reconstructs:

- the unconstrained continuous affine generator;
- the four-parameter canonical E002 generator;
- `omega_x`;
- `omega_z`;
- `gamma_dephasing`;
- `gamma_relaxation`;
- canonical structure residual relative to the unconstrained affine fit;
- canonical-vs-affine held-out one-step and rollout errors;
- the discrete full-VAR affine mean;
- fit-derived direct-Gaussian innovation variance;
- the calibrated v0.13 nonlinear residual model;
- direct-Gaussian-vs-nonlinear held-out NLL and RMSE gains;
- nonlinear feature-count / length-scale / ridge selection.

The purpose is to measure both **parameter stability** and **model-choice stability**.

## What is reported

For scalar quantities v0.14 reports:

- original point estimate;
- bootstrap mean;
- bootstrap median;
- bootstrap standard deviation;
- 2.5th / 97.5th percentile interval;
- finite-replicate fraction;
- sign consistency relative to the point estimate;
- positive fraction;
- interval width relative to the point estimate when defined;
- whether the percentile interval excludes zero.

For nonlinear model selection it reports:

- modal configuration;
- modal frequency;
- number of unique selected configurations;
- complete configuration-frequency ledger.

Every requested bootstrap replicate remains in the artifact. Failed replicates retain their failure reason and remain in the denominator.

## Execution, bootstrap coverage, and stability are different

A valid v0.14 artifact distinguishes three ideas:

```text
status = pass
execution_complete = true
bootstrap_coverage_sufficient = true | false
stability_gate_defined = false
stability_gate_pass = null
```

`status = pass` means the evidence transaction executed correctly.

`bootstrap_coverage_sufficient` means enough requested resamples produced estimable fits to support the reported bootstrap summaries. The research manifest currently requires a successful-replicate fraction of at least `0.90`.

That **is not a mechanism-stability threshold**. A model can converge on every bootstrap draw while its parameters wander wildly, change sign, or select different nonlinear configurations. v0.14 therefore reports the underlying intervals and frequencies rather than collapsing them into a universal binary “stable” verdict.

A future benchmark may preregister a stability criterion for a particular dataset/task/mechanism after appropriate power and calibration work. Until then:

```text
stability_gate_defined = false
```

Failure frequency itself is still evidence. It must never be hidden by dropping inconvenient bootstrap draws.

## A single-case bootstrap is not ICC

v0.14 explicitly serializes:

```text
single_case_bootstrap_is_icc = false
participant_icc_computed = false
```

An intraclass correlation coefficient requires a repeated-case structure in which between-case and within-case variation are estimable. Repeated bootstrap fits of one case do not create independent participants.

QuantumBCI reserves ICC for a later repeated participant/case reliability layer.

This distinction matters because a confident-looking number with the wrong statistical meaning is worse than no number at all.

## Upstream evidence is reconstructed

The promotion-grade task consumes:

```text
trajectory_contract.json
trajectory_index.json
matched_dynamics.json
classical_controls.json
probabilistic_ssm.json
switching_state.json
nonlinear_control.json
```

Before bootstrap fitting begins, v0.14 independently rebuilds the expected v0.13 nonlinear artifact from the same frozen descriptor and v0.8-v0.12 evidence.

A hand-edited `nonlinear_control.json` cannot become stability authority merely by retaining plausible hashes.

## Reproducibility identity

Research defaults:

```text
method:                   role_stratified_trajectory_block_bootstrap_v1
seed:                     1401
replicates:               200
minimum success fraction: 0.90
percentile interval:      [2.5, 97.5]
```

Each fit/calibration source draw is represented in the ledger by a SHA-256 digest.

CI uses fewer replicates for runtime reasons but exercises the same implementation and authority rules.

## Interpreting a stable canonical parameter

Suppose `gamma_dephasing` has a narrow positive bootstrap interval.

The valid conclusion is approximately:

> Under this declared representation, evidence authority, canonical family, estimator, and source-resampling procedure, the fitted damping coordinate is stable.

The invalid conclusion is:

> Neural tissue contains a measured microscopic quantum dephasing process with this rate.

The latter requires an independent physical substrate, operational witness, detection floor, discriminating perturbation, and strongest classical mimic.

## What “mechanism necessity” should eventually mean

QuantumBCI should use *necessity* operationally rather than rhetorically.

A mechanism can be called necessary **under a benchmark** only if all of the following survive:

1. **Predictive sufficiency:** it improves held-out behavior or provides a meaningful complexity advantage.
2. **Matched classical adversaries:** the gain survives affine, observed-state, probabilistic, switching-state, and flexible nonlinear controls with matched information sets.
3. **Source stability:** the effect survives trajectory-block source resampling.
4. **Repeated-case reliability:** the effect recurs across independent participants/cases when the dataset supports that analysis.
5. **Intervention fidelity:** interventions on the proposed mechanism predict the direction and magnitude of held-out changes.
6. **Ablation necessity:** removing the proposed mechanism destroys the relevant held-out computation while matched alternatives cannot recover it.
7. **Claim ceiling:** physical-quantum language remains prohibited unless an independent operational physical witness exists.

This is the path from an interesting mathematical representation to a defensible statement about recapitulating selected signatures of neural computation.

## Why this matters for QuantumBCI + neurOS

The intended architecture is:

```text
neurOS
  frozen neural/foundation-model evidence authority
        ↓
QuantumBCI
  representation + dynamics + falsification ladder
        ↓
neuros-mechint
  intervention/evidence contracts
        ↓
QuantumBCI evidence ledger
  prediction + stability + causal fidelity + claim ceiling
```

neurOS answers “what neural evidence and pretrained representation are we evaluating?”

QuantumBCI answers “which mathematical mechanism is actually needed after strong alternatives are ruled out?”

That division is useful whether the winning mechanism is classical, quantum-inspired, or eventually a genuinely physical quantum mechanism.
