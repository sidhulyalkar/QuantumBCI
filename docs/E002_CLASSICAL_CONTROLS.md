# E002 classical controls: count model classes, not model names

v0.10 extends E002 beyond the v0.9 continuous affine-versus-canonical baseline with a small observed-state classical prediction ladder:

```text
persistence                      0 fitted parameters
independent AR(1) + intercept    6 fitted parameters in 3D
full VAR(1) + intercept         12 fitted parameters in 3D
```

Every lane consumes the exact same `TrajectoryEvidenceAuthority`, state tensor, fit-transition graph, and final-evaluation graph as the v0.9 matched baseline.

The purpose is adversarial: if a straightforward classical predictor explains the held-out trajectory better, QuantumBCI should surface that result rather than protect the quantum-inspired parameterization.

## Evidence chain

The v0.10 stage requires all three earlier artifacts:

```text
trajectory_contract.json
        |
        +--> trajectory_index.json      v0.8 temporal authority
        |
        +--> matched_dynamics.json      v0.9 affine/canonical baseline
                         |
                         v
                 classical_controls.json
```

Before fitting, the control task reconstructs the descriptor authority and requires:

- exact equality with the serialized v0.8 authority;
- the same state SHA-256;
- the same authority fingerprint;
- the same fit-transition SHA-256;
- the same evaluation-transition SHA-256;
- the v0.9 baseline to have passed same-evidence verification;
- the v0.9 authoritative ridge to be zero;
- the v0.9 artifact to preserve its nonphysical-claim ceiling.

Any mismatch blocks the control artifact before it is written.

Run the stage with:

```bash
python -m quantumbci.experiments.classical_controls_task \
  --descriptor trajectory_contract.json \
  --trajectory-index trajectory_index.json \
  --matched matched_dynamics.json \
  --output classical_controls.json
```

## 1. Persistence

The zero-fit baseline is

```text
x[t+1] = x[t].
```

Persistence is intentionally simple. Neural trajectories can be highly autocorrelated, and a mechanism model should not receive credit merely for predicting that a short-step signal remains near its previous state.

## 2. Diagonal AR(1)

Each coordinate is fitted independently:

```text
x_j[t+1] = a_j x_j[t] + c_j.
```

In three dimensions this has six coefficients. It can capture coordinate-specific decay and drift but cannot use cross-coordinate coupling.

This is a useful middle control between persistence and a fully coupled linear transition.

## 3. Full affine VAR(1)

The strongest v0.10 observed-state linear control is

```text
x[t+1] = F x[t] + c.
```

In three dimensions, `F` contributes nine coefficients and `c` contributes three, for 12 total parameters.

It is fitted directly to the held-out prediction target rather than through a continuous derivative approximation.

### One model class, several familiar names

Under the current contract, the following descriptions refer to the same forecast-mean class:

- full affine VAR(1) with intercept;
- direct discrete affine transition regression;
- fully observed one-step discrete LDS mean with identity observation.

QuantumBCI records these as aliases of one model. They are not three independent classical controls.

This matters because a benchmark can look much stronger than it is if the same mathematical model is counted repeatedly under different disciplinary vocabulary.

## Why there is not a separate Kalman score yet

A Kalman model becomes meaningfully different when the scientific contract distinguishes a latent state from noisy observations and evaluates a probabilistic model of that uncertainty.

With the current frozen tensor treated as the fully observed state, identity observation, and deterministic point-prediction metrics, the forecast mean is still driven by the same fitted linear transition. Adding process/measurement covariance symbols without a latent observation contract would mostly add notation, not a new predictive adversary.

Therefore v0.10 explicitly records:

```text
kalman_forecast_mean_distinct_under_current_contract = false
```

A future probabilistic state-space release should instead define:

- latent versus observed variables;
- observation matrix semantics;
- process-noise covariance fitting authority;
- measurement-noise covariance fitting authority;
- calibration-only hyperparameter selection;
- held-out log likelihood or proper scoring rule;
- filtering versus forecasting information sets;
- missing-observation behavior.

Only then should “Kalman” count as a distinct control.

## Why direct VAR can beat a canonical-generated trajectory

This is an important expected result rather than a bug.

The v0.9 canonical baseline uses

```text
(x[t+dt] - x[t]) / dt
```

as a forward-difference estimate of the continuous vector field, fits the four canonical parameters to that derivative target, and then scores predictions through RK4.

v0.10 VAR(1) instead fits the fixed-step map

```text
x[t] -> x[t+dt]
```

directly.

For a linear time-homogeneous continuous system, the exact fixed-step flow itself is an affine discrete map. On noiseless data with enough state coverage, full VAR(1) can therefore recover that map essentially exactly even when the generating dynamics were canonical Lindblad-style dynamics.

If VAR(1) beats the forward-difference canonical baseline, the correct interpretation is:

> the direct discrete classical estimator is better matched to this prediction objective.

It is not evidence that the generating canonical family was absent, and it is certainly not evidence for quantum novelty. It is evidence that estimator choice matters.

## Same-complexity classical control

The four-parameter canonical Lindblad family already compiles exactly to a four-parameter classical affine Bloch generator. That exact classical representation is not a different model. Relabeling it as a second four-parameter control would duplicate the same hypothesis class.

This is why the key question is not “quantum model versus a separately named classical copy.” The useful questions are:

- does the four-parameter constraint predict well relative to more flexible controls?
- is the constraint stable across independent cases?
- does it generalize under intervention?
- does it reduce calibration cost?
- do its observables survive stronger non-equivalent classical models?

## Adversarial qualification

The v0.10 unit contract includes a stable cross-coupled discrete affine system. Full VAR(1) must:

- recover the transition matrix and intercept to numerical precision;
- achieve essentially zero held-out one-step error;
- achieve essentially zero held-out rollout error;
- outperform persistence;
- outperform diagonal AR(1).

This test is deliberately easy for the correct classical model. A classical control ladder that cannot win its own home game is not a credible adversary.

## What remains missing

The observed-state linear ladder is now materially stronger, but E002 is still not ready for mechanistic intervention promotion.

The next distinct controls are:

1. **probabilistic latent state-space model** with a real observation/noise contract and proper scoring;
2. **switching-state dynamics** for regime changes and nonstationarity;
3. **flexible nonlinear dynamics** when the number of independent trajectories is sufficient to fit and evaluate them without leakage.

A structured damped oscillator need not be counted as an independent strong predictive class if a full VAR already spans its fixed-step linear mean dynamics. It may still be valuable later as a lower-complexity interpretability/parsimony control, but it should be introduced for that explicit purpose rather than to inflate the number of classical baselines.

## Interpretation ceiling

A v0.10 result can support statements such as:

- the canonical constraint is more or less predictive than direct discrete linear dynamics;
- cross-coordinate coupling matters relative to diagonal AR;
- persistence is or is not a competitive short-horizon baseline;
- the quantum-inspired coordinate system is compact but not information-distinct.

It cannot support microscopic quantum biology.

Physical claims still require an independently specified substrate, operational witness, discriminating perturbation, detection floor, and strongest classical mimic.
