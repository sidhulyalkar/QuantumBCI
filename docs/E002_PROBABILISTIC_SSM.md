# E002 probabilistic state-space control

v0.11 adds the first Kalman-family E002 control that is genuinely distinct from the observed-state linear controls introduced in v0.10.

The release is deliberately narrow. It does **not** fit an arbitrary hidden-state model.

```text
frozen v0.10 full-VAR mean
          |
          +-- direct Gaussian VAR baseline
          |
          +-- same-coordinate latent state
                H = I
                diagonal Q
                diagonal R
                Kalman filtering
```

The point of the stage is to ask whether classical filtering and uncertainty decomposition improve predictive density or sequential predictive means while holding the mean dynamics fixed.

## Why v0.10 did not count Kalman as another control

For a fully observed state with an identity observation matrix and deterministic point-prediction scoring, the one-step Kalman forecast mean has no new mean-model class beyond the fitted linear transition.

Calling all of these independent controls would double-count one mathematical object:

- affine VAR(1) with intercept;
- direct discrete affine transition regression;
- fully observed discrete LDS mean with identity observation.

v0.11 makes the Kalman lane distinct by adding an explicit probabilistic contract:

```text
latent clean state:    z[t+1] = F z[t] + c + w[t]
observation:           x[t]   = I z[t]     + v[t]
w[t] ~ N(0, Q)
v[t] ~ N(0, R)
```

The observation matrix is fixed to identity and the hidden state uses the same three coordinates as the observed state.

That gauge choice matters. An unconstrained latent LDS admits similarity transformations of the hidden coordinate system. v0.11 does not pretend those arbitrary latent coordinates are identifiable neural mechanisms.

## The mean dynamics are frozen

The v0.11 evidence task does not refit `F` or `c`.

It consumes the exact `controls.full_var1` record from the v0.10 artifact, independently recomputes the v0.10 control ladder from the frozen trajectory tensor, and requires the supplied full-VAR lane to match that reconstruction exactly.

This blocks a subtle provenance failure where a user could edit the serialized transition matrix while leaving its surrounding authority hashes unchanged.

The artifact records:

```text
mean_transition_source = v0.10:controls.full_var1
mean_transition_refit   = false
mean_model_sha256       = <content hash of exact F,c>
```

Any predictive difference between the matched direct Gaussian VAR and Kalman lanes therefore comes from filtering and uncertainty modeling, not a different mean transition.

## Evidence roles

v0.11 uses all three `TrajectoryEvidenceAuthority` roles for different purposes.

### Fit

Fit evidence is used only to estimate a diagonal innovation scale around the already-frozen v0.10 mean transition.

For each coordinate,

```text
base_variance[j] = mean((x[t+1,j] - F x[t] - c[j])^2)
```

with a small state-scale-relative numerical floor.

### Calibration

Calibration evidence chooses only two positive scalar multipliers:

```text
Q = q_scale * diag(base_variance)
R = r_scale * diag(base_variance)
```

The preregistered v0.11 grid is:

```text
q_scale in {0.01, 0.03, 0.1, 0.3, 1.0, 3.0}
r_scale in {0.01, 0.03, 0.1, 0.3, 1.0, 3.0}
```

The primary calibration objective is:

```text
sequential_predictive_gaussian_nll_v1
```

The lowest calibration mean Gaussian negative log likelihood wins. Deterministic tie-breaking prefers smaller `q_scale + r_scale`, then smaller `q_scale`, then smaller `r_scale`.

### Final evaluation

Final evaluation is read-only.

It cannot:

- refit the transition;
- refit the intercept;
- alter the base innovation variance;
- choose Q/R scales;
- change the observation model;
- initialize hidden state from fit or calibration history.

An adversarial test changes only evaluation observations and requires the selected Q/R scales and complete calibration candidate ledger to remain unchanged.

## Role-local filter reset

A filter state is evidence.

Carrying the posterior state or covariance from calibration into final evaluation would leak calibration observations into the test sequence.

v0.11 therefore resets the hidden state and covariance for every disconnected role-local chain.

The first observation of each chain is context only. It initializes that role-local trajectory and is not itself scored as a prediction.

## Sequential versus open-loop prediction

These two metrics answer different questions and are never merged.

### Sequential filtered prediction

At every time point the model may condition on observations already revealed earlier in the **same role-local chain**.

This measures online filtering quality.

### Open-loop prediction

After the first role-local observation, the model receives no later measurements. It recursively propagates its own latent predictive state and covariance.

This measures autonomous multi-step predictive behavior.

A Kalman model may improve sequential prediction while offering no open-loop mean advantage. That is a scientifically meaningful outcome, not a contradiction.

## Proper probabilistic scoring

For predictive mean `mu` and covariance `Sigma`, v0.11 reports multivariate Gaussian negative log likelihood:

```text
NLL = 0.5 * [
    d log(2 pi)
    + log det(Sigma)
    + (x - mu)^T Sigma^-1 (x - mu)
]
```

The implementation uses `slogdet` and linear solves rather than explicit matrix inversion.

It also reports:

- total and mean predictive NLL;
- predictive-mean RMSE;
- predictive-mean MAE;
- mean squared Mahalanobis distance;
- mean predictive log-determinant;
- marginal 95% coverage;
- Bloch half-L2 mean error for three-dimensional states;
- predictive-mean physicality fraction;
- qubit trace distance only for physically valid Bloch-vector pairs.

## Matched direct Gaussian baseline

The direct Gaussian VAR baseline uses the exact same frozen `F,c`.

Its diagonal innovation variance is estimated from fit evidence only. Sequential prediction conditions directly on the latest observed state; open-loop prediction recursively propagates the state and innovation covariance.

This comparison keeps the mean model fixed:

```text
same F,c
same trajectory authority
same final evaluation
same Gaussian scoring semantics

Direct Gaussian VAR  vs  identity-observation Kalman
```

A lower Kalman NLL is therefore evidence that the latent-noise/filtering model gives a better classical uncertainty decomposition under this restricted contract.

## Physicality semantics

The predictive mean is still a three-dimensional coordinate prediction.

`0.5 * ||r_pred - r_true||` is always available as a Bloch-coordinate half-L2 metric. It is called qubit trace distance only when both vectors lie within the physical Bloch ball up to the declared numerical tolerance.

An unphysical Kalman predictive mean is never converted into quantum terminology merely because it has three coordinates.

## What Q and R do not mean

The selected process and measurement variances are statistical model parameters.

They are **not** evidence for:

- neuronal decoherence;
- microscopic dissipation;
- quantum measurement noise;
- biological process noise as a physically isolated source;
- sensor noise as an independently measured hardware property.

Those interpretations require independent operational measurements.

## Evidence-producing command

After the v0.8, v0.9, and v0.10 artifacts exist:

```bash
python -m quantumbci.experiments.probabilistic_ssm_task \
  --descriptor trajectory_contract.json \
  --trajectory-index trajectory_index.json \
  --matched matched_dynamics.json \
  --classical-controls classical_controls.json \
  --output probabilistic_ssm.json
```

The task independently verifies all upstream authority and model identities before writing output.

## Public Python surface

The root package exports safe lower-level primitives for analysis:

```python
from quantumbci import (
    PredictiveDensityMetrics,
    ProbabilisticStateSpaceResult,
    fit_base_innovation_variance,
    score_direct_gaussian_var,
    score_identity_observation_kalman,
)
```

The configurable Q/R grid-search runner intentionally remains module-level in v0.11. Promotion-grade runs use the preregistered experiment task above.

## What remains open after v0.11

Passing v0.11 does not unlock mechanistic intervention claims.

Still required:

1. a switching-state dynamics control;
2. a flexible nonlinear dynamics control when sample size independently supports it;
3. participant/recording-level bootstrap stability;
4. intervention-direction consistency;
5. independent physical witnesses for any physical-quantum interpretation.

A future arbitrary latent-state model with a learned observation matrix or reduced hidden dimension is also a separate research problem. It needs new representation-fit authority and latent-gauge/identifiability rules rather than inheriting the v0.11 interpretation.

## Interpretation ceiling

A useful v0.11 result is one of these:

1. **Kalman improves sequential NLL and mean error.** Classical filtering/noise decomposition helps.
2. **Kalman improves NLL but not mean error.** The probabilistic uncertainty model helps without changing point prediction quality.
3. **Kalman does not beat direct Gaussian VAR.** The extra latent-noise structure is unnecessary under this evidence authority.
4. **Kalman helps sequentially but not open loop.** Observation-history filtering helps, but the autonomous mean dynamics are not improved.

None of those outcomes establishes microscopic quantum dynamics in neural tissue.
