# E002 v0.13: flexible nonlinear classical control

QuantumBCI v0.13 closes the planned E002 predictive-adversary ladder with a flexible nonlinear classical model.

The control asks:

> After matched affine, observed-state, probabilistic and switching-state controls have been considered, is there still ordinary nonlinear predictive structure that explains the trajectory?

The answer can only weaken a quantum-inspired uniqueness claim. The model itself is classical.

## Model

The v0.13 mean model is

```text
x[t+1] = F x[t] + c + W phi(z[t])

z[t] = (x[t] - mu_fit) / sigma_fit
```

where:

- `F,c` are the exact frozen v0.10 full-VAR affine mean;
- `mu_fit,sigma_fit` are estimated on fit-state inputs only;
- `phi` is a deterministic random-Fourier-feature map;
- `W` is a fitted nonlinear residual readout.

The affine mean is not refit inside v0.13.

This makes attribution cleaner. A v0.13 gain is attributable to the nonlinear residual function and its calibrated complexity, not to quietly fitting a better linear baseline.

## Random Fourier features are classical

Random Fourier features approximate a stationary kernel feature map using ordinary classical random projections and cosine features.

They are unrelated to a quantum Fourier transform.

The v0.13 implementation uses one fixed seed:

```text
RFF seed = 1301
```

The seed is preregistered software configuration and is not tuned on calibration or evaluation evidence.

## Candidate grid

The calibrated grid is:

```text
feature count:             16, 32, 64
RFF length-scale multiple: 0.5, 1.0, 2.0
ridge:                     1e-4, 1e-2, 1.0
```

For one length scale, feature banks are nested by feature count. The 16-feature candidate uses a prefix of the 32/64-feature bank rather than drawing unrelated random features.

Each residual weight matrix has:

```text
feature_count x state_dimension
```

fitted coefficients.

## Sample-size authority

Candidate complexity is restricted before calibration selection.

A candidate is eligible only when:

```text
n_fit_transitions >= 4 * feature_count
```

This prevents final evaluation from deciding that a large model was retrospectively "safe enough."

If no candidate satisfies the rule, the nonlinear stage fails closed.

## Fit / calibration / evaluation roles

### Fit

Fit transitions determine:

- state standardization;
- RFF residual weights;
- diagonal residual innovation variance for every candidate.

### Calibration

Calibration transitions choose among already-fit candidates using:

```text
one-step Gaussian predictive NLL
```

The deterministic tie break is:

1. lower calibration NLL;
2. fewer features;
3. lower length-scale multiplier;
4. lower ridge.

### Final evaluation

Evaluation is read-only.

It cannot change:

- standardization;
- RFF frequencies or phases;
- residual weights;
- residual variance;
- feature count;
- length scale;
- ridge;
- affine baseline.

An adversarial test modifies final-evaluation observations only and requires the selected model SHA and entire calibration-candidate ledger to remain unchanged.

## One-step probabilistic score

For a selected nonlinear mean `m(x_t)`, fit transitions estimate a diagonal residual variance.

The held-out one-step score is Gaussian predictive NLL:

```text
p(x[t+1] | x[t]) = Normal(m(x[t]), diag(v_fit))
```

The implementation also reports:

- one-step RMSE;
- one-step MAE;
- Bloch half-L2 coordinate error;
- predictive-mean physicality fraction;
- qubit trace distance only on valid physical Bloch-vector pairs.

## Deterministic mean rollout

The nonlinear mean can be iterated autonomously:

```text
x_hat[t+1] = m(x_hat[t])
```

v0.13 reports deterministic mean-rollout:

- RMSE;
- MAE;
- Bloch half-L2;
- physicality-safe metrics.

This is a mean-trajectory test, not a predictive-density rollout.

## Nonlinear uncertainty rollout remains locked

Propagating a Gaussian residual distribution through a nonlinear feature map does not generally remain Gaussian.

v0.13 does not quietly use a local linearization, unscented approximation, particle approximation or moment collapse and then call the result exact.

The artifact therefore records:

```text
nonlinear_uncertainty_rollout_complete = false
rollout_likelihood_promotion_eligible = false
```

A future probabilistic nonlinear rollout must declare its approximation and validation contract explicitly.

## Matched comparisons

### One-step likelihood

The matched probabilistic comparison is:

```text
direct Gaussian VAR one-step NLL
vs
nonlinear residual one-step NLL
```

Both condition on the observed current state `x[t]` and use fit-derived diagonal innovation variance.

### Mean rollout

The matched autonomous mean comparison is:

```text
full affine VAR mean rollout RMSE
vs
nonlinear residual mean rollout RMSE
```

Both start from the same role-local chain context and then iterate their own deterministic means.

### Deliberate exclusions

The task does not present Kalman or switching sequential likelihood differences as matched nonlinear comparisons.

Those models use additional history through filtered latent state or regime belief. Their sequential information sets differ from the direct nonlinear `p(x[t+1] | x[t])` score.

## Positive nonlinear adversary

The v0.13 sensitivity fixture uses a stable cross-coordinate sinusoidal residual:

```text
r_1 = 0.16 sin(4 x_2)
r_2 = 0.14 sin(4 x_3)
r_3 = 0.15 sin(4 x_1)
```

around a stable affine transition.

On held-out evaluation transitions the software gate is:

```text
direct Gaussian VAR mean NLL
- nonlinear mean NLL
> 0.25 nats / transition
```

This proves the nonlinear model can detect meaningful nonlinearity rather than merely existing as an unused API.

## Linear Gaussian null

A second adversary is generated by one stationary affine Gaussian process.

The apparent nonlinear gain must satisfy:

```text
direct Gaussian VAR mean NLL
- nonlinear mean NLL
< 0.08 nats / transition
```

Negative values are acceptable and mean the simpler affine model wins.

The nonlinear control therefore has to demonstrate both sensitivity and restraint.

The 0.25 and 0.08 values are implementation qualification thresholds, not biological effect-size thresholds.

## Evidence transaction

The promotion-grade task requires the complete prior E002 chain:

```text
trajectory_contract.json
 -> trajectory_index.json          v0.8
 -> matched_dynamics.json          v0.9
 -> classical_controls.json        v0.10
 -> probabilistic_ssm.json         v0.11
 -> switching_state.json           v0.12
 -> nonlinear_control.json         v0.13
```

Before nonlinear fitting, the task independently reconstructs the expected v0.12 switching artifact from the earlier evidence files.

The supplied `switching_state.json` must match that reconstruction exactly.

Only after that check does the task reuse the verified v0.10 `controls.full_var1` transition and intercept.

## CLI

```bash
python -m quantumbci.experiments.nonlinear_control_task \
  --descriptor trajectory_contract.json \
  --trajectory-index trajectory_index.json \
  --matched matched_dynamics.json \
  --classical-controls classical_controls.json \
  --probabilistic-ssm probabilistic_ssm.json \
  --switching-state switching_state.json \
  --output nonlinear_control.json
```

Tampered v0.12 evidence causes the task to fail before output creation.

## What completing v0.13 means

After v0.13, E002 has qualified the planned predictive adversaries:

1. exact classical affine equivalence;
2. matched continuous-time affine control;
3. persistence / diagonal AR / full discrete VAR;
4. calibrated probabilistic Kalman control;
5. two-regime switching-state control;
6. flexible nonlinear residual control.

That does **not** mean a surviving Lindblad-style model is promoted automatically.

The remaining scientific gates are qualitatively different:

- bootstrap parameter / stability evidence;
- intervention-direction consistency.

The intervention task remains fail-closed until those contracts are implemented.

## Interpretation ceiling

A v0.13 win means:

> A flexible classical nonlinear residual improves held-out prediction under the frozen temporal authority.

It does not establish:

- quantum novelty;
- microscopic biological dynamics;
- biological meaning for RFF features;
- physical decoherence;
- an identified quantum substrate.

A v0.13 loss is also useful. It says the richer nonlinear classical explanation was not needed under the tested authority and complexity grid.

Physical-quantum promotion still requires an independent substrate, operational witness, discriminating perturbation, detection floor, strongest classical mimic, and replication.
