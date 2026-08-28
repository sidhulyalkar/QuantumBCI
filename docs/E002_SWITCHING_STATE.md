# E002 v0.12: switching-state classical control

QuantumBCI v0.12 adds a two-regime Markov-switching affine VAR as a stronger classical adversary for E002.

The purpose is narrow: determine whether temporal behavior that appears interesting under a single stationary open-system parameterization is better explained by ordinary classical regime switching.

This is a **classical control**. Nothing in this model, its hidden regime probabilities, or its fitted transition matrices is physical-quantum evidence.

## Model

For regime `s_t in {0, 1}`:

```text
x[t+1] = F[s_t] x[t] + c[s_t] + epsilon[s_t]

epsilon[s_t] ~ Normal(0, diag(v[s_t]))

s[t+1] ~ Categorical(P[s_t, :])
```

Each regime has:

- a 3 x 3 affine transition matrix `F`;
- a 3-vector intercept `c`;
- a diagonal 3-vector innovation variance.

The Markov layer has a two-state transition matrix plus an initial regime distribution.

In three dimensions the nominal parameter count is 33:

```text
2 * (9 transition + 3 intercept + 3 variance) = 30
2 free Markov transition probabilities              =  2
1 free initial-regime probability                   =  1
---------------------------------------------------------
total                                                = 33
```

That count is descriptive. Ordinary regular-model AIC/BIC asymptotics are not treated as authoritative here because finite-mixture / hidden-state likelihoods have singular and boundary behavior that violates the simplest regularity assumptions.

## Why this is distinct from v0.10 and v0.11

v0.10 asks whether one observed-state affine transition explains the fixed-step trajectory.

v0.11 freezes that same transition and asks whether a classical same-coordinate latent-noise / filtering model improves predictive density.

v0.12 asks a different question:

> Does the trajectory require more than one ordinary classical affine dynamical regime?

Unlike v0.11, v0.12 fits regime-specific means. For that reason its extra flexibility must be attacked with both positive and null synthetic controls.

## Evidence authority

The switching model may fit only source-fit transitions supplied by `TrajectoryEvidenceAuthority`.

It may not fit or select parameters from:

- calibration transitions;
- final evaluation transitions;
- a floating tensor not bound to the authority;
- a v0.11 artifact that cannot be independently reconstructed.

The promotion-grade task first reconstructs the complete v0.11 artifact from the descriptor plus v0.8-v0.10 evidence files. The supplied `probabilistic_ssm.json` must equal that reconstruction before switching fitting begins.

The v0.12 artifact binds:

- state-data SHA-256;
- trajectory authority fingerprint;
- fit-transition SHA-256;
- calibration-transition SHA-256;
- evaluation-transition SHA-256;
- exact upstream artifact filenames;
- deterministic multi-start diagnostics.

## Deterministic multi-start EM

The dependency-light implementation uses deterministic EM with four initializations:

1. residual principal-component median split;
2. state principal-component median split;
3. state-difference principal-component median split;
4. temporal-half split within each fit trajectory chain.

Each initialization is run independently.

The evidence artifact records every start as either:

- `success`, with likelihood, iteration count, convergence flag, and label-canonicalization permutation; or
- `failure`, with the failure message.

The selected model must be the successful initialization with highest fit log likelihood, with initialization ID as a deterministic tie break.

Failed starts are not deleted from the scientific ledger.

## Regime labels are exchangeable

A two-state hidden model has a label-permutation symmetry.

If we exchange regimes 0 and 1 consistently across:

- transition matrices;
- intercepts;
- innovation variances;
- Markov transition rows/columns;
- initial probabilities;

then the model defines the same predictive distribution.

QuantumBCI therefore canonicalizes labels only for reproducible serialization using a deterministic lexicographic parameter ordering.

This does **not** make labels mechanistically identifiable.

Do not write interpretations such as:

- "regime 0 is an excitatory state";
- "regime 1 is a decoherent state";
- "the model discovered two biological modes";

without an independent operational measurement that identifies those states outside the hidden-model fit.

## Sequential predictive density

v0.12 qualifies one information set:

```text
current observed x[t]
+ regime belief updated from earlier observations
  inside the same held-out evidence-role trajectory chain
```

At the beginning of every disconnected chain, regime belief resets to the fitted initial distribution.

For each transition, the predictive distribution is the two-component Gaussian mixture:

```text
p(x[t+1] | history) = sum_s p(s_t=s | history)
                       Normal(x[t+1]; F[s] x[t] + c[s], diag(v[s]))
```

The primary score is proper sequential mixture negative log likelihood.

The implementation also reports:

- predictive-mean RMSE;
- predictive-mean MAE;
- mean predictive regime entropy;
- mean maximum predictive regime probability;
- Bloch half-L2 coordinate error;
- predictive-mean physicality fraction;
- qubit trace distance only on physically valid Bloch-vector pairs.

## Why open-loop switching is still locked

An exact multi-step switching forecast becomes a growing mixture over regime and state histories.

A naive implementation can easily hide an approximation by collapsing that mixture to one Gaussian or one expected regime at every step. That would mix approximation error with model quality.

v0.12 therefore records:

```text
exact_open_loop_switching_forecast_complete = false
open_loop_promotion_eligible = false
```

Sequential switching scores must not be compared against v0.10 or v0.11 open-loop scores as if they used the same information.

A future open-loop switching implementation must declare its mixture propagation or approximation contract explicitly.

## Positive synthetic adversary

The first software falsifier contains two persistent hidden regimes with distinct stable cross-coupled affine transitions and low diagonal Gaussian innovations.

Under the preregistered fixture:

```text
direct Gaussian VAR evaluation mean NLL
- switching evaluation mean NLL
> 0.12 nats / transition
```

The switching model must show a material held-out likelihood advantage.

This tests whether the implementation can detect genuine regime structure.

## One-regime null adversary

A second fixture is generated by one stationary affine Gaussian process.

The same quantity must satisfy:

```text
direct Gaussian VAR evaluation mean NLL
- switching evaluation mean NLL
< 0.10 nats / transition
```

A negative value is allowed and means the simpler direct Gaussian VAR wins.

This is deliberately not a requirement that the switching model always lose. It is a requirement that extra hidden-state flexibility not manufacture a large held-out advantage on a stationary null.

The 0.12 / 0.10 values are **software qualification thresholds**, not biological effect-size thresholds.

## Evaluation is read-only

A dedicated adversary changes only final evaluation observations.

The fitted switching model must remain unchanged:

- selected initialization;
- fit likelihood;
- regime transitions;
- affine transition matrices;
- intercepts;
- innovation variances;
- initial regime probabilities.

Evaluation scores are expected to change. Fit parameters are not.

## Comparison to v0.11

The v0.12 task reports matched sequential comparisons against the two v0.11 evaluation lanes that consume the compatible information set:

- direct Gaussian VAR sequential NLL;
- identity-observation Kalman sequential NLL.

It reports:

```text
direct Gaussian VAR NLL - switching NLL
Kalman NLL              - switching NLL

direct Gaussian VAR RMSE - switching RMSE
Kalman RMSE               - switching RMSE
```

Positive differences favor switching.

These comparisons are descriptive classical-model comparisons. They do not establish quantum novelty.

## CLI stage

After v0.8-v0.11 evidence files exist:

```bash
python -m quantumbci.experiments.switching_state_task \
  --descriptor trajectory_contract.json \
  --trajectory-index trajectory_index.json \
  --matched matched_dynamics.json \
  --classical-controls classical_controls.json \
  --probabilistic-ssm probabilistic_ssm.json \
  --output switching_state.json
```

The task fails before output creation when the supplied v0.11 artifact differs from an independent reconstruction.

## What v0.12 does not unlock

A passing switching-state control is still not enough for mechanistic promotion.

The E002 ladder still requires:

1. a flexible nonlinear classical control when sample size supports it;
2. bootstrap parameter / stability evidence;
3. intervention-direction evidence;
4. an explicit open-loop switching contract if autonomous switching forecasts are used in a claim.

The intervention task remains fail-closed.

## Interpretation ceiling

A switching-state win means:

> A classical two-regime dynamical model provides better sequential predictive density under the frozen temporal authority.

It does not mean:

- that the two regimes are biological states;
- that regime switching is the uniquely correct mechanism;
- that fitted innovation variance is physical decoherence;
- that hidden-state dynamics are quantum;
- that a microscopic substrate has been identified.

Physical-quantum claims remain gated by an independent substrate, operational witness, discriminating perturbation, detection floor, strongest classical mimic, and replication.
