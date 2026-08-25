# E002 matched dynamics baseline

v0.9 is the first E002 release allowed to fit a predictive dynamics model after the synthetic identifiability gate and trajectory-authority gate have passed.

It deliberately starts with only two lanes:

```text
one frozen TrajectoryEvidenceAuthority
             |
             +-- unconstrained affine generator: 12 parameters
             |
             +-- canonical Lindblad family:       4 parameters
```

Both lanes use the exact same state tensor, legal fit transitions, legal evaluation transitions, estimator convention, and scoring integrator.

## Why one matched transaction?

The previous manifest had separate future placeholders for Lindblad and classical fitting. That would permit accidental drift in preprocessing, transition extraction, split authority, or even estimator details.

v0.9 instead materializes both lanes from one call:

```bash
python -m quantumbci.experiments.tasks \
  fit-matched-dynamics E002 \
  --input trajectory_contract.json \
  --trajectory-index trajectory_index.json \
  --output matched_dynamics.json \
  --ridge 0
```

Before fitting, the task reconstructs the trajectory authority from the descriptor and requires the complete serialized authority to equal the previously materialized `trajectory_index.json` authority. A changed tensor, source revision, timing policy, role index, representation-fit subset, or transition count blocks the fit.

## Estimator contract

The first estimator is intentionally simple and explicit:

```text
FIT_ESTIMATOR_ID = forward_difference_least_squares_v1
SCORE_INTEGRATOR_ID = rk4_one_step_and_rollout_v1
```

For every legal fit transition `(x_t, x_{t+dt})`, v0.9 forms

```text
(dx/dt)_target = (x_{t+dt} - x_t) / dt.
```

The affine lane fits

```text
dx/dt = A x + b
```

with an unconstrained `3 x 3` matrix and three-dimensional offset: 12 parameters.

The canonical lane fits

```text
H = (omega_x sigma_x + omega_z sigma_z) / 2
```

plus nonnegative z-dephasing and amplitude-relaxation rates: 4 parameters.

The two damping constraints are solved by exact active-set enumeration. No optimizer dependency is required.

This forward-difference estimator is a baseline, not an exact continuous-time likelihood. Future exact-discretization, Kalman, or state-space estimators must carry a distinct estimator identity rather than being compared as though they used the same fitting convention.

## Numerically stable least squares

v0.9 does not form normal equations. Unregularized fits use `numpy.linalg.lstsq` directly. Positive ridge values are represented as augmented least-squares systems.

The default is `ridge = 0` so the first 12-parameter versus 4-parameter comparison is not quietly shaped by incomparable regularization geometry.

Each lane records its design rank as a basic identifiability diagnostic.

## Evaluation

Both lanes are scored with the same RK4 implementation on:

- one-step prediction RMSE;
- one-step MAE;
- recursive rollout RMSE;
- Bloch-coordinate half-L2 error when the state dimension is three;
- prediction physicality fraction;
- physical prediction/target pair fraction;
- qubit trace distance only on physically valid pairs.

### Trace distance is not a generic 3D error metric

For two physical qubit states with Bloch vectors `r` and `s`,

```text
D(rho(r), rho(s)) = ||r - s||_2 / 2.
```

But an unconstrained affine classical control can predict a vector outside the Bloch ball. That vector does not define a physical qubit density state.

Therefore QuantumBCI always calls `||r-s||/2` a **Bloch half-L2 coordinate metric** first. It is reported as qubit trace distance only for pairs where both vectors have norm at most `1 + tolerance`.

This distinction prevents an unphysical classical prediction from being laundered into quantum terminology.

## Adversarial qualification

The matched baseline is tested in both directions.

### Canonical survival case

Trajectories generated from the declared four-parameter family must:

- be recovered with small parameter error;
- obtain low held-out one-step and rollout error;
- remain close to the unconstrained affine fit;
- show a small whole-generator canonical-structure residual;
- preserve the expected eight-parameter reduction.

Near parity here means only that the constrained family can summarize canonical dynamics efficiently.

### Stable noncanonical affine adversary

A stable affine system containing anisotropic damping and forbidden cross-axis couplings must favor the unconstrained classical lane.

If the canonical family ties or beats this adversary without actually representing its generator structure, the benchmark is not discriminating enough and must fail qualification.

## What v0.9 does not claim

v0.9 does **not** complete the classical-control ladder. The following remain future fail-closed stages:

- conventional LDS/Kalman;
- VAR and damped-oscillator controls where appropriate;
- switching-state dynamics;
- flexible nonlinear controls;
- intervention testing;
- participant/recording bootstrap stability;
- empirical promotion reporting.

The first matched baseline is useful because it establishes the scoring spine on which those stronger controls can be added without changing evidence authority.

## Interpretation ceiling

Three outcomes are possible:

1. **Canonical substantially worse than affine.** The declared family is too restrictive for the trajectory and is not promoted.
2. **Canonical near affine with fewer parameters.** This may support a useful constrained regularizer or interpretable coordinate system.
3. **Canonical better on held-out prediction.** This is interesting predictive evidence for the constraint, but still not additional quantum information because the fully observed qubit dynamics are exactly affine-representable.

None of these outcomes is evidence for microscopic biological decoherence.

A physical quantum claim still requires an independently specified substrate, operational witness, discriminating perturbation, detection floor, and strongest classical mimic.
