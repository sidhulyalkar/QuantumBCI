# E002: Open-system dynamics without parameter mythology

E002 asks whether a low-dimensional Lindblad/GKSL-style parameterization can provide useful,
stable and intervention-predictive structure for neural latent trajectories after the strongest
classical equivalent is made explicit.

Its first result is mathematical:

> A time-homogeneous qubit Lindblad equation on the fully observed density state is exactly a
> three-dimensional classical affine linear ODE on the Bloch vector.

Its second boundary is evidential:

> A real dynamics comparison is not matched unless every model consumes the same content-addressed
> state tensor and the same frozen temporal transition authority.

## 1. Exact qubit compilation

Write a qubit state as

```text
rho = (I + r_x sigma_x + r_y sigma_y + r_z sigma_z) / 2.
```

For a linear Lindblad generator `L`,

```text
d r_i / dt = Tr[sigma_i L(rho)]
```

which gives the exact classical system

```text
d r / dt = A r + b

b_i  = Tr[sigma_i L(I)] / 2
A_ij = Tr[sigma_i L(sigma_j)] / 2.
```

`compile_qubit_lindblad_to_affine(...)` performs this compilation directly. The audit compares both
instantaneous derivatives and RK4 trajectories.

A fully observed qubit Lindblad trajectory is therefore **not dynamical-information novel** relative
to the compiled affine state-space representation.

Lindblad coordinates can still be useful as constraints, regularizers, compact coordinates, or an
intervention language. Those advantages must be demonstrated empirically against matched classical
controls.

## 2. Parameter gauges

Arbitrary Hamiltonian and collapse-matrix entries are not separately identifiable.

QuantumBCI witnesses standard generator-preserving transformations:

```text
H -> H + c I
C_j -> exp(i phi_j) C_j
C'_a = sum_j U_aj C_j
```

where `U` is unitary for the multi-channel collapse-basis transformation.

Therefore E002 may recover only:

- a declared canonical gauge;
- generator-level quantities;
- invariants that survive the transformations.

An optimizer-returned collapse matrix is not a uniquely identified neural mechanism.

## 3. Canonical synthetic family

The first qualification family fixes

```text
H = (omega_x sigma_x + omega_z sigma_z) / 2
```

with z-dephasing at `gamma_dephasing` and amplitude relaxation toward `|0>` at
`gamma_relaxation`.

The benchmark deliberately does **not** fit Lindblad matrices directly. It:

1. generates trajectories through the Lindblad implementation;
2. adds frozen moderate observation noise;
3. fits an unconstrained classical affine generator `A,b`;
4. recovers the four canonical coordinates from redundant entries of `A,b`;
5. reconstructs the complete canonical generator;
6. checks every entry of the fitted generator against that projection;
7. runs exact affine-equivalence and parameter-gauge witnesses.

This keeps canonical recovery downstream of a classical fit rather than making the synthetic test a
circular quantum-model optimizer demo.

## 4. Family specificity

Parameter projection and family identification are different questions:

```text
projection: "which canonical numbers resemble this affine generator?"
identification: "does the full generator actually lie near that canonical family?"
```

`canonical_structure_residual(...)` evaluates the second question across every entry of `A,b`.

The preregistered synthetic gate requires:

```text
max canonical-case structure residual <= 0.05
stable noncanonical affine adversary residual >= 0.10
```

The classical adversary contains plausible omega/gamma-like coordinates plus forbidden anisotropic
decay, cross-axis couplings and an affine offset. It is dynamically stable, so the gate cannot pass
merely by distinguishing the canonical family from exploding nonsense.

## 5. Executable synthetic gate

Inspect one canonical model:

```bash
quantumbci-audit dynamics \
  --omega-x 1.2 \
  --omega-z 0.8 \
  --gamma-dephasing 0.25 \
  --gamma-relaxation 0.35 \
  --json
```

Run the frozen moderate-SNR recovery grid:

```bash
quantumbci-audit e002-synthetic \
  --seed 2027 \
  --noise-std 0.003 \
  --output synthetic_recovery.json \
  --json
```

The first two E002 manifest stages are executable:

```bash
python -m quantumbci.experiments.tasks \
  synthetic-recovery E002 \
  --seed 2027 \
  --noise-std 0.003 \
  --output synthetic_recovery.json

python -m quantumbci.experiments.tasks \
  gate E002 identifiability \
  --input synthetic_recovery.json \
  --output identifiability_gate.json
```

The downstream identifiability stage independently rechecks recovery, equivalence, gauges,
canonical-family residuals and classical-adversary rejection instead of trusting an upstream pass
bit.

## 6. Synthetic promotion ceiling

The trajectory stage opens only when all of these hold:

- median normalized canonical-parameter recovery error <= 0.20;
- no systematic Hamiltonian sign inversion;
- exact Lindblad-to-affine equivalence witness passes;
- gauge-nonidentifiability witnesses pass;
- maximum canonical-family structure residual <= 0.05;
- the stable noncanonical affine adversary has residual >= 0.10 and is rejected.

Passing means the declared family is recoverable and family-specific under the frozen synthetic
conditions. It does not create additional quantum trajectory information or evidence of quantum
biology.

## 7. v0.8 temporal evidence authority

The next E002 stage is now executable too.

A sample split alone cannot define a dynamics benchmark because model behavior also depends on
which rows are temporally adjacent. v0.8 therefore adds `TrajectoryEvidenceAuthority` on top of the
upstream neural/sample authority.

It binds:

- exact state-tensor SHA-256;
- trajectory IDs;
- window start and stop times;
- fit, calibration and final-evaluation index sets;
- the subset allowed to fit the representation or choose latent dimensionality;
- fixed window duration and stride;
- temporal purge gap;
- missing-window policy;
- upstream authority fingerprint when available;
- exact source revisions.

Materialize it with:

```bash
python -m quantumbci.experiments.tasks \
  trajectory-contract E002 \
  --input trajectory_contract.json \
  --output trajectory_index.json
```

The contract then exposes only legal transitions whose two endpoints belong to the same evidence
role and trajectory and whose start-time delta matches the frozen stride.

Cross-role transitions do not exist in the model-facing graph.

See [E002 trajectory evidence authority](E002_TRAJECTORY_AUTHORITY.md) for the descriptor schema,
purge semantics, content-addressed identity and failure modes.

## 8. Fixed-window v1

The v0.8 authority deliberately supports fixed-duration, fixed-stride windows only.

Large gaps break a trajectory into blocks. A denser-than-declared start-time lattice fails closed.
Irregular timing remains unsupported until it receives explicit maximum-gap, integration-time and
missingness semantics.

This is preferable to giving two model families subtly different interpretations of an irregular
sequence.

## 9. Required classical controls for real trajectories

Every future E002 model lane must consume the identical:

```text
trajectory authority fingerprint
state tensor SHA-256
fit transition graph
evaluation transition graph
```

The control ladder remains:

1. exact affine/Bloch representation;
2. conventional regularized LDS/Kalman-style dynamics;
3. VAR where the state definition supports it;
4. damped oscillator controls for oscillatory low-dimensional trajectories;
5. switching-state dynamics for nonstationary regimes;
6. a flexible nonlinear control when sample size justifies it.

A lane may not independently recompute a look-alike state tensor and call the evidence matched.

## 10. What can count as a useful Lindblad result?

Tying an affine control on forecast error is insufficient because the full qubit trajectory already
has an exact affine representation.

A useful result needs a reproducible advantage in a defensible dimension such as:

- held-out prediction at matched effective complexity;
- lower calibration cost;
- stronger parameter stability across independent recordings/participants;
- more accurate preregistered intervention direction;
- materially simpler constrained parameterization for the same predictive behavior.

If the classical state-space model is equally predictive and more stable or simpler, it wins.

## 11. Biological interpretation ceiling

A fitted `gamma_dephasing` or `gamma_relaxation` remains a parameter in a **quantum-inspired latent
model**. It is not evidence for microscopic biological decoherence, entanglement, or a neural
quantum substrate.

A physical claim requires an independently specified substrate, operational witness,
discriminating perturbation, detection floor, strongest classical mimic and replication design.
Model fit cannot cross that gate.

## 12. Next implementation boundary

With synthetic identifiability and temporal authority both executable, the next E002 work is the
**matched dynamics fitting API**.

The first implementation should stay intentionally small:

1. unconstrained affine dynamics;
2. gauge-fixed canonical-family fit/projection;
3. regularized classical linear/LDS dynamics;
4. damped oscillator where appropriate.

Every fit artifact must bind the trajectory authority fingerprint and tensor SHA. Model selection,
regularization and hyperparameters must use only fit authority. Final evaluation transitions remain
immutable.

`fit-lindblad` and `fit-dynamics-controls` remain fail-closed until that matched fitting surface is
implemented and qualified.
