# E002: Open-system dynamics without parameter mythology

E002 asks whether a low-dimensional Lindblad/GKSL-style parameterization can provide useful,
stable and intervention-predictive structure for neural latent trajectories after the strongest
classical equivalent is made explicit.

The first result is mathematical, not empirical:

> A time-homogeneous qubit Lindblad equation on the fully observed density state is exactly a
> three-dimensional classical affine linear ODE on the Bloch vector.

That equivalence is now executable in QuantumBCI.

## 1. Exact qubit compilation

Write a qubit density state as

```text
rho = (I + r_x sigma_x + r_y sigma_y + r_z sigma_z) / 2.
```

For a linear Lindblad generator `L`,

```text
d r_i / dt = Tr[sigma_i L(rho)].
```

Linearity gives an exact classical system

```text
d r / dt = A r + b
```

with

```text
b_i    = Tr[sigma_i L(I)] / 2
A_ij   = Tr[sigma_i L(sigma_j)] / 2.
```

`compile_qubit_lindblad_to_affine(...)` performs this compilation directly. The audit compares both
instantaneous derivatives and RK4 trajectories and fails if the two representations drift beyond
tolerance.

This means a fully observed qubit Lindblad trajectory is **not dynamical-information novel** relative
to the compiled affine state-space model.

The Lindblad coordinates may still be valuable if they impose useful constraints, regularize small
data, produce stable interpretable invariants, or make intervention predictions easier to state.
Those are empirical questions. Extra trajectory information is not.

## 2. Parameter gauges

Arbitrary Hamiltonian and collapse-matrix entries are not separately identifiable.

QuantumBCI explicitly witnesses several standard gauges:

### Hamiltonian identity shift

```text
H -> H + c I
```

leaves the commutator and therefore the generator unchanged.

### Collapse global phase

```text
C_j -> exp(i phi_j) C_j
```

leaves the corresponding dissipator unchanged.

### Unitary mixing of collapse channels

For multiple collapse operators,

```text
C'_a = sum_j U_aj C_j
```

with unitary `U` leaves the summed dissipator unchanged.

Therefore E002 may recover only:

- a declared canonical gauge;
- generator-level quantities;
- invariants that survive these transformations.

It must not report an arbitrary optimizer's collapse matrices as uniquely identified neural
mechanisms.

## 3. Canonical synthetic family

The first qualification family deliberately fixes the gauge:

```text
H = (omega_x sigma_x + omega_z sigma_z) / 2
```

with two declared dissipative processes:

- z dephasing at `gamma_dephasing`;
- amplitude relaxation toward `|0>` at `gamma_relaxation`.

In Bloch form this family has

```text
A_xx = A_yy = -(gamma_relaxation / 2 + gamma_dephasing)
A_zz = -gamma_relaxation
b_z  =  gamma_relaxation
A_yx = -A_xy = omega_z
A_zy = -A_yz = omega_x.
```

The recovery benchmark does **not** fit Lindblad matrices directly. It:

1. generates trajectories through the Lindblad implementation;
2. adds preregistered moderate observation noise;
3. fits an unconstrained classical affine generator `A,b` by least squares on trajectory finite
   differences;
4. recovers the four canonical parameters from redundant entries of that fitted generator;
5. reconstructs the complete canonical generator from those recovered parameters;
6. measures the maximum residual across **every** entry of `A,b`;
7. reports normalized recovery error and Hamiltonian sign inversions;
8. separately runs affine-equivalence and gauge witnesses.

The whole-generator residual is essential. Parameter extraction alone can project an arbitrary
classical affine system into plausible `omega` and `gamma` coordinates even when extra couplings,
anisotropic decay, or affine offsets make that system inconsistent with the declared canonical
family.

`canonical_structure_residual(...)` therefore separates two questions:

```text
parameter projection: "which canonical numbers resemble this generator?"
family identification: "does the full generator actually lie near that family?"
```

E002 requires both.

## 4. Classical family-specificity adversary

The synthetic gate includes an explicit stable affine look-alike that contains plausible
omega/gamma-like entries but also structure forbidden by the declared canonical family.

The adversary adds, among other terms:

- anisotropic transverse decay;
- additional cross-axis couplings;
- a noncanonical affine x-offset.

It is intentionally dynamically stable. Rejection therefore cannot be explained by comparing the
canonical family only against pathological or exploding dynamics.

The synthetic contract requires:

```text
maximum canonical-case structure residual <= 0.05
noncanonical adversary structure residual >= 0.10
```

This makes the gate adversarial in both directions: genuine canonical trajectories must project
back into the family, while a reasonable classical look-alike must remain outside it.

## 5. Executable audit

Inspect one canonical model:

```bash
quantumbci-audit dynamics \
  --omega-x 1.2 \
  --omega-z 0.8 \
  --gamma-dephasing 0.25 \
  --gamma-relaxation 0.35 \
  --json
```

Run the moderate-SNR synthetic grid:

```bash
quantumbci-audit e002-synthetic \
  --seed 2027 \
  --noise-std 0.003 \
  --output synthetic_recovery.json \
  --json
```

The resulting artifact records parameter recovery, generator equivalence, gauge witnesses,
canonical-structure residuals and the noncanonical classical-adversary result.

The first E002 manifest stage is also executable:

```bash
python -m quantumbci.experiments.tasks \
  synthetic-recovery E002 \
  --seed 2027 \
  --noise-std 0.003 \
  --output synthetic_recovery.json
```

Then apply the frozen identifiability gate:

```bash
python -m quantumbci.experiments.tasks \
  gate E002 identifiability \
  --input synthetic_recovery.json \
  --output identifiability_gate.json
```

The downstream gate independently rechecks the recovery and family-specificity criteria. It does
not trust only the upstream artifact's aggregate pass bit. A malformed or altered artifact that
claims successful parameter recovery while losing adversary rejection therefore remains ineligible
for the trajectory stage.

## 6. Synthetic promotion gate

The real-trajectory contract does not open unless all of these hold:

- median normalized canonical-parameter recovery error <= 0.20;
- no systematic sign inversion of nonzero Hamiltonian frequencies;
- exact qubit Lindblad-to-affine equivalence witness passes;
- gauge-nonidentifiability witnesses pass;
- maximum canonical-family structure residual <= 0.05;
- the stable noncanonical classical adversary has residual >= 0.10 and is rejected.

Passing this gate means only that the declared canonical family is numerically recoverable and
family-specific under the declared synthetic conditions, while the implementation correctly
recognizes its classical equivalence and parameter gauges.

It does not establish a quantum neural mechanism or additional quantum trajectory information.

## 7. Required classical controls for real trajectories

When E002 reaches real neural latent sequences, the control ladder must include:

1. the **exact affine/LDS representation** corresponding to the fitted qubit state surface;
2. conventional LDS/Kalman state-space fitting;
3. VAR where the observation/state definition makes it meaningful;
4. damped oscillator controls for oscillatory low-dimensional trajectories;
5. switching-state dynamics for nonstationary regimes;
6. a flexible nonlinear control when sample size permits.

The Lindblad parameterization is not promoted merely for tying an affine control on forecast error.
A useful result must show a reproducible advantage in at least one defensible dimension such as:

- held-out prediction at matched complexity;
- lower calibration cost;
- parameter stability across independent cases;
- more accurate intervention direction;
- substantially simpler constrained parameterization.

If the classical state-space model is equally predictive and more stable or simpler, it wins.

## 8. Biological interpretation ceiling

A fitted `gamma_dephasing` or `gamma_relaxation` is a parameter in a **quantum-inspired latent model**.
It is not evidence for microscopic biological decoherence or a neural quantum substrate.

A physical claim requires an independently specified substrate, operational witness, discriminating
perturbation, detection floor and strongest classical mimic. Model fit alone cannot cross that gate.

## 9. Next implementation boundary

After the synthetic gate is qualified, the next E002 work should be the trajectory evidence contract,
not a larger Lindblad model. That contract must specify:

- exact encoder/source revision;
- continuous-window chronology;
- train/calibration/final-evaluation boundaries;
- latent dimensionality selection using training authority only;
- time-step and missing-data semantics;
- frozen preprocessing;
- identical trajectory tensors for Lindblad and classical controls;
- participant/recording-level inference units;
- how the exact affine representation, conventional state-space controls and the constrained
  canonical family share the same evaluation authority;
- how family-specificity residuals are interpreted on held-out real trajectories without treating
  proximity to the canonical family as proof of quantum biology.

Until that contract exists, the real-data `fit-lindblad` and `fit-dynamics-controls` manifest stages
remain deliberately fail-closed.
