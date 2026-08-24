# Mechanism cards

QuantumBCI uses four claim classes. They are deliberately ordered by *meaning*, not prestige.
A model may use quantum mathematics without asserting microscopic quantum biology.

| Claim class | Meaning | What counts as evidence |
| --- | --- | --- |
| `classical_control` | Standard signal processing/statistics/ML | Predictive and mechanistic validation |
| `quantum_inspired` | Hilbert-space, density-operator, non-commutative, tensor-network, or open-system mathematics used as a model class | Reproducible advantage over complexity-matched classical alternatives plus identifiable mechanism variables |
| `quantum_algorithm` | Computation executed as a quantum algorithm or faithful resource model | End-to-end resource accounting including encoding, circuit, sampling, error, and readout |
| `physical_quantum` | The biological substrate itself is claimed to sustain a non-classical physical mechanism | Direct operational witnesses and exclusion of classical alternatives; model fit alone is insufficient |

## M1: Density-operator latent geometry

**Class:** `quantum_inspired`

**Hypothesis.** Normalized PSD latent states preserve mixtures, uncertainty, and cross-feature
structure in a useful geometry that transfers across people/tasks better than matched covariance
or linear representations.

**Primary observables.** Purity, von Neumann entropy, basis-dependent coherence, expectation
values of preregistered operators.

**Controls.** Covariance features, PCA/whitening, bilinear pooling, linear probes, random rotations,
and parameter-matched PSD models without quantum terminology.

**Falsify or demote if.** Gains disappear under subject-held-out evaluation, are explained by
normalization alone, or observables are unstable under bootstrap/resampling.

## M2: Lindblad latent dynamics

**Class:** `quantum_inspired` by default. It becomes `physical_quantum` only if the state and
operators are tied to independently measured physical quantum degrees of freedom.

**Hypothesis.** Hamiltonian-like coupling plus dissipative channels offers an interpretable compact
model of coupled latent oscillations, transitions, and loss of coherence.

**Primary observables.** Coupling strengths, dephasing/dissipation rates, purity/coherence decay,
transition probabilities, held-out trajectory likelihood/error.

**Controls.** Kalman/LDS, switching LDS, state-space Gaussian process, neural ODE/SDE, damped
coupled oscillators.

**Falsify or demote if.** Parameters are non-identifiable, trajectories do not transfer across
sessions, or classical state-space models match performance and explanatory compression.

## M3: Contextual/non-commuting measurement model

**Class:** `quantum_inspired`

**Hypothesis.** Some reproducible cue/order/context effects are represented more compactly by
non-commuting observables and state-updating measurements than by static classical probability.

**Primary observables.** Commutator norm; AB-versus-BA sequential probabilities; interference
terms; held-out order-effect prediction.

**Controls.** Explicit history features, HMMs, RNNs, context-conditioned logistic models, causal
state-space models.

**Falsify or demote if.** Order effects fail replication or disappear once history is represented
with a matched classical model.

## M4: QFT sampling

**Class:** `quantum_algorithm`

**Hypothesis.** For carefully selected spectral observables, amplitude encoding + QFT + targeted
measurement may provide a useful quantum computational primitive.

**Critical boundary.** QFT measurement probabilities are **not** the same object as a complex FFT.
They discard phase under computational-basis measurement. A quantum advantage claim must include
state preparation and the measurements required by the downstream task.

**Controls.** FFT, Goertzel/target-frequency estimators, random projections, classical sketching.

## M5: Quantum linear systems inside state estimation

**Class:** `quantum_algorithm`, currently research-only.

The previous repository treated `np.linalg.inv` as a quantum-enhanced placeholder and included an
Aqua HHL path. This is now intentionally removed from the active implementation. A QLSA does not
return a cheap full classical inverse; its asymptotic story is strongest when the desired output is
a quantum-state solution or a small number of observables. Kalman gain construction can require
multiple right-hand sides and substantial readout, so an advantage must be derived for the exact
estimator being used.

`quantumbci.kalman.qlsa_diagnostics` exposes matrix properties and the most important caveats before
an experimental backend is attempted.

## M6: Physical quantum neural mechanisms

**Class:** `physical_quantum`

This is deliberately a separate, high-evidence track. Candidate substrates (for example nuclear
spins, molecular excitations, microtubular proposals, or photonic mechanisms) should not be mixed
with quantum-like cognition merely because both use Hilbert-space language.

A credible experiment needs, at minimum:

1. a specified physical degree of freedom and coupling pathway to neural computation;
2. a predicted coherence/relaxation timescale under physiological conditions;
3. an operational witness or perturbation that distinguishes the hypothesis from classical
   nonlinear/stochastic dynamics;
4. independent replication and adversarial classical controls;
5. no inference from classifier accuracy alone.

This repository currently implements **no physical-quantum neural mechanism** and therefore makes
no claim that neural tissue exhibits biologically functional entanglement or long-lived coherence.
