# QuantumBCI research agenda

## North star

Build an adversarial research workbench for one question:

> **When does quantum structure provide a uniquely useful and mechanistically interpretable model
> of neural information, after strong classical explanations are given their best chance?**

This splits the project into three independent programs: quantum-inspired neural representation,
quantum algorithms for neural computation, and physical quantum hypotheses in neural tissue. A win
in one program is not evidence for the others.

A fourth rule cuts across every program: **audit mathematical equivalence before measuring predictive
advantage**. If a proposed quantum object is an invertible, normalized, or otherwise information-
equivalent rewrite of a classical statistic, QuantumBCI records that as a successful falsification
result and attributes later gains only to parameterization, regularization, observables, or downstream
operator structure.

See [Mathematical equivalence gates](MATHEMATICAL_EQUIVALENCE.md).

## Track A: Quantum-structured representations for EEG foundation models

### A1. Density/operator geometry on frozen latent tokens

Use frozen token-level representations from LaBraM, EEGPT, BrainWave, NeuroLM, neurOS-compatible
encoders, or specialist models. Convert each window's token cloud to a trace-one PSD operator, but
first determine exactly what classical statistic that operator contains.

For the current constructor,

```text
rho = X^H X / Tr(X^H X)
```

after optional centering. This is exactly a trace-normalized Hermitian second moment, and therefore
contains no information beyond the corresponding normalized covariance object. The density notation
may still provide a useful constrained operator coordinate system, but its constructor is not an
information-novel mechanism.

**Experiments**

1. Run the exact density/covariance equivalence audit before fitting predictive models.
2. Use the merged neurOS `LongitudinalCaseAuthority` so every method shares source history,
   calibration examples, and fixed final evaluation examples.
3. Compare full density features to exact normalized covariance, ordinary covariance,
   log-covariance geometry, bilinear second moment, pooled statistics, train-only PCA, and other
   declared controls on the same token tensor.
4. Treat off-diagonal deletion as a cross-feature-covariance intervention, not evidence of
   microscopic quantum coherence.
5. Test whether density-derived observables, constraints, or later non-commuting operator dynamics
   contribute reproducible value beyond the information-equivalent classical representation.
6. Aggregate repeated sessions within participant and bootstrap participants, rather than windows,
   for promotion-oriented longitudinal inference.

**Success condition.** The current density constructor itself is not eligible for an
information-novelty claim while exact normalized-covariance equivalence holds. A useful result can
instead be a reproducible benefit attributable to a clearly named operator inductive bias,
observable, intervention, or downstream non-equivalent mechanism. A null/equivalence result is a
successful scientific outcome.

### A2. Open-system latent dynamics

Fit small Hamiltonian + collapse-operator models to trajectories of latent states across windows.
Start with 2-8 modes chosen from interpretable frequency/spatial components, not hundreds of opaque
dimensions.

Before empirical promotion, identify the closest classical constrained dynamical system. A Lindblad
parameterization that collapses to a linear dynamical system or damped coupled oscillator under the
measured observables should be treated as an equivalence class, not as unique mechanism evidence.

**Questions**

- Do coupling parameters track known task transitions or oscillatory interactions?
- Do fitted dephasing parameters predict loss of decoding confidence or state transitions beyond
  matched damping/noise parameters in classical systems?
- Are parameters identifiable and stable across sessions/subjects?
- Can the model predict targeted interventions better than an LDS/Kalman/damped-oscillator/neural
  ODE control?

### A3. Contextual measurements

Use tasks where cue order or measurement context is experimentally manipulated. Pre-register the
operators and compare predicted AB/BA effects against history-aware classical models. EEG alone is
not enough: the experimental design must create a falsifiable context/order prediction.

The first equivalence question is whether an explicit-history or latent-state classical model can
make the same prospective predictions. Non-commuting notation without a distinct prediction is not
mechanism novelty.

## Track B: Quantum algorithms without imaginary speedups

### B1. QFT observable experiments

Start only where a downstream task needs a small set of spectral observables. Compare the requested
observable first with FFT, Goertzel, classical sketching, or other direct estimators. Include
amplitude-loading and readout costs. Simulation is a correctness test, not evidence of acceleration.

### B2. Variational quantum feature maps

Compress neural/foundation latents to a genuinely small dimension, run parameter-matched quantum
kernels or variational circuits, and compare under identical train/test splits. Report shot noise,
noise-model sensitivity, circuit depth, effective dimension, and matched classical kernel/random-
feature controls. The relevant question is not whether a circuit is nonlinear; it is whether its
kernel or observable family remains distinct after the strongest classical reconstruction.

### B3. QLSA/estimation observables

Do not reconstruct a full matrix inverse. Identify a Kalman/state-estimation quantity that can be
written as a small number of observables of a linear-system solution. Only then derive a resource
model and test a circuit implementation. State preparation, condition number, sparsity, sampling,
and readout belong inside the comparison rather than outside the claimed speedup.

## Track C: Physical quantum mechanisms

Treat this like experimental physics, not architecture search. Candidate mechanisms need a
specified substrate, coupling route, physiological timescale, operational witness, and classical
adversary. The repository should host preregistered simulations and analysis code, but no physical
claim should be inferred from quantum-inspired model performance.

A mathematically non-classical model fit is not itself an operational witness of a physical quantum
substrate. Physical hypotheses must survive instrument/detection-floor analysis and perturbations
that distinguish the proposed quantum mechanism from classical mimics.

## Benchmark ladder

### Level 0: Mathematical invariants and equivalence

Check Hermiticity, PSD, trace preservation, unitary norm preservation, normalized measurement
probabilities, deterministic seeds, numerical stability, and whether the proposed representation or
observable is exactly reconstructible from a standard classical statistic.

A failed novelty gate stops information-novelty claims but does not stop the software run.

### Level 1: Synthetic mechanism recovery

Generate known oscillatory/contextual/open-system mechanisms and test parameter recovery and null
rejection. Include classical look-alikes deliberately designed to fool the model. Synthetic data
qualify identifiability and implementation, not biological truth.

### Level 2: Public EEG tasks with frozen evidence authority

Start with auditable public datasets and subject/session-held-out protocols. For longitudinal EEG,
reuse neurOS authority rather than inventing a parallel split. Report calibration, data efficiency,
uncertainty, compute, and participant-level paired inference alongside accuracy/AUROC.

### Level 3: Frozen foundation representations

Probe LaBraM/EEGPT/BrainWave/NeuroLM or other token representations without changing the base
encoder. Preserve exact checkpoint/model revision and representation hashes so incremental mechanism
value can be isolated from representation quality.

### Level 4: Prospective/interventional data

Design cue-order, perturbation, or longitudinal protocols around a preregistered mechanism
prediction. This is where mechanistic claims become substantially more valuable.

### Level 5: Quantum hardware/resource study

Only after a circuit-level hypothesis survives mathematical equivalence and classical controls.
Report end-to-end resources and hardware noise; do not use simulator speed as evidence of quantum
advantage.

## Ecosystem composition

QuantumBCI should remain a research extension rather than a second neural runtime:

```text
neurOS
  data identity / replay / chronology / split + calibration authority
      |
      v
specialist + foundation token representations
      |
      v
QuantumBCI
  equivalence audit / quantum-structured mechanisms / matched controls
      |
      v
neuros-mechint
  intervention evidence / replication / evidence packs
```

This one-way dependency protects attribution. neurOS can compare EEGNet, EEG-Conformer,
SourceWeigher, ORION, and other methods independently. QuantumBCI is responsible only for the
incremental mechanism it adds on the same evidence authority.

## Modern baseline context

The project should benchmark against the current EEG foundation-model landscape rather than against
FFT/Kalman alone. Useful starting points include:

- LaBraM, ICLR 2024: https://openreview.net/forum?id=QzTpTRVtrP
- EEGPT, NeurIPS 2024: https://github.com/BINE022/EEGPT
- BrainWave: https://arxiv.org/abs/2402.10251
- NeuroLM, ICLR 2025: https://arxiv.org/abs/2409.00101
- 2026 EEG foundation-model benchmark: https://arxiv.org/abs/2601.17883

The 2026 benchmark is especially useful because it finds that larger foundation models do not
uniformly beat specialist baselines and that linear probing is often insufficient. QuantumBCI
should therefore measure *incremental mechanism value* rather than assume a large frozen encoder is
a solved baseline.

## Near-term milestone sequence

1. **v0.2 research kernel:** claim ledger, density/open-system/contextual primitives, stable Kalman
   baseline, QFT semantics, tests, CI.
2. **v0.3 orchestration + neurOS bridge:** reviewed experiment DAGs, shared evidence authority,
   foundation/runtime interoperability, mech-int bridge.
3. **v0.4 local research workbench:** installed CLI, run registry, deterministic synthetic smoke,
   frozen-embedding benchmark, HTML evidence reports.
4. **v0.5 portable public research:** content-addressed recipes, closed-world artifact verification,
   RO-Crate export, BIDS-aware evidence containers, contribution/citation surfaces.
5. **v0.6 equivalence-first longitudinal E001:** mathematical equivalence auditor, adversarial
   covariance/control gauntlet, merged neurOS `LongitudinalCaseAuthority`, calibration-frontier
   execution, representation fingerprints, and participant-level inference.
6. **v0.7 open-system dynamics:** synthetic identifiability gate followed by Lindblad-vs-LDS/
   oscillator comparisons on promoted latent trajectories.
7. **v0.8 prospective contextual experiment:** preregistered AB/BA predictions with explicit-history
   and latent-state classical adversaries.
8. **v0.9 quantum resource sandbox:** hardware only for observables that survive the earlier
   mathematical, predictive, and mechanistic gates.
