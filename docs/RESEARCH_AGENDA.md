# QuantumBCI research agenda

## North star

Build an adversarial research workbench for one question:

> **When does quantum structure provide a uniquely useful and mechanistically interpretable model
> of neural information, after strong classical explanations are given their best chance?**

This splits the project into three independent programs: quantum-inspired neural representation,
quantum algorithms for neural computation, and physical quantum hypotheses in neural tissue. A win
in one program is not evidence for the others.

## Track A: Quantum-structured representations for EEG foundation models

### A1. Density geometry on frozen latent tokens

Use frozen embeddings from LaBraM, EEGPT, BrainWave, NeuroLM, or a specialist encoder. Convert each
window's token cloud to a trace-one PSD operator, then test operator observables and geometry as a
small probe layer.

**Experiments**

1. Cross-subject motor imagery/event decoding with identical frozen embeddings and identical heads.
2. Few-shot calibration curves at 1, 2, 5, 10, and 20 labelled examples per subject.
3. Cross-dataset transfer with no representation retraining.
4. Ablate off-diagonal terms while preserving eigenvalues or diagonal mass.
5. Compare to covariance, bilinear pooling, PCA, random PSD maps, and a learned SPD manifold model.

**Success condition.** A reproducible benefit in transfer/data efficiency plus stable interpretable
observables, not merely higher in-sample accuracy.

### A2. Open-system latent dynamics

Fit small Hamiltonian + collapse-operator models to trajectories of latent states across windows.
Start with 2-8 modes chosen from interpretable frequency/spatial components, not hundreds of opaque
dimensions.

**Questions**

- Do coupling parameters track known task transitions or oscillatory interactions?
- Do fitted dephasing rates predict loss of decoding confidence or state transitions?
- Are parameters stable across sessions/subjects?
- Can the model predict targeted perturbations better than an LDS/Kalman/neural ODE?

### A3. Contextual measurements

Use tasks where cue order or measurement context is experimentally manipulated. Pre-register the
operators and compare predicted AB/BA effects against history-aware classical models. EEG alone is
not enough: the experimental design must create a falsifiable context/order prediction.

## Track B: Quantum algorithms without imaginary speedups

### B1. QFT observable experiments

Start only where a downstream task needs a small set of spectral observables. Compare full resource
costs with FFT/Goertzel/classical sketching. Include amplitude-loading and readout costs. Simulation
is a correctness test, not evidence of acceleration.

### B2. Variational quantum feature maps

Compress neural/foundation latents to a genuinely small dimension, run parameter-matched quantum
kernels or variational circuits, and compare under identical train/test splits. Report shot noise,
noise-model sensitivity, circuit depth, effective dimension, and classical kernel controls.

### B3. QLSA/estimation observables

Do not reconstruct a full matrix inverse. Identify a Kalman/state-estimation quantity that can be
written as a small number of observables of a linear-system solution. Only then derive a resource
model and test a circuit implementation. This is a lower-priority track until the observable-level
problem is compelling.

## Track C: Physical quantum mechanisms

Treat this like experimental physics, not architecture search. Candidate mechanisms need a
specified substrate, coupling route, physiological timescale, operational witness, and classical
adversary. The repository should host preregistered simulations and analysis code, but no physical
claim should be inferred from quantum-inspired model performance.

A 2026 review of quantum/non-classical consciousness proposals emphasizes that operational evidence
for biologically relevant neural entanglement or long-lived coherence remains unestablished and
that classical nonlinear dynamics can reproduce putative signatures. This makes falsification and
classical controls the interesting research problem, not a reason to avoid the topic.

## Benchmark ladder

### Level 0: Mathematical invariants

Hermiticity, PSD, trace preservation, unitary norm preservation, normalized measurement
probabilities, deterministic seeds, numerical stability.

### Level 1: Synthetic mechanism recovery

Generate known oscillatory/contextual/open-system mechanisms and test parameter recovery and null
rejection. Include classical look-alikes deliberately designed to fool the model.

### Level 2: Public EEG tasks

Start with small, auditable datasets and subject-held-out protocols. Report calibration, data
efficiency, uncertainty, and compute alongside accuracy/AUROC.

### Level 3: Frozen foundation representations

Probe LaBraM/EEGPT/BrainWave/NeuroLM representations without changing the base encoder. This makes
it possible to isolate whether the quantum-structured layer contributes anything new.

### Level 4: Prospective/interventional data

Design cue-order, perturbation, or longitudinal protocols around a preregistered mechanism
prediction. This is where mechanistic claims become substantially more valuable.

### Level 5: Quantum hardware/resource study

Only after a circuit-level hypothesis survives classical controls. Report end-to-end resources and
hardware noise; do not use simulator speed as evidence of quantum advantage.

## Modern baseline context

The project should benchmark against the current EEG foundation-model landscape rather than against
FFT/Kalman alone. Useful starting points include:

- LaBraM, ICLR 2024: https://openreview.net/forum?id=QzTpTRVtrP
- EEGPT, NeurIPS 2024: https://github.com/BINE022/EEGPT
- BrainWave: https://arxiv.org/abs/2402.10251
- NeuroLM, ICLR 2025: https://arxiv.org/abs/2409.00101
- 2026 EEG foundation-model benchmarking survey: https://arxiv.org/abs/2601.17883

The 2026 benchmark is especially useful because it finds that larger foundation models do not
uniformly beat specialist baselines and that linear probing is often insufficient. QuantumBCI
should therefore measure *incremental mechanism value* rather than assume a large frozen encoder is
a solved baseline.

## Near-term milestone sequence

1. **v0.2 research kernel**: claim ledger, density/open-system/contextual primitives, stable Kalman
   baseline, QFT semantics, tests, CI. (This PR.)
2. **v0.3 benchmark harness**: BIDS/MNE-compatible dataset adapters, subject/session split registry,
   metric manifests, matched-null runner, reproducible experiment artifacts.
3. **v0.4 foundation adapters**: frozen LaBraM + EEGPT extraction and density-geometry probes.
4. **v0.5 dynamic mechanisms**: fit/identify Lindblad versus LDS/Kalman on latent trajectories.
5. **v0.6 contextual experiment**: preregistered order-effect dataset/task and classical adversaries.
6. **v0.7 quantum hardware sandbox**: only for mechanisms that survived the prior ladder.
