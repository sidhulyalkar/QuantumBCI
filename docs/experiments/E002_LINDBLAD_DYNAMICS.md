# E002: Lindblad-style latent dynamics vs classical dynamical systems

**Claim ceiling:** quantum-inspired.

## Question

Can an open-system density-state parameterization produce low-dimensional neural dynamics that are simultaneously predictive, identifiable, stable, and intervention-informative compared with classical dynamical controls?

## Stage 0: synthetic identifiability before EEG

Generate trajectories with known Hamiltonian couplings, dephasing/relaxation operators, sampling intervals, noise levels, and missingness. Fit the model from multiple initializations and seeds. The initial promotion threshold is median normalized parameter error <=0.20 at the preregistered moderate-SNR tier, with correct coupling signs and no systematic collapse-rate swaps.

Also generate classical LDS and damped-oscillator trajectories. A healthy mechanism learner should *not* hallucinate stable Lindblad-specific structure when the generator is classical.

## Real-data trajectories

Use continuous/sliding-window representations rather than independently shuffled epochs. Initial candidates:

- EEGMMIDB within-trial motor dynamics;
- Sleep-EDF across sleep-state transitions;
- later, event/seizure trajectories where licensing and preprocessing are controlled.

Trajectory state can be built from validated E001 embeddings or a deliberately low-dimensional classical feature basis. Keep the latent dimension small (initial grid 2, 4, 8) so parameters remain inspectable.

## Model suite

Quantum-inspired candidate:

- Hamiltonian + dephasing/relaxation Lindblad generator;
- constrained parameterization that preserves Hermiticity/trace and rejects invalid trajectories.

Required controls:

- Kalman/LDS;
- VAR;
- damped coupled oscillators;
- switching LDS/HMM;
- neural ODE or another flexible nonlinear model when sample size supports it.

Compare models under equal train/test trajectories and transparent parameter/search budgets.

## Prediction endpoints

- held-out sequence negative log likelihood where available;
- one-step and multi-step forecast error;
- trace/Frobenius distance between predicted and observed density summaries;
- forecast-horizon degradation curves;
- calibration/uncertainty if the model is probabilistic.

## Mechanistic endpoints

- bootstrap stability of Hamiltonian coupling signs/magnitudes;
- stability of collapse/dephasing rates;
- parameter-recovery accuracy on synthetic data;
- intervention response: zero/perturb a coupling and predict the direction of downstream change;
- state observables such as purity/coherence decay, treated only as model quantities.

Target bootstrap ICC for promoted real-data parameters is >=0.60. This is a pragmatic preregistered stability threshold, not a universal law.

## Falsification logic

Reject mechanism promotion when any of the following holds:

- strong classical models match/exceed held-out prediction;
- parameters change sign or swap identity across resamples;
- synthetic recovery fails at realistic SNR;
- interventions do not have consistent directional effects;
- a more parsimonious classical oscillator model explains the same trajectories.

A failed gate is a scientifically useful result and should still produce a completed report.

## Deliverables

- synthetic recovery surface across SNR/dimension;
- real trajectory contract/index;
- predictive comparison matrix;
- parameter stability atlas;
- intervention response curves;
- mechanism/evidence ledger.
