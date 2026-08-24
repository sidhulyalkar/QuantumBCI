# Mechanistic interpretability protocol

The goal is not to decorate a neural decoder with quantum vocabulary. The goal is to determine
whether a proposed mechanism exposes variables that predict interventions and generalize better
than plausible alternatives.

## 1. Freeze the claim before the score

Every experiment names a mechanism card and its maximum claim class before training. A
`quantum_inspired` result cannot be promoted to `physical_quantum` because its AUROC is high.

## 2. Use three model layers

**Observation layer.** Raw EEG/MEG/ECoG or frozen foundation-model tokens with explicit units,
channels, sampling rate, preprocessing, and subject/session identifiers.

**Mechanism layer.** The smallest proposed state and operators: density state, Hamiltonian,
collapse operators, contextual projectors, or a circuit/measurement protocol.

**Readout layer.** A deliberately simple task head when possible. If a deep head is required, run
a control where the mechanism representation is replaced while the head is held fixed.

## 3. Expose mechanism observables

For density/open-system models record at least purity, entropy, coherence, operator expectations,
and fitted coupling/dissipation parameters. For contextual models record the commutator and the
predicted AB/BA effect. For algorithms record qubits, depth, shots, preparation/readout strategy,
noise model, and classical wall-clock/energy baseline.

## 4. Perform interventions, not only saliency

Recommended tests:

- channel/band/token ablations;
- phase randomization while preserving power spectra;
- time reversal and temporal shuffling;
- subject/session swap controls;
- targeted latent-mode suppression;
- perturb one fitted coupling or collapse rate and predict the direction of the downstream change;
- replace non-commuting operators with commuting matched operators;
- destroy off-diagonal density terms while preserving the diagonal.

A mechanism becomes more credible when its exposed variable predicts the intervention response.

## 5. Demand matched classical nulls

At minimum compare against linear/covariance features, a robust state-space model, and a modern
specialist or pretrained EEG encoder where appropriate. Match parameter count, frozen features,
training data, hyperparameter budget, and task head as tightly as possible.

## 6. Test identifiability and stability

Use subject-held-out splits, session-held-out splits, bootstrap confidence intervals, multiple
seeds, parameter recovery on synthetic data, and permutation/null datasets. A mechanistic parameter
that changes sign across resamples is not yet an interpretable mechanism.

## 7. Separate predictive evidence from mechanistic evidence

Report two ledgers:

- **Predictive:** discrimination/regression error, calibration, transfer, data efficiency, compute.
- **Mechanistic:** parameter recovery, intervention prediction, invariance, identifiability,
  falsification tests, and evidence against classical alternatives.

## 8. Promote claims only with new evidence

`classical_control` -> `quantum_inspired` requires a specific quantum-structured inductive bias and
matched controls. `quantum_inspired` -> `physical_quantum` is not a modelling promotion; it requires
independent physical evidence about the biological substrate.
