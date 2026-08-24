# Mathematical equivalence gates

QuantumBCI treats mathematical equivalence as a scientific result, not as an implementation detail.
Before a quantum-structured representation can be interpreted as carrying new information, the
workbench asks whether an ordinary classical statistic contains exactly the same information.

## Why this gate exists

A representation can look quantum because it is Hermitian, positive semidefinite, trace one, or
written with bra-ket/operator notation while still being an invertible or normalized rewrite of a
classical statistic. Predictive performance cannot distinguish two such descriptions if the
information content is identical and the downstream learner is allowed a fair parameterization.

The correct sequence is therefore:

```text
proposed quantum-structured object
             |
             v
   mathematical equivalence audit
             |
       +-----+-----+
       |           |
  equivalent   non-equivalent
       |           |
       v           v
attribute gains   test the genuinely
only to bias /    new information or
normalization /   mechanism against
readout geometry  matched controls
```

A finding of equivalence is a successful scientific outcome. It prevents overclaiming and narrows
the search to mechanisms that can actually make distinct predictions.

## E001 density/covariance identity

For a token matrix `X` with shape `tokens × features`, QuantumBCI currently constructs

```text
rho = X^H X / Tr(X^H X)
```

after optional feature-wise centering of the token cloud.

`X^H X` is the Hermitian second-moment matrix. After centering it is proportional to the ordinary
sample covariance matrix; trace normalization removes only a scalar degree of freedom. Therefore
`rho` is exactly the trace-normalized Hermitian second moment used by the mandatory classical
control.

The identity holds for both real and complex `X`.

### Consequence

The present density constructor **does not add representation information beyond normalized
covariance**. It may still be useful as an operator coordinate system for:

- constrained trace-one readouts;
- purity, entropy, or other spectral observables;
- operator-valued interventions;
- downstream non-commuting dynamics;
- regularization or parameter-sharing choices.

But those benefits must be described as inductive-bias, normalization, observable, or downstream
operator effects. They cannot be described as evidence that the density constructor discovered
additional quantum information in EEG.

Likewise, deleting density off-diagonal terms tests dependence on **cross-feature covariance**. It
is not evidence for microscopic quantum coherence.

## Installed audit

Audit one token matrix or a batch of token embeddings:

```bash
quantumbci-audit density embeddings.npy
quantumbci-audit density embeddings.npy --json
```

For the current constructor, the expected result is:

```text
equivalence class: trace_normalized_hermitian_second_moment
equivalent within tolerance: True
novel representation information: False
```

Run the broader E001 representation-control gauntlet with an explicit evidence split:

```bash
quantumbci-audit e001 embeddings.npy labels.npy \
  --train-indices train_indices.npy \
  --test-indices test_indices.npy \
  --output e001-audit.json
```

The E001 audit compares the density representation against normalized covariance, ordinary
covariance, log-covariance geometry, bilinear second moment, pooled statistics, train-only PCA,
diagonal density, and a fixed-readout off-diagonal intervention.

## Manifest gate

The E001 experiment DAG contains a real executable mathematical stage:

```bash
python -m quantumbci.experiments.tasks \
  equivalence-audit E001 density-covariance \
  --output equivalence_audit.json
```

This stage uses deterministic real and complex probes to regression-test that the implementation
continues to realize the algebraic identity. It does not use those probes as neuroscience evidence.
The algebra establishes the equivalence; the numerical probes protect the software contract.

Dataset extraction and model-fitting stages remain fail-closed until their executors are explicitly
implemented. QuantumBCI does not generate placeholder evidence merely so a manifest appears fully
runnable.

## Longitudinal E001 authority

Empirical E001 runs use the merged neurOS `LongitudinalCaseAuthority` rather than defining another
sample split. QuantumBCI first restores the neurOS authority, which revalidates processed-data
identity and the frozen source/calibration/evaluation boundary. It then evaluates each calibration
budget on the same final evaluation examples.

This separates responsibilities cleanly:

```text
neurOS
  data identity + chronology + calibration/evaluation authority
          |
          v
QuantumBCI
  token representation + equivalence audit + mechanism controls
          |
          v
participant-level paired inference
```

Repeated target sessions from one participant are aggregated within participant before bootstrap
resampling. Trial/window bootstrap is not accepted as promotion evidence when the independent unit
is the participant.

## Generalizing the pattern

The same discipline should be applied to every future mechanism:

| Candidate | First classical question |
| --- | --- |
| QFT observable | Is the requested observable exactly available from FFT/Goertzel? |
| quantum kernel | Does a matched classical kernel or random-feature map reproduce it? |
| Lindblad latent dynamics | Is it equivalent to a constrained LDS or damped oscillator system? |
| contextual operator model | Can an explicit history/state model make the same predictions? |
| QLSA observable | Does end-to-end state preparation/readout erase the claimed resource advantage? |

The goal is not to eliminate quantum structure by definition. The goal is to identify the smallest
residue that *cannot* be compiled back into a simpler classical description. That residue is where
new experiments are worth spending time and compute.
