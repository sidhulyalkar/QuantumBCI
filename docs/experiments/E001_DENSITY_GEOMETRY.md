# E001: Density geometry on EEG foundation-model representations

**Claim ceiling:** quantum-inspired. This experiment cannot establish microscopic quantum states in brain tissue.

## Question

When the upstream EEG encoder, subjects, split, and readout budget are held fixed, does a trace-one positive-semidefinite representation provide transferable information that is not captured by strong classical geometry?

This is intentionally narrower than asking whether a foundation model is state of the art. Recent 2026 benchmarks report that linear probing can underuse EEG foundation models and that specialist models remain competitive. Frozen encoders are used here to isolate the representation layer; a later adapted-encoder experiment is required for an application-performance claim.

## Hypotheses

- **H1 predictive:** density geometry improves subject-held-out macro-F1/balanced accuracy or calibration against the strongest complexity-matched representation control.
- **H2 mechanism:** the benefit depends on structured off-diagonal terms. Zeroing off-diagonals should reduce the effect if a coherence-like cross-feature structure is genuinely carrying information.
- **H3 stability:** purity, entropy, coherence, or learned observables associated with performance are stable across seeds and subject bootstrap samples.
- **Null:** a covariance/Riemannian/bilinear representation explains the same gains, or density-specific observables are unstable.

## Data ladder

1. **Smoke:** synthetic fixtures already in QuantumBCI.
2. **Open initial:** EEGMMIDB for motor imagery and Sleep-EDF for a distinct clinical/physiological paradigm.
3. **External replication:** TUAB/TUEV when access/licensing credentials are available, because LaBraM has established preprocessing/fine-tuning paths there.

Every adapter writes a data contract: source/version, license/access status, checksum/fingerprint, channel map, units, sampling rate, filtering, epoching, rejected segments, and subject/session identifiers.

## Encoders

Start with LaBraM and EEGPT because they provide materially different pretrained EEG representations. Add one compact strong foundation model from the current benchmark landscape when adapter licensing is clear, plus a specialist EEGNet-style control. The same preprocessed example must never be encoded differently based on downstream representation choice.

## Split protocol

- Group exclusively by subject; session is nested inside subject.
- Initial development: fixed group holdout plus group-aware inner CV.
- Final report: LOSO where computationally feasible or repeated subject-group holdout with a locked seed registry.
- Few-shot calibration: `k={1,2,4,8,16}` labelled examples per class/subject, with all hyperparameter decisions made without the final test labels.
- No EEG window from a test subject may enter normalization fitting, PCA fitting, representation-basis fitting, or readout selection.

## Representation suite

All representations receive the same dimensional budget where possible:

1. token mean / pooled latent;
2. covariance features;
3. Riemannian covariance tangent-space features;
4. bilinear/Gram features;
5. PCA projection;
6. random PSD projection with matched dimension;
7. QuantumBCI density state + registered observables.

Density-specific ablations:

- diagonal-only density state;
- off-diagonal phase/sign scrambling while preserving diagonal mass;
- eigenvalue-preserving random eigenbasis;
- token permutation;
- observable removal one at a time.

## Readout controls

Primary readout is a regularized linear/logistic model because it limits capacity confounding. A small fixed MLP can be a secondary readout, with identical architecture and search budget for every representation. Report parameter count and fit time.

## Primary statistics

The independent unit is subject. Report macro-F1 and balanced accuracy, AUROC where meaningful, ECE/Brier calibration, paired per-subject deltas, 95% subject bootstrap confidence intervals, and a subject-level sign/permutation test.

The preregistered promotion target is: against the strongest matched control, the paired 95% bootstrap interval excludes zero on at least two datasets/paradigms and there is either >=1 percentage point absolute classification gain or >=2 percentage points absolute calibration improvement. A smaller repeatable effect remains publishable/descriptive but does not trigger mechanism promotion automatically.

## Mechanistic interpretation gate

A predictive win is not enough. At least one density-specific intervention must causally degrade the claimed benefit, and the implicated observable must be reproducible across bootstrap resamples. If diagonal-only or a classical tangent-space model retains the entire advantage, record the result as a classical geometry result.

## Deliverables

- immutable embedding cache index;
- representation artifact hashes;
- subject-level prediction table;
- matched-control leaderboard with confidence intervals;
- ablation matrix;
- observable stability report;
- predictive and mechanistic evidence ledgers;
- negative-result report when gates fail.

## References

- LaBraM: https://openreview.net/forum?id=QzTpTRVtrP
- EEGPT: https://arxiv.org/abs/2410.19779
- EEG foundation-model benchmark (2026): https://arxiv.org/abs/2601.17883
- ICLR 2026 comparative evaluation: https://proceedings.iclr.cc/paper_files/paper/2026/hash/f0f39b7686634fc81ca0112566b2c05f-Abstract-Conference.html
