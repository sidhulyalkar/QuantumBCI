# Architecture

QuantumBCI is a dependency-light research workbench for **falsifying neural mechanism claims**. Its architecture is designed around a stricter principle than ordinary benchmark software: evidence authority, matched alternatives, artifact integrity, and claim ceilings are part of the computation rather than reporting decorations.

The base package remains NumPy-first. Dataset loading, large neural models, and cross-project evidence runtimes belong behind optional adapters.

## System boundary

```text
raw/public neural data or frozen representations
                    |
                    v
                 neurOS
      data identity / chronology / replay
      participant + session evidence authority
      calibration and final-evaluation boundaries
                    |
                    v
               QuantumBCI
    +---------------+----------------+
    |               |                |
    v               v                v
 classical       candidate       equivalence /
 controls        mechanisms      adversary tests
    |               |                |
    +---------------+----------------+
                    |
                    v
        authority-bound case artifacts
        fingerprints + tamper rejection
                    |
           +--------+---------+
           |                  |
           v                  v
     BMRB aggregation    neuros-mechint
     reliability /       interventions /
     representation      ablations /
     conservation        matched recovery
           |                  |
           +--------+---------+
                    |
                    v
       confirmatory evidence profile
       preregistered policy + inference
                    |
                    v
        known-truth validation layer
```

The dependency direction is intentional. **neurOS must not depend on QuantumBCI.** QuantumBCI consumes evidence authority from neurOS and interoperates with `neuros-mechint` through explicit artifact contracts.

## Architectural layers

### 1. Mathematical and mechanism kernels

Small numerical modules implement inspectable candidate mechanisms and strong alternatives:

- `states.py`, `spectral.py`, `contextuality.py`, `open_system.py`: quantum-inspired and quantum-algorithm primitives;
- `equivalence.py`, `dynamics_equivalence.py`: exact or near-exact mathematical equivalence audits;
- `classical_dynamics.py`, `probabilistic_ssm.py`, `switching_dynamics.py`, `nonlinear_dynamics.py`: increasingly flexible classical adversaries;
- `dynamics_fitting.py`, `e002_synthetic.py`: constrained/open-system fitting and known-structure recovery.

These modules should remain as pure and dependency-light as practical. Optimization belongs here only after profiling and must preserve deterministic scientific outputs.

### 2. Evidence authority and reproducibility

Evidence-boundary modules prevent convenience code from quietly changing the scientific question:

- `trajectory_authority.py`: immutable fit/calibration/evaluation authority for E002-style trajectories;
- `longitudinal.py` and E001 prepared/study modules: participant/session-aware evaluation;
- `recipes.py`: portable frozen-input recipes;
- `exporting.py`: closed-world artifact verification plus conservative RO-Crate/BIDS export;
- `preregistration.py`: external-registration metadata and exact policy fingerprint binding.

Final evaluation data must never become fitting, preprocessing, model-selection, comparator-selection, or adaptation authority.

### 3. Stability and repeated-case evidence

Single held-out gains do not establish a robust mechanism. The next layer asks whether results survive source perturbation and independent cases:

- `stability.py`: trajectory-block bootstrap stability for E002;
- `reliability.py`: participant-primary repeated-case recurrence and ICC only when the panel identifies it;
- `inference.py`: participant-level bootstrap intervals and sign-flip inference.

Trials and windows are not promoted to independent participant-level units.

### 4. BMRB evidence profiles

The Brain Mechanism Recapitulation Benchmark is the main project-level abstraction:

- `recapitulation.py`: evidence tiers, gates, monotonic promotion, and mechanism-necessity profiles;
- `bmrb.py`: dynamics aggregation;
- `representation_conservation.py`, `representation_studies.py`, `bmrb_representation.py`: exact-paired cross-representation evidence;
- `causal_recapitulation.py`, `matched_recovery.py`, `bmrb_causal.py`: intervention/ablation necessity and matched classical recovery;
- `neuros_mechint_artifacts.py`: dependency-light verification of compatible mechanistic evidence artifacts.

BMRB does not produce one synthetic "brain score." Failed gates remain visible and scientifically useful.

## Evidence ladder

```text
0  descriptive
1  predictive
2  adversary_surviving
3  source_stability
4  repeated_case
5  causal_mechanistic
6  physical_quantum
```

Promotion is monotonic. Later impressive evidence cannot erase an earlier falsifier. A physical-quantum claim is separately gated and cannot be inferred from predictive fit, density notation, contextual mathematics, or a causal model alone.

## Confirmatory research layer

`confirmatory_representation.py` and `confirmatory_cli.py` provide the stricter confirmatory BMRB-Representation surface. A confirmatory policy fixes before final evaluation:

- the primary calibration budget;
- the primary classical comparator;
- effect and ablation thresholds;
- participant/representation coverage requirements;
- conservation/adversary criteria;
- inference settings;
- the exact machine-readable policy fingerprint.

Secondary calibration budgets remain descriptive. They are not averaged into the primary estimand.

An external preregistration record is **bound**, not authenticated, by local code. Reviewers must still independently verify the external registration itself.

## Known-truth validation

`bmrb_validation.py` and `bmrb_validation_stress.py` attack the production confirmatory evaluator with declared synthetic truth. The validation layer deliberately calls the same production decision machinery rather than maintaining a second easier implementation.

Current scenarios include effect nulls, classical-equivalence nulls, shared positives, predictive shortcuts, representation-specific failures, calibration reversals, stronger heterogeneity, repeated sessions, and exact-pair deletion.

Synthetic qualification validates behavior under the declared data-generating mechanisms. It is not a certificate of biological truth.

## Integration boundary with neurOS

`integrations/neuros.py` provides the optional runtime seam:

- `DensityGeometryTransform` exposes QuantumBCI representations to neurOS-compatible pipelines;
- `NeurOSFoundationEncoder` wraps runnable frozen foundation-model adapters without adding heavy dependencies to the base package;
- evidence binding records raw/upstream dataset identity separately from neurOS partition/split identity.

Optional integrations fail closed. An unavailable foundation model must not silently become random or placeholder output.

## Public interfaces

The package exposes several installed commands:

```text
quantumbci
quantumbci-audit
quantumbci-kumar2024
quantumbci-bmrb
quantumbci-confirmatory
quantumbci-bmrb-validate
```

The root Python namespace is broader than the eventual stable API. `quantumbci.api_contract` defines the smaller pre-1.0 compatibility-candidate surface. See `docs/API_STABILITY.md`.

## Artifact design rules

Promotion-grade artifacts should bind the identities needed to reconstruct what was actually tested, including where relevant:

- source/dataset fingerprints;
- participant/session/case identity;
- split/calibration/evaluation authority;
- model/checkpoint/revision identity;
- representation identity;
- policy/preregistration identity;
- upstream evidence fingerprints;
- source code revisions;
- output artifact fingerprints or SHA-256 ledgers.

A SHA-256 ledger establishes content integrity, not authorship or external authenticity.

## Dependency policy

The base package depends only on NumPy. Heavy or ecosystem-specific capabilities remain optional extras:

- `quantum`: Qiskit execution;
- `neuros`: neurOS runtime/foundation integration;
- `real-eeg`: real EEG study dependencies;
- `neuros-mechint`: mechanistic intervention interoperability.

Do not add a heavyweight dependency merely to simplify one benchmark implementation. Prefer small typed/adapted boundaries and independently installable integrations.

## Development rule

New work should answer one of four questions before expanding the package:

1. Does this add a scientifically distinct mechanism or adversary?
2. Does this strengthen evidence authority, reproducibility, or falsification?
3. Does this validate the benchmark itself on known truth or independent data?
4. Does this improve production reliability without weakening scientific contracts?

Feature count is not an objective. A simpler system that exposes a falsifier more clearly is an architectural improvement.
