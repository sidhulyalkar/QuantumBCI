# Brain Mechanism Recapitulation Benchmark (BMRB)

QuantumBCI should not answer the vague question “is the brain quantum?” BMRB asks a narrower and falsifiable question:

> **What computational structure is required to reproduce a declared neural signature, which alternatives have been ruled out, how stable and causal is the evidence, and what is the strongest claim the data actually permit?**

A classical model falsifying a quantum-inspired mechanism is a successful benchmark result.

## Current scope

v0.15 introduced executable `BMRB_DYNAMICS_V1` repeated-case evidence. v0.16 adds the causal-mechanistic handoff and chain-of-custody needed to use those reports as inputs to stronger claims.

The current path is:

```text
qualified E002 case artifacts
        ↓
BMRB-Dynamics
  predictive/adversary evidence
  source stability
  population recurrence
  repeated-case reliability
        ↓
verified neuros-mechint evidence
  intervention direction
  dose response
  held-out faithfulness
  ablation necessity
        +
fingerprinted matched-classical recovery
        ↓
BMRB causal necessity
  participant-balanced causal evidence
  monotonic promotion decision
        ↓
physical quantum remains separately locked
```

## Evidence ladder

`MechanismNecessityProfile` uses these tiers:

```text
0 descriptive
1 predictive
2 adversary_surviving
3 source_stability
4 repeated_case
5 causal_mechanistic
6 physical_quantum
```

Each tier has one status:

```text
not_run
characterized
pass
fail
not_applicable
```

A `pass` requires an explicit decision rule. Evidence without a preregistered threshold is `characterized`, not `pass`.

Two summary fields remain intentionally different:

- `evidence_coverage_tier`: how far the study has actually measured;
- `promotion_ceiling`: the highest contiguous tier with explicit PASS decisions.

A later result can never jump an unresolved or failed earlier tier. A later FAIL is still allowed to act as an independent falsifier.

## Repeated-case evidence

BMRB separates population recurrence from reproducibility of individual differences.

```text
population recurrence
  participant-weighted grand mean
  participant sign consistency
  hierarchical-bootstrap interval
  bootstrap probability positive

person-specific reliability
  within-participant SD
  between-participant mean SD
  ICC(A,1), only for complete balanced panels
```

A quantity can recur consistently across everyone while having a low or undefined ICC if there is little between-person variance. BMRB does not convert that distinction into a cosmetic single score.

### Hierarchical bootstrap

The inference unit is the participant. Each bootstrap replicate:

1. resamples participants with replacement;
2. resamples declared occasions within each sampled participant;
3. computes one mean per sampled participant;
4. averages participant means with equal participant weight.

Method ID:

```text
participant_primary_hierarchical_bootstrap_v1
```

### ICC contract

BMRB computes a two-way random-effects, single-measure, absolute-agreement ICC only when all participants share one complete balanced occasion panel.

```text
icc_a1_two_way_random_absolute_agreement_balanced_v1
```

Incomplete, unbalanced, or numerically degenerate panels preserve population recurrence but report ICC as unavailable instead of inventing another statistic.

## BMRB-Dynamics case manifest

```json
{
  "schema_version": 1,
  "study_id": "my-e002-cohort",
  "metadata": {
    "dataset": "example",
    "split_policy": "participant-longitudinal"
  },
  "cases": [
    {
      "participant_id": "sub-01",
      "occasion_id": "ses-01",
      "case_id": "sub-01_ses-01",
      "artifact": "artifacts/sub-01_ses-01/bootstrap_stability.json"
    },
    {
      "participant_id": "sub-01",
      "occasion_id": "ses-02",
      "case_id": "sub-01_ses-02",
      "artifact": "artifacts/sub-01_ses-02/bootstrap_stability.json"
    },
    {
      "participant_id": "sub-02",
      "occasion_id": "ses-01",
      "case_id": "sub-02_ses-01",
      "artifact": "artifacts/sub-02_ses-01/bootstrap_stability.json"
    },
    {
      "participant_id": "sub-02",
      "occasion_id": "ses-02",
      "case_id": "sub-02_ses-02",
      "artifact": "artifacts/sub-02_ses-02/bootstrap_stability.json"
    }
  ]
}
```

Run:

```bash
quantumbci-bmrb dynamics cases.json \
  --output-dir bmrb-dynamics \
  --resamples 5000 \
  --seed 1501
```

v0.16 emits a self-verifying schema-v2 dynamics artifact with two different identities:

- `source_fingerprint` binds source identity, case SHA-256 values, reliability fingerprint and analysis settings;
- `artifact_fingerprint` binds the complete serialized scientific report.

An older v0.15 dynamics JSON remains a valid terminal report, but it must be regenerated with v0.16 before causal promotion because v0.15 did not serialize enough source identity for independent downstream verification.

## Default E002 quantities

BMRB-Dynamics currently audits each quantity on its own scale:

- `omega_x`
- `omega_z`
- `gamma_dephasing`
- `gamma_relaxation`
- `canonical_structure_residual`
- `canonical_minus_affine_one_step_rmse`
- `canonical_minus_affine_rollout_rmse`
- `direct_minus_nonlinear_mean_nll`
- `direct_minus_nonlinear_one_step_rmse`

There is deliberately no synthetic “brain score.”

## v0.16 causal necessity

The causal tier asks more than whether an ablation hurts one model. It requires three evidence surfaces.

### 1. Intervention direction and dose response

QuantumBCI consumes the versioned neuros-mechint dose-response scientific result:

```text
neuros-mechint.dose-response-study.v1
```

The bridge verifies the native scientific fingerprint before using endpoint effect, monotonicity, or the original neuros-mechint pass decision.

### 2. Held-out faithfulness and ablation necessity

QuantumBCI consumes:

```text
neuros-mechint.evidence-pack.v1
```

The evidence-pack study fingerprint binds the frozen candidate, per-example candidate cases, policies and discovery/validation identities. Some aggregate fields are convenience summaries rather than direct fingerprint inputs, so BMRB recomputes the validation quantities it consumes from the fingerprint-bound cases and requires serialized aggregate/promotion summaries to agree.

This prevents a changed top-level necessity number from silently becoming causal authority.

### 3. Matched-classical recovery

A candidate is not necessary under BMRB merely because ablation causes a loss. The strongest declared classical alternative is given the same information/evidence budget and asked to recover that loss.

Recovery is a derived artifact:

```text
quantumbci.matched-classical-recovery.v1
```

For a higher-is-better metric:

```text
ablation_loss = baseline_metric - ablated_metric
restored_loss = max(0, recovered_metric - ablated_metric)
recovery_fraction = restored_loss / ablation_loss
```

For a lower-is-better metric the orientation is reversed consistently.

A valid recovery artifact records:

- participant, occasion and case identity;
- candidate mechanism ID;
- classical model ID;
- one declared `information_set_id`;
- metric and favorable direction;
- baseline, ablated and recovered values;
- candidate-evidence fingerprint;
- classical-evidence fingerprint;
- derived loss/restoration/recovery fields;
- deterministic source fingerprint.

A nonpositive candidate ablation loss is not a recovery experiment and fails closed. A classical model that makes the ablated result worse receives zero recovery rather than negative “credit.”

All recovery cases in one causal bundle must use the same declared information-set authority.

## Causal policy

`CausalNecessityPolicy` currently evaluates:

```text
minimum independent participants
minimum intervention-direction agreement
minimum dose-response pass fraction
minimum faithfulness pass fraction
minimum participant-balanced mean necessity
minimum joint random-control percentile
maximum matched-classical recovery fraction
```

The method ID is:

```text
participant_balanced_causal_necessity_v1
```

Two outcomes are distinct:

```text
scientific_criteria_passed
promotion_eligible
```

`scientific_criteria_passed` says the observed evidence satisfies the policy's numerical/decision criteria.

`promotion_eligible` additionally requires `policy.preregistered == true`.

A policy supplied after seeing the result can characterize evidence, but it cannot retroactively preregister itself.

## Participant balancing

Repeated sessions do not create extra people. Causal evidence is first aggregated within participant and only then averaged across participants with equal weight.

If one participant contributes five sessions and two others contribute one each, the first participant still contributes one third of the participant-level population summary, not five sevenths.

## Causal manifest

```json
{
  "schema_version": 1,
  "study_id": "my-causal-study",
  "upstream_bmrb": "bmrb-dynamics/bmrb_dynamics.json",
  "policy": {
    "policy_id": "my-preregistered-causal-policy-v1",
    "preregistered": true,
    "min_participants": 3,
    "min_direction_match_fraction": 0.8,
    "min_dose_response_pass_fraction": 0.8,
    "min_faithfulness_pass_fraction": 0.8,
    "min_mean_necessity_fraction": 0.5,
    "min_mean_joint_random_percentile": 0.95,
    "max_mean_classical_recovery_fraction": 0.25
  },
  "cases": [
    {
      "participant_id": "sub-01",
      "occasion_id": "ses-01",
      "case_id": "sub-01_ses-01",
      "dose_response_artifact": "causal/sub-01/dose_response.json",
      "faithfulness_artifact": "causal/sub-01/evidence_pack.json",
      "matched_recovery_artifact": "causal/sub-01/matched_recovery.json"
    }
  ]
}
```

The real manifest must contain enough independent participants to satisfy its declared policy.

Run:

```bash
quantumbci-bmrb causal causal-manifest.json \
  --output-dir bmrb-causal
```

Outputs:

```text
bmrb-causal/
  bmrb_causal.json
  report.html
```

The bundle records exact file SHA-256 values, native scientific fingerprints, the upstream BMRB source/artifact fingerprints, participant-balanced causal summaries, first falsifier, promotion ceiling and the physical-quantum lock.

## Current E002 interpretation

The current E002 evidence chain exposes `dynamical_information_novel` after affine, observed-state, probabilistic, switching and nonlinear predictive adversaries.

If every declared case reports `dynamical_information_novel=false`, the matched-classical-adversary tier is an explicit falsification. Even excellent later causal evidence cannot erase that earlier failure or jump the evidence ladder.

This is a feature, not an inconvenience. BMRB is designed so negative evidence remains scientifically visible.

## Python API

```python
from quantumbci import (
    build_bmrb_dynamics_bundle,
    write_bmrb_dynamics_bundle,
    build_bmrb_causal_bundle,
    write_bmrb_causal_bundle,
)

dynamics = build_bmrb_dynamics_bundle("cases.json")
write_bmrb_dynamics_bundle(dynamics, "bmrb-dynamics")

causal = build_bmrb_causal_bundle("causal-manifest.json")
write_bmrb_causal_bundle(causal, "bmrb-causal")

print(causal.causal_result.scientific_criteria_passed)
print(causal.causal_result.promotion_eligible)
print(causal.profile.promotion_ceiling)
print(causal.profile.first_failing_gate)
```

Lower-level public APIs include:

- `RepeatedCaseEstimate`
- `audit_repeated_case_reliability`
- `RecapitulationSignature`
- `MechanismNecessityProfile`
- `CausalNecessityPolicy`
- `evaluate_causal_necessity`
- `MatchedClassicalRecoveryEvidence`
- `build_matched_classical_recovery_evidence`
- `verify_bmrb_dynamics_mapping`
- `verify_dose_response_result`
- `verify_evidence_pack_result`

## neurOS / QuantumBCI / neuros-mechint boundary

```text
neurOS
  neural dataset authority
  participant/session/split authority
  frozen foundation-model representations
        ↓
QuantumBCI
  mechanism candidates
  matched predictive adversaries
  stability and repeated-case evidence
        ↓
neuros-mechint
  intervention / ablation / dose response / faithfulness
        ↓
QuantumBCI BMRB
  matched-classical recovery
  causal necessity
  evidence ledger and claim ceiling
```

The dependency remains one-way. neurOS does not depend on QuantumBCI, and the base QuantumBCI install does not require PyTorch merely to verify JSON causal artifacts.

## Physical-quantum boundary

A causal/mechanistic result can support a quantum-inspired mechanism under the declared benchmark. It still cannot establish a physical quantum substrate.

Tier 6 requires an independent protocol with, at minimum:

- a declared physical substrate;
- an operational witness;
- a detection floor;
- a discriminating perturbation;
- the strongest plausible classical mimic/control.

Model fit, causal ablation, density notation, contextual statistics, or a quantum circuit by themselves cannot satisfy that tier.

## Next benchmark expansion

After v0.16 qualification, the same evidence architecture should be generalized to `BMRB-Representation` on E001:

1. frozen raw/foundation-model embeddings under neurOS authority;
2. density/operator features versus covariance, tangent, bilinear, pooled and random-PSD controls;
3. cross-participant/session/dataset recurrence;
4. intervention of candidate representation components through neuros-mechint;
5. matched recovery by classical representation families;
6. raw-signal ↔ foundation-model mechanism conservation;
7. cross-foundation-model conservation.

Only after a neural computation survives that style of evidence ladder should E004 hardware/resource studies become a central research path.
