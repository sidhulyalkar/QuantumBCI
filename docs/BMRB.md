# Brain Mechanism Recapitulation Benchmark (BMRB)

QuantumBCI should not answer the vague question “is the brain quantum?” BMRB asks a narrower and falsifiable question:

> **What computational structure is required to reproduce a declared neural signature, which alternatives have been ruled out, how stable and repeatable is the evidence, and what is the strongest claim the data permit?**

A classical model falsifying a quantum-inspired mechanism is a successful benchmark result.

## v0.15 scope

v0.15 makes the first `BMRB_DYNAMICS_V1` bundle executable. It consumes independently qualified E002 v0.14 case artifacts and adds:

- participant-primary hierarchical bootstrap inference;
- population mechanism recurrence across participants;
- within-participant repeated-occasion variation;
- balanced-panel ICC(A,1) when, and only when, the repeated-case design identifies it;
- a machine-readable mechanism-necessity evidence ladder;
- separate **evidence coverage** and **promotion ceiling** fields;
- standalone JSON and HTML reports;
- an installed `quantumbci-bmrb` CLI.

It does **not** yet implement causal intervention/ablation promotion. That tier remains visibly `not_run`.

## Why recurrence and ICC are different

Suppose every participant has `gamma_relaxation ≈ 0.25` at every session.

That can be excellent **population recurrence** because the parameter has the same direction and similar magnitude across people. Yet the ICC may be small because there is almost no real between-person variance to preserve.

Conversely, an ICC can be high when people have stable but very different values, even if the population mean is scientifically uninteresting.

BMRB therefore reports both surfaces:

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

Neither surface is turned into a universal pass/fail threshold in v0.15.

## Case manifest

Create a JSON file pointing to qualified `bootstrap_stability.json` artifacts:

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

Artifact paths are resolved relative to the manifest file. Each artifact is hashed before analysis. BMRB v1 accepts exactly one E002 stability artifact per participant/occasion pair.

## Run it

```bash
quantumbci-bmrb dynamics cases.json \
  --output-dir bmrb-dynamics \
  --resamples 5000 \
  --seed 1501
```

Outputs:

```text
bmrb-dynamics/
  bmrb_dynamics.json
  report.html
```

To audit only selected mechanism quantities:

```bash
quantumbci-bmrb dynamics cases.json \
  --estimate gamma_dephasing \
  --estimate gamma_relaxation \
  --output-dir bmrb-dynamics
```

## Default E002 reliability quantities

The first BMRB-Dynamics bundle audits:

- `omega_x`
- `omega_z`
- `gamma_dephasing`
- `gamma_relaxation`
- `canonical_structure_residual`
- `canonical_minus_affine_one_step_rmse`
- `canonical_minus_affine_rollout_rmse`
- `direct_minus_nonlinear_mean_nll`
- `direct_minus_nonlinear_one_step_rmse`

These mix mechanism parameters and predictive comparison quantities deliberately, but each is analyzed on its own scale. BMRB never pools them into one synthetic “brain score.”

## Hierarchical bootstrap

The inference unit is the participant.

For each bootstrap replicate:

1. participants are resampled with replacement;
2. within each sampled participant, that participant's declared occasions/cases are resampled with replacement;
3. a participant mean is computed;
4. participant means are averaged with equal participant weighting.

This avoids treating correlated sessions as independent people.

The method ID is:

```text
participant_primary_hierarchical_bootstrap_v1
```

## ICC contract

BMRB computes a two-way random-effects, single-measure, absolute-agreement ICC only when all participants share the same complete occasion set.

Method ID:

```text
icc_a1_two_way_random_absolute_agreement_balanced_v1
```

If the panel is incomplete or unbalanced, the artifact preserves population recurrence results but serializes:

```json
{
  "icc": null,
  "icc_unavailable_reason": "..."
}
```

It does not impute missing cells or silently switch to another reliability statistic.

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

Each tier has a status:

```text
not_run
characterized
pass
fail
not_applicable
```

A `pass` requires an explicit decision rule. Evidence without a preregistered threshold is `characterized`, not `pass`.

This yields two different summary fields:

- `evidence_coverage_tier`: how far the study has actually measured;
- `promotion_ceiling`: highest contiguous tier with explicit PASS decisions.

A study can therefore have repeated-case evidence while still being blocked at a lower adversary gate.

## Current E002 interpretation

The current E002 chain deliberately exposes `dynamical_information_novel` after affine, observed-state, probabilistic, switching and nonlinear predictive adversaries.

For BMRB-Dynamics v1:

- if every case reports `dynamical_information_novel=false`, the matched-classical-adversary gate is a **falsification**;
- if at least one case reports surviving information, the cross-case gate remains `characterized` until a preregistered pooled decision rule exists;
- source bootstrap and repeated-case reliability remain `characterized` without universal thresholds;
- causal intervention/ablation remains `not_run`;
- physical-quantum evidence is `not_applicable` for the quantum-inspired E002 model.

This is intentionally conservative. The report is designed to make a negative result informative instead of encouraging claim inflation.

## Python API

```python
from quantumbci import build_bmrb_dynamics_bundle, write_bmrb_dynamics_bundle

bundle = build_bmrb_dynamics_bundle(
    "cases.json",
    n_resamples=5000,
    seed=1501,
)
write_bmrb_dynamics_bundle(bundle, "bmrb-dynamics")

print(bundle.profile.evidence_coverage_tier)
print(bundle.profile.promotion_ceiling)
print(bundle.profile.first_failing_gate)
```

Lower-level APIs are public too:

- `RepeatedCaseEstimate`
- `audit_repeated_case_estimate`
- `audit_repeated_case_reliability`
- `ICCResult`
- `RecapitulationSignature`
- `EvidenceGate`
- `MechanismNecessityProfile`

## neurOS and neuros-mechint boundary

The intended division remains:

```text
neurOS
  dataset / participant / session authority
  frozen neural and foundation-model representations
        ↓
QuantumBCI
  mechanism candidates
  matched predictive adversaries
  source stability
  repeated-case reliability
  mechanism-necessity profile
        ↓
neuros-mechint
  intervention / ablation / dose-response / faithfulness evidence
        ↓
BMRB causal promotion
```

QuantumBCI must not require neurOS to depend on it. The evidence handoff should remain one-way and explicit.

## Next promotion layer

The next BMRB-Dynamics stage should consume qualified v0.15 reliability plus neuros-mechint intervention evidence and test:

1. predicted intervention direction;
2. effect magnitude calibration;
3. dose-response monotonicity or preregistered shape;
4. mechanism ablation loss;
5. recovery by matched classical alternatives;
6. participant-level replication of the intervention effect.

Only after those gates pass can `necessity_claim_permitted` become true for a quantum-inspired mechanism. Physical-quantum language remains separately gated by an independent physical witness protocol.
