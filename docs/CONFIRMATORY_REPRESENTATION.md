# Confirmatory BMRB-Representation v2

`BMRB_REPRESENTATION_CONFIRMATORY_V2` is the promotion-grade counterpart to the v1
cross-representation exploration/conservation surface.

It exists because four things must not be chosen after final evaluation:

1. the primary calibration budget;
2. the primary classical comparator;
3. the minimum effects/coverage required by the study;
4. the rule that turns evidence into a confirmatory PASS.

## Workflow

### 1. Write the decision policy before evaluating the final evidence

A policy has no universal scientific defaults. The study must explicitly declare its
minimum effect of interest, sample-size rationale, coverage, and conservation thresholds.

```json
{
  "policy_id": "kumar2024-bmrb-v2",
  "reference_representation_id": "raw",
  "primary_calibration_per_class": 10,
  "primary_classical_control": "normalized_covariance",
  "min_participants": 18,
  "min_representations": 3,
  "min_representation_families": 2,
  "min_candidate_advantage": 0.01,
  "min_ablation_necessity": 0.02,
  "min_reference_positive_fraction": 0.67,
  "min_all_lane_positive_fraction": 0.67,
  "min_direction_match_fraction": 0.67,
  "min_ablation_direction_match_fraction": 0.67,
  "min_information_novel_representation_fraction": 1.0,
  "sample_size_rationale": "Replace this example with a study-specific power/precision rationale.",
  "inference_seed": 1801,
  "bootstrap_resamples": 5000,
  "preregistration": null
}
```

The numerical values above are examples only. They are **not** QuantumBCI recommendations
or biological thresholds.

### 2. Fingerprint the exact policy

```bash
quantumbci-confirmatory policy-fingerprint policy.json
```

The output is the SHA-256 fingerprint that the external registration must bind. Changing a
budget, comparator, threshold, sample-size rationale, inference seed, or resampling count
changes the fingerprint.

### 3. Register externally

Use an externally timestamped, immutable registration system appropriate for the study.
The registered document should contain at least:

- hypotheses;
- primary and secondary outcomes;
- inclusion/exclusion rules;
- participant/dataset authority;
- primary calibration budget;
- model/checkpoint/layer/pooling choices or their predeclared selection rule;
- primary classical comparator;
- minimum effect(s) of interest;
- sample-size/power or precision rationale;
- statistical tests and multiplicity handling;
- contingencies/deviation policy;
- the QuantumBCI policy fingerprint.

After registration, add the external evidence record:

```json
"preregistration": {
  "artifact_role": "external_preregistration_evidence",
  "registration_uri": "https://<registry>/<immutable-registration>",
  "registered_at": "2026-08-01T12:30:00Z",
  "registration_document_sha256": "<sha256-of-registered-document>",
  "registered_policy_sha256": "<quantumbci-policy-fingerprint>",
  "registry": "<registry-name>"
}
```

QuantumBCI verifies that the supplied policy fingerprint matches the current decision
policy. It deliberately does not pretend that an offline Python process can authenticate
the external registry or its timestamp; publication/review should independently resolve
the URI.

Check the binding:

```bash
quantumbci-confirmatory policy-validate registered-policy.json
```

### 4. Build representation lanes under frozen evidence authority

Use the same v1 E001 representation-lane artifact format for raw, specialist, and
foundation-model representations. This preserves compatibility with qualified v0.17 lane
artifacts.

Every lane still needs exact participant/session/calibration/evaluation authority and
closed-world artifact verification.

### 5. Build a schema-v2 confirmatory manifest

```json
{
  "schema_version": 2,
  "study_id": "kumar2024-confirmatory-bmrb",
  "mechanism_id": "cross_feature_second_moment",
  "participant_key": "subject",
  "policy": {
    "policy_id": "kumar2024-bmrb-v2",
    "reference_representation_id": "raw",
    "primary_calibration_per_class": 10,
    "primary_classical_control": "normalized_covariance",
    "min_participants": 18,
    "min_representations": 3,
    "min_representation_families": 2,
    "min_candidate_advantage": 0.01,
    "min_ablation_necessity": 0.02,
    "min_reference_positive_fraction": 0.67,
    "min_all_lane_positive_fraction": 0.67,
    "min_direction_match_fraction": 0.67,
    "min_ablation_direction_match_fraction": 0.67,
    "min_information_novel_representation_fraction": 1.0,
    "sample_size_rationale": "<registered rationale>",
    "inference_seed": 1801,
    "bootstrap_resamples": 5000,
    "preregistration": {
      "artifact_role": "external_preregistration_evidence",
      "registration_uri": "https://<registry>/<registration>",
      "registered_at": "2026-08-01T12:30:00Z",
      "registration_document_sha256": "<sha256>",
      "registered_policy_sha256": "<policy-sha256>",
      "registry": "<registry>"
    }
  },
  "lanes": [
    {"lane_id": "raw", "artifact_dir": "lanes/raw"},
    {"lane_id": "labram", "artifact_dir": "lanes/labram"},
    {"lane_id": "eegpt", "artifact_dir": "lanes/eegpt"}
  ]
}
```

### 6. Run

```bash
quantumbci-confirmatory run confirmatory-manifest.json \
  --output-dir bmrb-confirmatory-representation
```

Output is a closed-world verified bundle containing:

```text
run.json
bmrb_confirmatory_representation.json
report.md
artifact_hashes.json
```

## Primary vs secondary estimands

Only `primary_calibration_per_class` contributes to the confirmatory effect and
cross-representation decision.

Every available calibration budget is still summarized in `calibration_frontier`, marked
`secondary_descriptive_frontier`. A reversal between 0-shot and 10-shot behavior therefore
remains visible rather than disappearing into an average.

## Classical control selection

The v2 loader reads the method named by `primary_classical_control` directly from each E001
benchmark artifact. It **does not** use the v1 `strongest_classical_control` field to choose
the confirmatory comparator.

For the current density constructor, `normalized_covariance` is the scientifically natural
primary comparator because exact equivalence is already known. A different comparator
requires a different registered policy.

## Participant-level uncertainty

For each representation lane at the primary budget, v2 reports:

- participant-balanced mean and median candidate advantage;
- participant positive fraction;
- percentile bootstrap confidence interval for the mean;
- deterministic two-sided sign-flip p-value;
- the same summaries for representation-ablation necessity.

The sign-flip test is exact for small participant counts and Monte Carlo for larger ones.
Neither the confidence interval nor p-value becomes a hidden promotion criterion. If a
study wants to gate on one, that rule must be written into a future registered policy
schema explicitly.

## Independent evidence decisions

The v2 result never compresses everything into one uninterpretable score. It records:

```text
effect_criteria_passed
adversary_survival_passed
conservation_criteria_passed
coverage_criteria_passed
scientific_criteria_passed
confirmatory_authority
promotion_eligible
```

This distinction matters. The current density mechanism can have:

```text
predictive primary effect: PASS
matched classical equivalence: FAIL
cross-representation conservation: CHARACTERIZED
replication coverage: CHARACTERIZED
```

That is a scientifically coherent negative result.

## What external preregistration evidence does and does not prove

A matching `PreregistrationEvidence` record establishes that the run bundle is bound to a
specific external registration reference and policy hash. It does not prove that:

- the external registry is authentic;
- the timestamp predates every data/model decision;
- the registered document was followed without deviations.

A manuscript should independently verify the registration URI and disclose deviations.

## Claim ceiling

This protocol remains `quantum_inspired` for the current QuantumBCI mechanisms.

A confirmatory BMRB PASS means only that the declared computational candidate survived the
registered evidence ladder under the specified neural/model evidence authority. It does
not establish microscopic quantum biology. Physical-quantum promotion remains a separate
protocol requiring an independently measurable substrate and witness.
