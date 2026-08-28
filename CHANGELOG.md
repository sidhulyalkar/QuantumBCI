# Changelog

All notable user-facing changes to QuantumBCI are recorded here. Scientific claim boundaries, artifact/schema changes, and compatibility changes should be called out explicitly rather than hidden inside implementation notes.

## Unreleased

### Engineering

- Add a clean sdist/wheel package-quality qualification contract.
- Define a smaller pre-1.0 root API compatibility-candidate surface.
- Refresh architecture and release-process documentation around BMRB.
- Add structured issue templates for software defects, scientific-validity concerns, and research proposals.

## 0.19.0 - 2026-08-28

### Scientific validation

- Add known-truth BMRB validation against declared positive, null, shortcut, representation-specific, calibration-reversal, heterogeneity, repeated-session, and exact-pairing scenarios.
- Keep validation downstream of the production confirmatory evaluator so decision-semantic regressions are visible.

### Confirmatory research

- Carry forward the v0.18 preregistration-bound confirmatory representation policy, fixed primary calibration estimand, preregistered comparator selection, participant-level uncertainty, and explicit promotion/claim ceilings.

### Repository governance

- Add `SECURITY.md`, Dependabot configuration, CODEOWNERS, and a scientific-authority-aware pull-request template after the v0.12-v0.19 research stack was merged.

### Claim boundary

- Known-truth qualification validates software behavior only under declared synthetic data-generating mechanisms. It does not validate biological truth or authorize a physical-quantum interpretation.
