# Changelog

All notable user-facing changes to QuantumBCI are recorded here. Scientific claim boundaries, artifact/schema changes, and compatibility changes should be called out explicitly rather than hidden inside implementation notes.

## Unreleased

### Scientific validation

- Add a frozen BMRB operating-characteristics study layer that reuses the production known-truth/confirmatory evaluator over predeclared participant, effect, heterogeneity, and measurement-noise grids.
- Separate development and final-evaluation Monte Carlo seed authority with fingerprinted, non-overlapping deterministic partitions.
- Report per-cell decision error, Monte Carlo standard error, Wilson pass-rate intervals, failure localization, reference-effect bias/RMSE, and participant-bootstrap interval coverage without defining universal biological qualification thresholds.
- Add a preregisterable BMRB final-evaluation seal that binds development evidence, the exact future evaluation policy, explicit caller-supplied acceptance bounds, multiplicity policy, and external preregistration before evaluation seeds may be observed.
- Require explicit aggregate false-promotion/known-positive recovery criteria plus exact-equivalence, predictive-shortcut, and shared-mechanism scenario criteria without supplying universal numeric thresholds.
- Add participant-dependence stress DGMs that force unequal session counts and responder mixtures to make row-pooled summaries point in the wrong direction while the production BMRB participant-level estimand remains authoritative.
- Distinguish structured missing representation pairs as software-invalid evidence that must fail closed, rather than silently converting them into a negative mechanism result.

### Engineering

- Qualify and adopt current major GitHub Actions runtimes for checkout, Python setup, and artifact upload across the inherited workflow matrix.
- Add a clean sdist/wheel package-quality qualification contract.
- Define a smaller pre-1.0 root API compatibility-candidate surface.
- Add an executable static-debt ratchet that prevents broad-exception and typing-suppression debt from silently spreading.
- Refresh architecture, release-process, API-stability, and code-quality documentation around BMRB.
- Add structured issue templates for software defects, scientific-validity concerns, and research proposals.
- Add fail-closed round-trip verification for final-evaluation seal artifacts and keep the seal surface physically separate from final-evaluation execution.
- Remove invalid-escape warnings from mechanistic artifact regex tests without changing test semantics.
- Extend the known-truth validation workflow to qualify participant-dependence stress tests, claim boundaries, documentation, and installed wheel contents.

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
