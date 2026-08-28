## Summary

<!-- What changed, and why is this the smallest appropriate change? -->

## Change class

- [ ] Numerical/scientific kernel
- [ ] Evidence authority, artifact, or schema contract
- [ ] Public Python API or CLI
- [ ] Optional integration
- [ ] CI/release/governance
- [ ] Documentation only

## Validation

<!-- List the exact tests/workflows/commands that qualify this head. Do not cite an older SHA. -->

- [ ] Relevant unit/adversarial tests pass
- [ ] Installed CLI/package surface checked when applicable
- [ ] Wheel/package contents checked when applicable
- [ ] Cross-repository integration checked when applicable

## Scientific authority and leakage

For changes that can affect research results:

- [ ] Fit/calibration/evaluation authority remains explicit and disjoint
- [ ] Final evaluation data cannot influence fitting, model/control selection, preprocessing, or adaptation
- [ ] Participant/case inference units are preserved
- [ ] Artifact/source fingerprints and tamper rejection remain intact
- [ ] Matched controls consume the declared comparable information set
- [ ] Not applicable to this PR

## Claim boundary

<!-- State the strongest claim this PR may support and the first important claim it still cannot support. -->

- [ ] No stronger claim language is introduced without a corresponding evidence gate
- [ ] Classical-equivalence/adversary failures remain visible rather than being averaged away
- [ ] Physical-quantum language remains separately gated
- [ ] Not applicable to this PR

## Compatibility and release surface

- [ ] Public API changes are deliberate and documented
- [ ] Machine-readable schema changes include an explicit version/compatibility decision
- [ ] Package version and `CITATION.cff` stay synchronized for release changes
- [ ] New dependencies are justified and remain optional when appropriate
- [ ] Not applicable to this PR

## Reviewer focus

<!-- What should a reviewer try hardest to falsify? -->
