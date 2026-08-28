# Release process

QuantumBCI treats a release as a qualified scientific-software boundary, not merely a version-number change.

## Release candidate requirements

Before tagging a release:

1. merge only exact-head-qualified changes into `main`;
2. ensure `pyproject.toml` and `CITATION.cff` report the same version;
3. require the generic test matrix, relevant scientific contracts, and the package-quality contract to pass on the release commit;
4. review the changelog for scientific claim changes, schema/API compatibility changes, dependency changes, and known limitations;
5. verify that any changed machine-readable artifact format has an explicit schema-version/compatibility decision;
6. ensure no release note upgrades a scientific claim beyond the evidence gates implemented by the package.

## Package-quality artifacts

The `Package quality contract` builds both sdist and wheel, validates metadata, installs each artifact in a clean virtual environment, exercises the installed CLI surface, runs `pip check`, and emits SHA-256 checksums.

Those workflow artifacts are qualification evidence, but they are not yet a permanent GitHub Release or package-registry provenance record.

## Tagging

Qualified releases should use a version tag of the form:

```text
vX.Y.Z
```

The tag must point at the exact qualified `main` commit whose package metadata reports `X.Y.Z`.

Do not move a published version tag to a different commit. If a released boundary is wrong, publish a new patch version and document the correction.

## GitHub Release

For each qualified version, create a GitHub Release from the immutable version tag and include:

- concise release notes;
- major scientific/engineering changes;
- compatibility or schema changes;
- important negative/falsification findings where relevant;
- known limitations and claim ceilings;
- the qualified sdist and wheel;
- `SHA256SUMS`.

At the time this document was introduced, GitHub Release creation remains a maintainer action rather than an automated publish step.

## Package registry publication

If QuantumBCI is published to PyPI, prefer GitHub/PyPI trusted publishing over a long-lived repository secret. Publication automation should be added only after the repository's protected-branch/ruleset policy is active.

A publishing workflow must consume artifacts built from the release commit rather than rebuilding different bytes after approval whenever practical.

## Supply-chain roadmap

Before a production-grade 1.0 candidate, add:

- protected `main` / required checks;
- release provenance or attestations;
- a machine-readable SBOM for release artifacts;
- dependency vulnerability auditing with a documented triage policy;
- a stable public API and deprecation policy.

These controls improve software trust. They do not strengthen the biological interpretation of BMRB results.
