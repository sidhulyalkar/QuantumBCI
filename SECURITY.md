# Security Policy

QuantumBCI is research software that processes scientific evidence artifacts and may be used in larger neuroscience workflows. Security reports should distinguish software-security vulnerabilities from scientific-validity concerns; both matter, but they follow different review paths.

## Supported versions

Until QuantumBCI reaches a stable 1.0 API, security fixes are applied to the current `main` line and the most recent qualified release boundary. Older alpha releases may not receive backports.

## Reporting a vulnerability

Please do **not** publish exploit details, credentials, private data, or a working proof of concept in a public issue.

1. Prefer GitHub's private vulnerability-reporting / security-advisory flow for this repository when it is available.
2. If a private reporting flow is unavailable, open a minimal public issue requesting private maintainer contact **without** including exploitable technical details.
3. Include the affected QuantumBCI version or commit, environment, impact, reproduction conditions, and the smallest safe evidence needed to reproduce the problem.

Reports involving a dependency should include the dependency name/version and, when known, the upstream advisory identifier.

## Scope

Security-relevant reports include, for example:

- arbitrary code execution or command injection;
- unsafe archive/path handling or unintended file overwrite;
- credential or secret exposure;
- dependency vulnerabilities affecting a documented QuantumBCI execution path;
- untrusted artifact parsing that can escape the declared output/input boundary;
- integrity-check bypasses that allow a modified evidence artifact to be accepted as the original artifact.

Scientific issues such as data leakage, invalid inference units, incorrect equivalence claims, or over-broad mechanistic interpretation are also important, but normally belong in the public issue tracker unless disclosure would expose sensitive data or create a software-security risk.

## Sensitive data

Do not attach private EEG/BCI data, participant identifiers, credentials, access tokens, or proprietary model checkpoints to a report. Use synthetic/minimized reproductions whenever possible.

## Disclosure and fixes

A vulnerability fix should preserve QuantumBCI's fail-closed scientific contracts. Security remediation must not silently weaken artifact verification, evidence authority, preregistration binding, matched-control requirements, or claim ceilings merely to make a failing workflow pass.
