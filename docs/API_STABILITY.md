# API stability

QuantumBCI is still a pre-1.0 research package. The package therefore distinguishes **availability** from **compatibility commitment**.

The root `quantumbci` namespace intentionally remains broad so research notebooks can import commonly used mechanisms and evidence objects conveniently. A name being importable from the root does **not** mean it is permanently stable.

## Compatibility-candidate surface

`quantumbci.api_contract.COMPATIBILITY_CANDIDATE_ROOT_API` defines the smaller surface that should not be removed or renamed casually during the remaining alpha period. CI verifies that every listed symbol remains available from the root package and in `quantumbci.__all__`.

The initial candidate surface covers:

- claim contracts;
- the principal E001 benchmark entry point;
- mathematical equivalence auditing;
- trajectory evidence authority;
- BMRB evidence-tier/profile primitives;
- high-level dynamics, causal, representation-conservation, and stability entry points.

This is deliberately smaller than the complete root namespace.

## Pre-1.0 change policy

Before 1.0:

1. research-stage modules may still evolve when scientific correctness requires it;
2. compatibility-candidate names should receive a documented migration path before removal or incompatible renaming;
3. machine-readable artifact changes require an explicit schema-version decision and should fail closed on incompatible evidence;
4. CLI behavior used in qualified workflows should be treated as a compatibility surface even when the underlying Python helper remains experimental;
5. a scientific bug fix may require an incompatible change rather than preserving an incorrect result, but the release notes must make that break explicit.

## Experimental APIs

Any public object not included in the compatibility-candidate list should currently be treated as research-stage unless its module documentation says otherwise. This includes many low-level dynamics/model objects, serialization helpers, and study-specific implementation details.

Experimental does not mean untested. Many experimental surfaces are covered by strict scientific contracts. It means their naming and composition may still change before 1.0.

## Toward 1.0

A 1.0 candidate should replace this pre-1.0 contract with a documented stable API and deprecation policy. That transition should be based on real BMRB usage rather than freezing every historical helper simply because it happened to be exported during research development.
