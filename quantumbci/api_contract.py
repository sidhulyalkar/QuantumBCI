"""Pre-1.0 public API compatibility contract.

QuantumBCI intentionally keeps a broad root namespace for research convenience. That does not mean
every root export is stable. This module names the smaller compatibility-candidate surface that
should not be removed or renamed casually while the package remains alpha.
"""

from __future__ import annotations

from types import ModuleType

API_CONTRACT_VERSION = 1
API_STABILITY = "pre-1.0-compatibility-candidate"

COMPATIBILITY_CANDIDATE_ROOT_API: tuple[str, ...] = (
    "ClaimClass",
    "MechanismCard",
    "mechanism_card",
    "IndexSplit",
    "benchmark_e001_embeddings",
    "audit_density_covariance_equivalence",
    "TrajectoryEvidenceAuthority",
    "TrajectoryEvidenceData",
    "EvidenceTier",
    "GateStatus",
    "MechanismNecessityProfile",
    "build_bmrb_dynamics_bundle",
    "evaluate_causal_necessity",
    "evaluate_representation_conservation",
    "run_e002_bootstrap_stability",
)


def missing_compatibility_candidate_exports(module: ModuleType) -> tuple[str, ...]:
    """Return compatibility-candidate names absent from the supplied root module."""

    return tuple(name for name in COMPATIBILITY_CANDIDATE_ROOT_API if not hasattr(module, name))
