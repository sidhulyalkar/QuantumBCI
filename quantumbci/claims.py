"""Scientific claim contracts for QuantumBCI.

The central rule of this package is that a mathematical object is not evidence for a
physical quantum mechanism. Every implemented mechanism should say which kind of
claim it is capable of supporting and what would falsify that claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class ClaimClass(str, Enum):
    """The strongest claim an implementation is allowed to imply."""

    CLASSICAL_CONTROL = "classical_control"
    QUANTUM_INSPIRED = "quantum_inspired"
    QUANTUM_ALGORITHM = "quantum_algorithm"
    PHYSICAL_QUANTUM = "physical_quantum"


@dataclass(frozen=True)
class MechanismCard:
    """A compact, machine-readable hypothesis and falsification contract."""

    name: str
    claim_class: ClaimClass
    hypothesis: str
    observables: tuple[str, ...]
    classical_alternatives: tuple[str, ...]
    falsifiers: tuple[str, ...]

    def __post_init__(self) -> None:
        fields: Iterable[tuple[str, object]] = (
            ("name", self.name),
            ("hypothesis", self.hypothesis),
            ("observables", self.observables),
            ("classical_alternatives", self.classical_alternatives),
            ("falsifiers", self.falsifiers),
        )
        for field, value in fields:
            if not value:
                raise ValueError(f"MechanismCard.{field} must not be empty")


IMPLEMENTED_MECHANISMS: dict[str, MechanismCard] = {
    "density_geometry": MechanismCard(
        name="density_geometry",
        claim_class=ClaimClass.QUANTUM_INSPIRED,
        hypothesis=(
            "Representing a trace-normalized Hermitian second moment as a density operator "
            "may provide useful constrained normalization, spectral observables, or downstream "
            "operator inductive bias. The current constructor is information-equivalent to "
            "trace-normalized covariance and therefore cannot claim additional representation "
            "information from density notation alone."
        ),
        observables=("purity", "von_neumann_entropy", "l1_coherence"),
        classical_alternatives=(
            "trace-normalized covariance",
            "covariance/log-covariance geometry",
            "bilinear second moments",
            "PCA and linear probes",
        ),
        falsifiers=(
            "Exact normalized-covariance control reproduces the same representation/predictions",
            "Density-specific downstream constraints or observables add no reproducible held-out value",
            "Claimed observables are unstable across subjects, sessions, or resampling",
        ),
    ),
    "lindblad_latent_dynamics": MechanismCard(
        name="lindblad_latent_dynamics",
        claim_class=ClaimClass.QUANTUM_INSPIRED,
        hypothesis=(
            "Open-system dynamics provide an identifiable low-dimensional description "
            "of coupled neural latent modes and their loss of coherence."
        ),
        observables=("coupling response", "purity decay", "coherence decay"),
        classical_alternatives=("Kalman filter", "linear dynamical system", "neural ODE"),
        falsifiers=(
            "Matched classical state-space models explain held-out dynamics as well or better",
            "Hamiltonian/collapse parameters are non-identifiable across bootstrap fits",
        ),
    ),
    "contextual_measurement": MechanismCard(
        name="contextual_measurement",
        claim_class=ClaimClass.QUANTUM_INSPIRED,
        hypothesis=(
            "Non-commuting measurement operators compactly model reproducible context "
            "or order dependence in neural/cognitive readouts."
        ),
        observables=("commutator norm", "AB-vs-BA order effect"),
        classical_alternatives=("history-augmented logistic model", "HMM", "RNN"),
        falsifiers=(
            "Order effects disappear under preregistered replication",
            "A matched classical history model explains the effect without extra complexity",
        ),
    ),
    "qft_sampling": MechanismCard(
        name="qft_sampling",
        claim_class=ClaimClass.QUANTUM_ALGORITHM,
        hypothesis=(
            "A QFT circuit can transform an amplitude-encoded signal state and estimate "
            "selected Fourier-basis observables when state preparation and readout are accounted for."
        ),
        observables=("Fourier-basis measurement probabilities",),
        classical_alternatives=("FFT", "Goertzel transform"),
        falsifiers=(
            "End-to-end classical computation is faster at matched tolerance",
            "State preparation or sampling dominates the resource budget",
        ),
    ),
    "qlsa_resource_study": MechanismCard(
        name="qlsa_resource_study",
        claim_class=ClaimClass.QUANTUM_ALGORITHM,
        hypothesis=(
            "A quantum linear-system algorithm may estimate selected observables of a sparse, "
            "well-conditioned encoded system without reconstructing the full solution vector."
        ),
        observables=("selected solution observable", "resource ledger"),
        classical_alternatives=("dense solve", "sparse iterative solver"),
        falsifiers=(
            "The application requires a full classical solution vector",
            "Conditioning/state-preparation/readout erase the algorithmic advantage",
        ),
    ),
}


def mechanism_card(name: str) -> MechanismCard:
    try:
        return IMPLEMENTED_MECHANISMS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown mechanism {name!r}; choose from {sorted(IMPLEMENTED_MECHANISMS)}") from exc
