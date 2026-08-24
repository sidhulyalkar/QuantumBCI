"""QuantumBCI: falsifiable quantum and quantum-inspired neural modelling."""

from .claims import ClaimClass, MechanismCard, mechanism_card
from .contextuality import commutator_norm, order_effect, projector
from .kalman import QLSADiagnostics, kalman_filter, qlsa_diagnostics
from .open_system import dephasing_collapse, evolve_lindblad, lindblad_rhs
from .spectral import amplitude_encode, classical_fft, qft_probabilities, qft_state
from .states import density_from_samples, l1_coherence, purity, von_neumann_entropy

__all__ = [
    "ClaimClass",
    "MechanismCard",
    "QLSADiagnostics",
    "amplitude_encode",
    "classical_fft",
    "commutator_norm",
    "dephasing_collapse",
    "density_from_samples",
    "evolve_lindblad",
    "kalman_filter",
    "l1_coherence",
    "lindblad_rhs",
    "mechanism_card",
    "order_effect",
    "projector",
    "purity",
    "qft_probabilities",
    "qft_state",
    "qlsa_diagnostics",
    "von_neumann_entropy",
]
