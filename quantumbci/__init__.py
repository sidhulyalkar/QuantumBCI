"""QuantumBCI: falsifiable quantum and quantum-inspired neural modelling."""

from .benchmarking import (
    BenchmarkMetrics,
    DensityBenchmarkResult,
    IndexSplit,
    benchmark_density_embeddings,
)
from .claims import ClaimClass, MechanismCard, mechanism_card
from .contextuality import commutator_norm, order_effect, projector
from .exporting import (
    export_run_bids_derivative_container,
    export_run_ro_crate,
    verify_run_artifacts,
)
from .kalman import QLSADiagnostics, kalman_filter, qlsa_diagnostics
from .open_system import dephasing_collapse, evolve_lindblad, lindblad_rhs
from .recipes import FrozenEmbeddingRecipe, RecipeRunResult, load_recipe, run_recipe
from .spectral import amplitude_encode, classical_fft, qft_probabilities, qft_state
from .states import density_from_samples, l1_coherence, purity, von_neumann_entropy

__all__ = [
    "BenchmarkMetrics",
    "ClaimClass",
    "DensityBenchmarkResult",
    "FrozenEmbeddingRecipe",
    "IndexSplit",
    "MechanismCard",
    "QLSADiagnostics",
    "RecipeRunResult",
    "amplitude_encode",
    "benchmark_density_embeddings",
    "classical_fft",
    "commutator_norm",
    "dephasing_collapse",
    "density_from_samples",
    "evolve_lindblad",
    "export_run_bids_derivative_container",
    "export_run_ro_crate",
    "kalman_filter",
    "l1_coherence",
    "lindblad_rhs",
    "load_recipe",
    "mechanism_card",
    "order_effect",
    "projector",
    "purity",
    "qft_probabilities",
    "qft_state",
    "qlsa_diagnostics",
    "run_recipe",
    "verify_run_artifacts",
    "von_neumann_entropy",
]
