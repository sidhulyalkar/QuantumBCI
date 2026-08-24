"""QuantumBCI: falsifiable quantum and quantum-inspired neural modelling."""

from .benchmarking import (
    BenchmarkMetrics,
    DensityBenchmarkResult,
    E001RepresentationBenchmarkResult,
    IndexSplit,
    benchmark_density_embeddings,
    benchmark_e001_embeddings,
)
from .claims import ClaimClass, MechanismCard, mechanism_card
from .contextuality import commutator_norm, order_effect, projector
from .equivalence import (
    BatchEquivalenceAudit,
    DensityCovarianceAudit,
    audit_density_covariance_equivalence,
    audit_embedding_batch,
    trace_normalized_second_moment,
)
from .exporting import (
    export_run_bids_derivative_container,
    export_run_ro_crate,
    verify_run_artifacts,
)
from .kalman import QLSADiagnostics, kalman_filter, qlsa_diagnostics
from .longitudinal import (
    LongitudinalE001CaseResult,
    LongitudinalE001Row,
    PairedBootstrapSummary,
    evaluate_density_information_gate,
    paired_participant_bootstrap,
    run_longitudinal_e001_case,
)
from .open_system import dephasing_collapse, evolve_lindblad, lindblad_rhs
from .recipes import FrozenEmbeddingRecipe, RecipeRunResult, load_recipe, run_recipe
from .spectral import amplitude_encode, classical_fft, qft_probabilities, qft_state
from .states import density_from_samples, l1_coherence, purity, von_neumann_entropy

__all__ = [
    "BatchEquivalenceAudit",
    "BenchmarkMetrics",
    "ClaimClass",
    "DensityBenchmarkResult",
    "DensityCovarianceAudit",
    "E001RepresentationBenchmarkResult",
    "FrozenEmbeddingRecipe",
    "IndexSplit",
    "LongitudinalE001CaseResult",
    "LongitudinalE001Row",
    "MechanismCard",
    "PairedBootstrapSummary",
    "QLSADiagnostics",
    "RecipeRunResult",
    "amplitude_encode",
    "audit_density_covariance_equivalence",
    "audit_embedding_batch",
    "benchmark_density_embeddings",
    "benchmark_e001_embeddings",
    "classical_fft",
    "commutator_norm",
    "dephasing_collapse",
    "density_from_samples",
    "evaluate_density_information_gate",
    "evolve_lindblad",
    "export_run_bids_derivative_container",
    "export_run_ro_crate",
    "kalman_filter",
    "l1_coherence",
    "lindblad_rhs",
    "load_recipe",
    "mechanism_card",
    "order_effect",
    "paired_participant_bootstrap",
    "projector",
    "purity",
    "qft_probabilities",
    "qft_state",
    "qlsa_diagnostics",
    "run_longitudinal_e001_case",
    "run_recipe",
    "trace_normalized_second_moment",
    "verify_run_artifacts",
    "von_neumann_entropy",
]
