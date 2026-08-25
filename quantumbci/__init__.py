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
from .dynamics_equivalence import (
    BlochAffineGenerator,
    LindbladAffineEquivalenceAudit,
    LindbladGaugeAudit,
    affine_rhs,
    audit_lindblad_gauge_nonidentifiability,
    audit_qubit_lindblad_affine_equivalence,
    bloch_to_density,
    compile_qubit_lindblad_to_affine,
    density_to_bloch,
    evolve_affine_bloch,
)
from .e002_synthetic import (
    CANONICAL_STRUCTURE_RESIDUAL_MAX,
    CLASSICAL_ADVERSARY_RESIDUAL_MIN,
    CanonicalQubitParameters,
    canonical_qubit_model,
    canonical_structure_residual,
    fit_affine_generator_from_trajectories,
    recover_canonical_parameters,
    run_e002_synthetic_recovery_grid,
    simulate_canonical_bloch_trajectories,
)
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
from .trajectory_authority import (
    TrajectoryEvidenceAuthority,
    TrajectoryEvidenceData,
    load_trajectory_contract_descriptor,
)

__all__ = [
    "BatchEquivalenceAudit",
    "BenchmarkMetrics",
    "BlochAffineGenerator",
    "CANONICAL_STRUCTURE_RESIDUAL_MAX",
    "CLASSICAL_ADVERSARY_RESIDUAL_MIN",
    "CanonicalQubitParameters",
    "ClaimClass",
    "DensityBenchmarkResult",
    "DensityCovarianceAudit",
    "E001RepresentationBenchmarkResult",
    "FrozenEmbeddingRecipe",
    "IndexSplit",
    "LindbladAffineEquivalenceAudit",
    "LindbladGaugeAudit",
    "LongitudinalE001CaseResult",
    "LongitudinalE001Row",
    "MechanismCard",
    "PairedBootstrapSummary",
    "QLSADiagnostics",
    "RecipeRunResult",
    "TrajectoryEvidenceAuthority",
    "TrajectoryEvidenceData",
    "affine_rhs",
    "amplitude_encode",
    "audit_density_covariance_equivalence",
    "audit_embedding_batch",
    "audit_lindblad_gauge_nonidentifiability",
    "audit_qubit_lindblad_affine_equivalence",
    "benchmark_density_embeddings",
    "benchmark_e001_embeddings",
    "bloch_to_density",
    "canonical_qubit_model",
    "canonical_structure_residual",
    "classical_fft",
    "commutator_norm",
    "compile_qubit_lindblad_to_affine",
    "dephasing_collapse",
    "density_from_samples",
    "density_to_bloch",
    "evaluate_density_information_gate",
    "evolve_affine_bloch",
    "evolve_lindblad",
    "export_run_bids_derivative_container",
    "export_run_ro_crate",
    "fit_affine_generator_from_trajectories",
    "kalman_filter",
    "l1_coherence",
    "lindblad_rhs",
    "load_recipe",
    "load_trajectory_contract_descriptor",
    "mechanism_card",
    "order_effect",
    "paired_participant_bootstrap",
    "projector",
    "purity",
    "qft_probabilities",
    "qft_state",
    "qlsa_diagnostics",
    "recover_canonical_parameters",
    "run_e002_synthetic_recovery_grid",
    "run_longitudinal_e001_case",
    "run_recipe",
    "simulate_canonical_bloch_trajectories",
    "trace_normalized_second_moment",
    "verify_run_artifacts",
    "von_neumann_entropy",
]
