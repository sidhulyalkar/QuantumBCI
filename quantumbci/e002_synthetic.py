"""Gauge-fixed synthetic recovery benchmark for E002 open-system dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .dynamics_equivalence import (
    SIGMA_X,
    SIGMA_Z,
    audit_lindblad_gauge_nonidentifiability,
    audit_qubit_lindblad_affine_equivalence,
    compile_qubit_lindblad_to_affine,
    density_to_bloch,
)
from .open_system import evolve_lindblad
from .dynamics_equivalence import bloch_to_density

Array = np.ndarray
SIGMA_MINUS = np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=complex)


@dataclass(frozen=True)
class CanonicalQubitParameters:
    """Declared gauge-fixed E002 family used for parameter-recovery qualification.

    ``H = (omega_x sigma_x + omega_z sigma_z) / 2``

    The dissipators are a z-dephasing channel and amplitude relaxation toward
    ``|0>``. This is deliberately a small identifiable family, not a claim that
    these parameters correspond to microscopic neural quantum processes.
    """

    omega_x: float
    omega_z: float
    gamma_dephasing: float
    gamma_relaxation: float

    def __post_init__(self) -> None:
        values = (
            self.omega_x,
            self.omega_z,
            self.gamma_dephasing,
            self.gamma_relaxation,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("canonical parameters must be finite")
        if self.gamma_dephasing < 0 or self.gamma_relaxation < 0:
            raise ValueError("canonical damping rates must be non-negative")

    def to_mapping(self) -> dict[str, float]:
        return {
            "omega_x": float(self.omega_x),
            "omega_z": float(self.omega_z),
            "gamma_dephasing": float(self.gamma_dephasing),
            "gamma_relaxation": float(self.gamma_relaxation),
        }


def canonical_qubit_model(
    parameters: CanonicalQubitParameters,
) -> tuple[Array, tuple[Array, ...]]:
    """Build the declared canonical Hamiltonian and collapse operators."""

    p = parameters
    hamiltonian = 0.5 * (p.omega_x * SIGMA_X + p.omega_z * SIGMA_Z)
    collapses: list[Array] = []
    if p.gamma_dephasing > 0:
        collapses.append(np.sqrt(p.gamma_dephasing / 2.0) * SIGMA_Z)
    if p.gamma_relaxation > 0:
        collapses.append(np.sqrt(p.gamma_relaxation) * SIGMA_MINUS)
    return hamiltonian, tuple(collapses)


def recover_canonical_parameters(
    matrix: Array,
    offset: Array,
) -> CanonicalQubitParameters:
    """Recover the gauge-fixed canonical parameters from an affine Bloch generator.

    For the declared family,

    ``A_xx = A_yy = -(gamma_relaxation/2 + gamma_dephasing)``
    ``A_zz = -gamma_relaxation`` and ``b_z = gamma_relaxation``.

    Hamiltonian frequencies occupy the antisymmetric portion of ``A``. We average
    redundant entries to reduce numerical finite-difference noise.
    """

    a = np.asarray(matrix, dtype=float)
    b = np.asarray(offset, dtype=float).reshape(-1)
    if a.shape != (3, 3) or b.shape != (3,):
        raise ValueError("canonical recovery requires A=(3,3) and b=(3,)")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("affine generator contains non-finite values")

    omega_x = 0.5 * (a[2, 1] - a[1, 2])
    omega_z = 0.5 * (a[1, 0] - a[0, 1])
    gamma_relaxation = 0.5 * (b[2] - a[2, 2])
    transverse_decay = -0.5 * (a[0, 0] + a[1, 1])
    gamma_dephasing = transverse_decay - gamma_relaxation / 2.0

    # Small negative values can arise from trajectory finite differences. The
    # recovered object remains a physical canonical parameterization only when the
    # estimates are non-negative beyond numerical tolerance.
    tolerance = 1e-8
    if gamma_relaxation < -tolerance or gamma_dephasing < -tolerance:
        raise ValueError(
            "fitted affine generator is outside the declared canonical damping family"
        )
    return CanonicalQubitParameters(
        omega_x=float(omega_x),
        omega_z=float(omega_z),
        gamma_dephasing=float(max(0.0, gamma_dephasing)),
        gamma_relaxation=float(max(0.0, gamma_relaxation)),
    )


def fit_affine_generator_from_trajectories(
    trajectories: Array,
    times: Array,
) -> tuple[Array, Array]:
    """Estimate ``A,b`` from observed Bloch trajectories using finite differences.

    Multiple independent trajectories are stacked into one least-squares system.
    No Lindblad structure is used by this fit, making it a direct classical control.
    """

    values = np.asarray(trajectories, dtype=float)
    t = np.asarray(times, dtype=float)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("trajectories must have shape (cases, times, 3)")
    if t.ndim != 1 or len(t) != values.shape[1] or len(t) < 3:
        raise ValueError("times must align with trajectory time axis and contain >=3 points")
    if np.any(np.diff(t) <= 0):
        raise ValueError("times must be strictly increasing")
    if not np.all(np.isfinite(values)):
        raise ValueError("trajectories contain non-finite values")

    designs: list[Array] = []
    derivatives: list[Array] = []
    for trajectory in values:
        derivative = np.gradient(trajectory, t, axis=0, edge_order=2)
        designs.append(
            np.concatenate([trajectory, np.ones((len(t), 1))], axis=1)
        )
        derivatives.append(derivative)
    design = np.concatenate(designs, axis=0)
    target = np.concatenate(derivatives, axis=0)
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    return coefficients[:3].T, coefficients[3]


def simulate_canonical_bloch_trajectories(
    parameters: CanonicalQubitParameters,
    times: Array,
    initial_vectors: Array,
) -> Array:
    """Generate synthetic observations through the Lindblad implementation itself."""

    hamiltonian, collapses = canonical_qubit_model(parameters)
    initial = np.asarray(initial_vectors, dtype=float)
    if initial.ndim != 2 or initial.shape[1] != 3:
        raise ValueError("initial_vectors must have shape (cases, 3)")
    trajectories: list[Array] = []
    for vector in initial:
        density_trajectory = evolve_lindblad(
            bloch_to_density(vector),
            hamiltonian,
            times,
            collapse_operators=collapses,
            project_each_step=False,
        )
        trajectories.append(
            np.stack(
                [density_to_bloch(state, require_physical=False) for state in density_trajectory]
            )
        )
    return np.stack(trajectories)


def _parameter_errors(
    truth: CanonicalQubitParameters,
    recovered: CanonicalQubitParameters,
) -> dict[str, float]:
    true = truth.to_mapping()
    fit = recovered.to_mapping()
    floors = {
        "omega_x": 0.25,
        "omega_z": 0.25,
        "gamma_dephasing": 0.10,
        "gamma_relaxation": 0.10,
    }
    return {
        name: float(abs(fit[name] - true[name]) / max(abs(true[name]), floors[name]))
        for name in true
    }


@dataclass(frozen=True)
class SyntheticRecoveryCase:
    case_id: str
    truth: CanonicalQubitParameters
    recovered: CanonicalQubitParameters
    normalized_parameter_errors: dict[str, float]
    mean_normalized_error: float
    affine_fit_residual: float
    equivalence_audit: dict[str, Any]
    gauge_audit: dict[str, Any]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "truth": self.truth.to_mapping(),
            "recovered": self.recovered.to_mapping(),
            "normalized_parameter_errors": dict(self.normalized_parameter_errors),
            "mean_normalized_error": float(self.mean_normalized_error),
            "affine_fit_residual": float(self.affine_fit_residual),
            "equivalence_audit": dict(self.equivalence_audit),
            "gauge_audit": dict(self.gauge_audit),
        }


def _canonical_cases() -> tuple[CanonicalQubitParameters, ...]:
    return (
        CanonicalQubitParameters(1.20, 0.80, 0.25, 0.35),
        CanonicalQubitParameters(0.70, 1.10, 0.15, 0.20),
        CanonicalQubitParameters(1.50, -0.60, 0.40, 0.50),
        CanonicalQubitParameters(-0.90, 0.50, 0.30, 0.25),
        CanonicalQubitParameters(0.40, -1.30, 0.10, 0.45),
        CanonicalQubitParameters(-1.10, -0.75, 0.22, 0.30),
    )


def run_e002_synthetic_recovery_grid(
    *,
    seed: int = 2027,
    noise_std: float = 0.003,
) -> dict[str, Any]:
    """Run the preregistration-style moderate-SNR synthetic E002 gate.

    The benchmark first fits an unconstrained classical affine generator to noisy
    trajectories, then maps that generator into the declared gauge-fixed canonical
    family. This keeps parameter recovery separate from the exact Lindblad notation.
    """

    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and non-negative")
    rng = np.random.default_rng(int(seed))
    times = np.linspace(0.0, 3.0, 301)
    initial_vectors = np.asarray(
        [
            [0.80, 0.00, 0.00],
            [0.00, 0.80, 0.00],
            [0.00, 0.00, 0.80],
            [-0.60, 0.20, 0.10],
            [0.20, -0.60, 0.30],
            [0.25, 0.30, -0.55],
        ],
        dtype=float,
    )

    cases: list[SyntheticRecoveryCase] = []
    sign_inversions = 0
    for index, truth in enumerate(_canonical_cases(), start=1):
        hamiltonian, collapses = canonical_qubit_model(truth)
        exact = compile_qubit_lindblad_to_affine(hamiltonian, collapses)
        trajectories = simulate_canonical_bloch_trajectories(
            truth,
            times,
            initial_vectors,
        )
        observed = trajectories + rng.normal(0.0, noise_std, size=trajectories.shape)
        fitted_matrix, fitted_offset = fit_affine_generator_from_trajectories(observed, times)
        recovered = recover_canonical_parameters(fitted_matrix, fitted_offset)
        errors = _parameter_errors(truth, recovered)
        residual = float(
            max(
                np.max(np.abs(fitted_matrix - exact.matrix)),
                np.max(np.abs(fitted_offset - exact.offset)),
            )
        )
        for name in ("omega_x", "omega_z"):
            true_value = getattr(truth, name)
            fit_value = getattr(recovered, name)
            if abs(true_value) > 1e-8 and np.sign(true_value) != np.sign(fit_value):
                sign_inversions += 1
        cases.append(
            SyntheticRecoveryCase(
                case_id=f"canonical-{index:02d}",
                truth=truth,
                recovered=recovered,
                normalized_parameter_errors=errors,
                mean_normalized_error=float(np.mean(list(errors.values()))),
                affine_fit_residual=residual,
                equivalence_audit=audit_qubit_lindblad_affine_equivalence(
                    hamiltonian,
                    collapses,
                ).to_mapping(),
                gauge_audit=audit_lindblad_gauge_nonidentifiability(
                    hamiltonian,
                    collapses,
                ).to_mapping(),
            )
        )

    mean_errors = np.asarray([case.mean_normalized_error for case in cases], dtype=float)
    median_error = float(np.median(mean_errors))
    affine_equivalence_pass = all(
        bool(case.equivalence_audit["equivalent_within_tolerance"]) for case in cases
    )
    gauge_audit_pass = all(
        bool(case.gauge_audit["equivalent_within_tolerance"]) for case in cases
    )
    promotion_pass = bool(
        median_error <= 0.20
        and sign_inversions == 0
        and affine_equivalence_pass
        and gauge_audit_pass
    )
    return {
        "schema_version": 1,
        "experiment": "E002",
        "evidence_tier": "synthetic_parameter_recovery",
        "claim_class": "quantum_inspired",
        "family": "gauge_fixed_qubit_hamiltonian_dephasing_relaxation",
        "seed": int(seed),
        "noise_std": float(noise_std),
        "n_cases": len(cases),
        "median_normalized_recovery_error": median_error,
        "max_case_mean_normalized_recovery_error": float(np.max(mean_errors)),
        "systematic_sign_inversions": int(sign_inversions),
        "affine_equivalence_pass": affine_equivalence_pass,
        "gauge_nonidentifiability_witness_pass": gauge_audit_pass,
        "synthetic_identifiability_gate_pass": promotion_pass,
        "dynamical_information_novel": False,
        "physical_quantum_promotion_eligible": False,
        "cases": [case.to_mapping() for case in cases],
        "interpretation": (
            "The declared gauge-fixed canonical parameters can be tested for recovery, "
            "but the fully observed qubit trajectory remains exactly equivalent to a "
            "classical affine Bloch ODE. Recovery therefore validates parameterization and "
            "identifiability inside this family, not uniquely quantum neural dynamics."
        ),
    }
