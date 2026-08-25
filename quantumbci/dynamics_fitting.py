"""Authority-bound matched dynamics fitting for E002.

v0.9 deliberately starts with the smallest defensible real fitting comparison:
an unconstrained three-dimensional affine generator versus the declared four-
parameter canonical qubit family. Both lanes consume the exact same
``TrajectoryEvidenceAuthority`` and therefore the same content-addressed state
tensor and legal transition graphs.

The fitting estimator is intentionally explicit and modest. Continuous-time
vector fields are estimated from forward-difference targets on legal fit
transitions, then both fitted generators are scored with the same RK4 integrator.
This is a baseline estimator contract, not an exact continuous-time likelihood.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from .dynamics_equivalence import compile_qubit_lindblad_to_affine
from .e002_synthetic import (
    CanonicalQubitParameters,
    canonical_qubit_model,
    canonical_structure_residual,
)
from .trajectory_authority import TrajectoryEvidenceAuthority, TrajectoryEvidenceData

Array = np.ndarray
FIT_ESTIMATOR_ID = "forward_difference_least_squares_v1"
SCORE_INTEGRATOR_ID = "rk4_one_step_and_rollout_v1"
PHYSICALITY_TOLERANCE = 1e-9


def _transition_sha256(pairs: Array) -> str:
    values = np.ascontiguousarray(np.asarray(pairs, dtype=np.int64).reshape(-1, 2))
    digest = hashlib.sha256()
    digest.update(b"quantumbci.trajectory-transitions.v1\0")
    digest.update(str(values.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def _transition_arrays(
    data: TrajectoryEvidenceData,
    pairs: Array,
) -> tuple[Array, Array, Array, Array]:
    values = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if len(values) == 0:
        raise ValueError("dynamics fitting requires at least one transition")
    left = values[:, 0]
    right = values[:, 1]
    x = np.asarray(data.states[left], dtype=float)
    y = np.asarray(data.states[right], dtype=float)
    dt = np.asarray(data.start_times_s[right] - data.start_times_s[left], dtype=float)
    if np.any(dt <= 0) or not np.all(np.isfinite(dt)):
        raise ValueError("transition time steps must be finite and positive")
    derivative = (y - x) / dt[:, None]
    return x, y, dt, derivative


def _solve_ridge(design: Array, target: Array, *, ridge: float, regularize: Array) -> Array:
    """Solve least squares without forming normal equations.

    ``ridge=0`` is exact least squares. Positive ridge is represented as an
    augmented least-squares system, which avoids squaring the condition number.
    """

    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and non-negative")
    x = np.asarray(design, dtype=float)
    y = np.asarray(target, dtype=float)
    if x.ndim != 2 or y.ndim not in {1, 2} or len(x) != len(y):
        raise ValueError("ridge design/target shapes are incompatible")
    mask = np.asarray(regularize, dtype=float).reshape(-1)
    if len(mask) != x.shape[1] or np.any(mask < 0):
        raise ValueError("regularization mask must match design columns and be non-negative")
    if ridge == 0:
        return np.linalg.lstsq(x, y, rcond=None)[0]
    penalty = np.sqrt(float(ridge)) * np.diag(np.sqrt(mask))
    augmented = np.concatenate([x, penalty], axis=0)
    if y.ndim == 1:
        target_augmented = np.concatenate([y, np.zeros(x.shape[1])])
    else:
        target_augmented = np.concatenate([y, np.zeros((x.shape[1], y.shape[1]))], axis=0)
    return np.linalg.lstsq(augmented, target_augmented, rcond=None)[0]


def fit_affine_generator(
    data: TrajectoryEvidenceData,
    authority: TrajectoryEvidenceAuthority,
    *,
    ridge: float = 0.0,
) -> tuple[Array, Array, float, int]:
    """Fit ``dx/dt = A x + b`` using legal source-fit transitions only."""

    authority.restore(data)
    pairs = authority.transition_pairs(data, "fit")
    x, _, _, derivative = _transition_arrays(data, pairs)
    design = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    coefficients = _solve_ridge(
        design,
        derivative,
        ridge=ridge,
        regularize=np.asarray([1.0] * data.state_dimension + [0.0]),
    )
    matrix = coefficients[: data.state_dimension].T
    offset = np.asarray(coefficients[data.state_dimension], dtype=float).reshape(-1)
    fitted_derivative = x @ matrix.T + offset
    fit_rmse = float(np.sqrt(np.mean((fitted_derivative - derivative) ** 2)))
    return matrix, offset, fit_rmse, int(np.linalg.matrix_rank(design))


def _canonical_design(states: Array) -> Array:
    """Linear derivative design for [omega_x, omega_z, gamma_d, gamma_r]."""

    values = np.asarray(states, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("canonical qubit fitting requires states with dimension 3")
    rows: list[list[float]] = []
    for x, y, z in values:
        rows.extend(
            [
                [0.0, -y, -x, -0.5 * x],
                [-z, x, -y, -0.5 * y],
                [y, 0.0, 0.0, 1.0 - z],
            ]
        )
    return np.asarray(rows, dtype=float)


def _fit_nonnegative_canonical_parameters(
    states: Array,
    derivative: Array,
    *,
    ridge: float,
) -> tuple[CanonicalQubitParameters, tuple[str, ...], float, int]:
    """Solve the tiny nonnegative damping problem by exact active-set enumeration."""

    design = _canonical_design(states)
    target = np.asarray(derivative, dtype=float).reshape(-1)
    names = ("omega_x", "omega_z", "gamma_dephasing", "gamma_relaxation")
    constrained = {2, 3}
    candidates: list[tuple[float, Array, tuple[str, ...]]] = []
    for zero_indices in ((), (2,), (3,), (2, 3)):
        zero = set(zero_indices)
        free = [index for index in range(4) if index not in zero]
        values = _solve_ridge(
            design[:, free],
            target,
            ridge=ridge,
            regularize=np.ones(len(free)),
        ).reshape(-1)
        parameters = np.zeros(4, dtype=float)
        parameters[free] = values
        if any(parameters[index] < -1e-10 for index in constrained if index in free):
            continue
        parameters[2:] = np.maximum(parameters[2:], 0.0)
        residual = design @ parameters - target
        objective = float(np.dot(residual, residual) + ridge * np.dot(parameters, parameters))
        candidates.append((objective, parameters, tuple(names[index] for index in zero_indices)))
    if not candidates:
        raise RuntimeError("canonical active-set solver found no feasible damping solution")
    _, best, active = min(candidates, key=lambda item: item[0])
    fitted = design @ best
    return (
        CanonicalQubitParameters(
            omega_x=float(best[0]),
            omega_z=float(best[1]),
            gamma_dephasing=float(best[2]),
            gamma_relaxation=float(best[3]),
        ),
        active,
        float(np.sqrt(np.mean((fitted - target) ** 2))),
        int(np.linalg.matrix_rank(design)),
    )


def fit_canonical_qubit_generator(
    data: TrajectoryEvidenceData,
    authority: TrajectoryEvidenceAuthority,
    *,
    ridge: float = 0.0,
) -> tuple[CanonicalQubitParameters, Array, Array, tuple[str, ...], float, int]:
    """Fit the four-parameter canonical family on the same legal fit transitions."""

    authority.restore(data)
    if data.state_dimension != 3:
        raise ValueError("canonical qubit dynamics require state_dimension=3")
    pairs = authority.transition_pairs(data, "fit")
    x, _, _, derivative = _transition_arrays(data, pairs)
    parameters, active, fit_rmse, rank = _fit_nonnegative_canonical_parameters(
        x, derivative, ridge=ridge
    )
    hamiltonian, collapses = canonical_qubit_model(parameters)
    generator = compile_qubit_lindblad_to_affine(hamiltonian, collapses)
    return parameters, generator.matrix, generator.offset, active, fit_rmse, rank


def _rk4_step(states: Array, matrix: Array, offset: Array, dt: Array) -> Array:
    values = np.asarray(states, dtype=float)
    a = np.asarray(matrix, dtype=float)
    b = np.asarray(offset, dtype=float).reshape(-1)
    steps = np.asarray(dt, dtype=float).reshape(-1, 1)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if len(values) != len(steps):
        if len(steps) == 1:
            steps = np.repeat(steps, len(values), axis=0)
        else:
            raise ValueError("dt must contain one value per state")

    def rhs(x: Array) -> Array:
        return x @ a.T + b

    k1 = rhs(values)
    k2 = rhs(values + 0.5 * steps * k1)
    k3 = rhs(values + 0.5 * steps * k2)
    k4 = rhs(values + steps * k3)
    return values + steps * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def _rollout_predictions(
    data: TrajectoryEvidenceData,
    pairs: Array,
    matrix: Array,
    offset: Array,
) -> tuple[Array, Array]:
    transitions = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    next_by_left = {int(left): int(right) for left, right in transitions.tolist()}
    right_nodes = {int(right) for right in transitions[:, 1].tolist()}
    starts = sorted(left for left in next_by_left if left not in right_nodes)
    predictions: list[Array] = []
    targets: list[Array] = []
    visited = 0
    for start in starts:
        current_index = start
        current_state = np.asarray(data.states[start], dtype=float).reshape(1, -1)
        while current_index in next_by_left:
            next_index = next_by_left[current_index]
            dt = np.asarray(
                [float(data.start_times_s[next_index] - data.start_times_s[current_index])]
            )
            current_state = _rk4_step(current_state, matrix, offset, dt)
            predictions.append(current_state[0].copy())
            targets.append(np.asarray(data.states[next_index], dtype=float).copy())
            current_index = next_index
            visited += 1
    if visited != len(transitions):
        raise RuntimeError("evaluation transition graph is not a set of simple forward chains")
    return np.asarray(predictions, dtype=float), np.asarray(targets, dtype=float)


def _qubit_coordinate_metrics(prediction: Array, target: Array) -> dict[str, float | None]:
    """Separate coordinate error from physically valid qubit trace distance.

    ``0.5 * ||r-s||`` equals qubit trace distance only when both Bloch vectors are
    physical. The half-L2 coordinate metric is always reported for 3D states, while
    trace distance is restricted to valid physical pairs.
    """

    pred = np.asarray(prediction, dtype=float)
    truth = np.asarray(target, dtype=float)
    if pred.ndim != 2 or truth.shape != pred.shape or pred.shape[1] != 3:
        return {
            "mean_bloch_half_l2": None,
            "prediction_physical_fraction": None,
            "target_physical_fraction": None,
            "valid_qubit_pair_fraction": None,
            "mean_valid_qubit_trace_distance": None,
        }
    half_l2 = 0.5 * np.linalg.norm(pred - truth, axis=1)
    pred_physical = np.linalg.norm(pred, axis=1) <= 1.0 + PHYSICALITY_TOLERANCE
    target_physical = np.linalg.norm(truth, axis=1) <= 1.0 + PHYSICALITY_TOLERANCE
    valid = pred_physical & target_physical
    return {
        "mean_bloch_half_l2": float(np.mean(half_l2)),
        "prediction_physical_fraction": float(np.mean(pred_physical)),
        "target_physical_fraction": float(np.mean(target_physical)),
        "valid_qubit_pair_fraction": float(np.mean(valid)),
        "mean_valid_qubit_trace_distance": (
            float(np.mean(half_l2[valid])) if np.any(valid) else None
        ),
    }


@dataclass(frozen=True)
class DynamicsMetrics:
    n_transitions: int
    one_step_rmse: float
    one_step_mae: float
    rollout_rmse: float
    one_step_mean_bloch_half_l2: float | None
    rollout_mean_bloch_half_l2: float | None
    one_step_prediction_physical_fraction: float | None
    rollout_prediction_physical_fraction: float | None
    target_physical_fraction: float | None
    one_step_valid_qubit_pair_fraction: float | None
    rollout_valid_qubit_pair_fraction: float | None
    one_step_mean_valid_qubit_trace_distance: float | None
    rollout_mean_valid_qubit_trace_distance: float | None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "n_transitions": int(self.n_transitions),
            "one_step_rmse": float(self.one_step_rmse),
            "one_step_mae": float(self.one_step_mae),
            "rollout_rmse": float(self.rollout_rmse),
            "one_step_mean_bloch_half_l2": self.one_step_mean_bloch_half_l2,
            "rollout_mean_bloch_half_l2": self.rollout_mean_bloch_half_l2,
            "one_step_prediction_physical_fraction": self.one_step_prediction_physical_fraction,
            "rollout_prediction_physical_fraction": self.rollout_prediction_physical_fraction,
            "target_physical_fraction": self.target_physical_fraction,
            "one_step_valid_qubit_pair_fraction": self.one_step_valid_qubit_pair_fraction,
            "rollout_valid_qubit_pair_fraction": self.rollout_valid_qubit_pair_fraction,
            "one_step_mean_valid_qubit_trace_distance": self.one_step_mean_valid_qubit_trace_distance,
            "rollout_mean_valid_qubit_trace_distance": self.rollout_mean_valid_qubit_trace_distance,
        }


def evaluate_generator(
    data: TrajectoryEvidenceData,
    authority: TrajectoryEvidenceAuthority,
    matrix: Array,
    offset: Array,
    *,
    role: str = "evaluation",
) -> DynamicsMetrics:
    authority.restore(data)
    if role not in {"fit", "evaluation"}:
        raise ValueError("matched v0.9 scoring supports fit or evaluation roles only")
    pairs = authority.transition_pairs(data, role)  # type: ignore[arg-type]
    x, y, dt, _ = _transition_arrays(data, pairs)
    prediction = _rk4_step(x, matrix, offset, dt)
    error = prediction - y
    rollout_prediction, rollout_target = _rollout_predictions(data, pairs, matrix, offset)
    rollout_error = rollout_prediction - rollout_target
    one_step_qubit = _qubit_coordinate_metrics(prediction, y)
    rollout_qubit = _qubit_coordinate_metrics(rollout_prediction, rollout_target)
    return DynamicsMetrics(
        n_transitions=int(len(pairs)),
        one_step_rmse=float(np.sqrt(np.mean(error**2))),
        one_step_mae=float(np.mean(np.abs(error))),
        rollout_rmse=float(np.sqrt(np.mean(rollout_error**2))),
        one_step_mean_bloch_half_l2=one_step_qubit["mean_bloch_half_l2"],
        rollout_mean_bloch_half_l2=rollout_qubit["mean_bloch_half_l2"],
        one_step_prediction_physical_fraction=one_step_qubit["prediction_physical_fraction"],
        rollout_prediction_physical_fraction=rollout_qubit["prediction_physical_fraction"],
        target_physical_fraction=one_step_qubit["target_physical_fraction"],
        one_step_valid_qubit_pair_fraction=one_step_qubit["valid_qubit_pair_fraction"],
        rollout_valid_qubit_pair_fraction=rollout_qubit["valid_qubit_pair_fraction"],
        one_step_mean_valid_qubit_trace_distance=one_step_qubit[
            "mean_valid_qubit_trace_distance"
        ],
        rollout_mean_valid_qubit_trace_distance=rollout_qubit[
            "mean_valid_qubit_trace_distance"
        ],
    )


@dataclass(frozen=True)
class DynamicsLaneResult:
    model_id: str
    claim_class: str
    parameter_count: int
    fit_estimator_id: str
    score_integrator_id: str
    ridge: float
    fit_design_rank: int
    authority_fingerprint: str
    data_sha256: str
    fit_transition_sha256: str
    evaluation_transition_sha256: str
    fit_derivative_rmse: float
    fit_metrics: DynamicsMetrics
    evaluation_metrics: DynamicsMetrics
    matrix: Array
    offset: Array
    stability_abscissa: float
    calibration_used: bool = False
    canonical_parameters: CanonicalQubitParameters | None = None
    active_rate_constraints: tuple[str, ...] = ()
    canonical_structure_residual_to_affine: float | None = None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "claim_class": self.claim_class,
            "parameter_count": int(self.parameter_count),
            "fit_estimator_id": self.fit_estimator_id,
            "score_integrator_id": self.score_integrator_id,
            "ridge": float(self.ridge),
            "fit_design_rank": int(self.fit_design_rank),
            "authority_fingerprint": self.authority_fingerprint,
            "data_sha256": self.data_sha256,
            "fit_transition_sha256": self.fit_transition_sha256,
            "evaluation_transition_sha256": self.evaluation_transition_sha256,
            "fit_derivative_rmse": float(self.fit_derivative_rmse),
            "fit_metrics": self.fit_metrics.to_mapping(),
            "evaluation_metrics": self.evaluation_metrics.to_mapping(),
            "matrix": np.asarray(self.matrix, dtype=float).tolist(),
            "offset": np.asarray(self.offset, dtype=float).tolist(),
            "stability_abscissa": float(self.stability_abscissa),
            "calibration_used": bool(self.calibration_used),
            "canonical_parameters": (
                None if self.canonical_parameters is None else self.canonical_parameters.to_mapping()
            ),
            "active_rate_constraints": list(self.active_rate_constraints),
            "canonical_structure_residual_to_affine": (
                None
                if self.canonical_structure_residual_to_affine is None
                else float(self.canonical_structure_residual_to_affine)
            ),
        }


@dataclass(frozen=True)
class MatchedDynamicsBenchmarkResult:
    authority_fingerprint: str
    data_sha256: str
    fit_transition_sha256: str
    evaluation_transition_sha256: str
    affine: DynamicsLaneResult
    canonical: DynamicsLaneResult
    canonical_minus_affine_one_step_rmse: float
    canonical_minus_affine_rollout_rmse: float
    parameter_reduction: int
    same_evidence_verified: bool = True
    calibration_used: bool = False
    dynamical_information_novel: bool = False
    physical_quantum_promotion_eligible: bool = False

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "experiment": "E002",
            "authority_fingerprint": self.authority_fingerprint,
            "data_sha256": self.data_sha256,
            "fit_transition_sha256": self.fit_transition_sha256,
            "evaluation_transition_sha256": self.evaluation_transition_sha256,
            "affine": self.affine.to_mapping(),
            "canonical": self.canonical.to_mapping(),
            "canonical_minus_affine_one_step_rmse": float(
                self.canonical_minus_affine_one_step_rmse
            ),
            "canonical_minus_affine_rollout_rmse": float(
                self.canonical_minus_affine_rollout_rmse
            ),
            "parameter_reduction": int(self.parameter_reduction),
            "same_evidence_verified": bool(self.same_evidence_verified),
            "calibration_used": bool(self.calibration_used),
            "dynamical_information_novel": bool(self.dynamical_information_novel),
            "physical_quantum_promotion_eligible": bool(self.physical_quantum_promotion_eligible),
            "interpretation": (
                "Both lanes consume the identical authority, state tensor, transition graphs, "
                "forward-difference fitting target, and RK4 scoring convention. The canonical "
                "lane is a constrained four-parameter coordinate system whose fully observed "
                "qubit trajectories remain classically affine-equivalent. Three-dimensional "
                "coordinate errors are not called trace distances unless both compared Bloch "
                "vectors are physical. Predictive or complexity advantages do not establish "
                "a physical neural quantum mechanism."
            ),
        }


def run_matched_qubit_dynamics_benchmark(
    data: TrajectoryEvidenceData,
    authority: TrajectoryEvidenceAuthority,
    *,
    ridge: float = 0.0,
) -> MatchedDynamicsBenchmarkResult:
    """Fit and score affine + canonical lanes under one immutable authority."""

    authority.restore(data)
    if data.state_dimension != 3:
        raise ValueError("matched qubit dynamics benchmark requires state_dimension=3")
    fit_pairs = authority.transition_pairs(data, "fit")
    evaluation_pairs = authority.transition_pairs(data, "evaluation")
    fit_sha = _transition_sha256(fit_pairs)
    evaluation_sha = _transition_sha256(evaluation_pairs)

    affine_matrix, affine_offset, affine_derivative_rmse, affine_rank = fit_affine_generator(
        data, authority, ridge=ridge
    )
    (
        canonical_parameters,
        canonical_matrix,
        canonical_offset,
        active,
        canonical_derivative_rmse,
        canonical_rank,
    ) = fit_canonical_qubit_generator(data, authority, ridge=ridge)

    affine_fit_metrics = evaluate_generator(data, authority, affine_matrix, affine_offset, role="fit")
    affine_eval_metrics = evaluate_generator(
        data, authority, affine_matrix, affine_offset, role="evaluation"
    )
    canonical_fit_metrics = evaluate_generator(
        data, authority, canonical_matrix, canonical_offset, role="fit"
    )
    canonical_eval_metrics = evaluate_generator(
        data, authority, canonical_matrix, canonical_offset, role="evaluation"
    )

    shared = {
        "fit_estimator_id": FIT_ESTIMATOR_ID,
        "score_integrator_id": SCORE_INTEGRATOR_ID,
        "ridge": float(ridge),
        "authority_fingerprint": authority.authority_fingerprint,
        "data_sha256": data.data_sha256,
        "fit_transition_sha256": fit_sha,
        "evaluation_transition_sha256": evaluation_sha,
    }
    affine = DynamicsLaneResult(
        model_id="unconstrained_affine_generator",
        claim_class="classical_control",
        parameter_count=12,
        fit_design_rank=affine_rank,
        fit_derivative_rmse=affine_derivative_rmse,
        fit_metrics=affine_fit_metrics,
        evaluation_metrics=affine_eval_metrics,
        matrix=affine_matrix,
        offset=affine_offset,
        stability_abscissa=float(np.max(np.real(np.linalg.eigvals(affine_matrix)))),
        **shared,
    )
    canonical = DynamicsLaneResult(
        model_id="canonical_lindblad_family",
        claim_class="quantum_inspired",
        parameter_count=4,
        fit_design_rank=canonical_rank,
        fit_derivative_rmse=canonical_derivative_rmse,
        fit_metrics=canonical_fit_metrics,
        evaluation_metrics=canonical_eval_metrics,
        matrix=canonical_matrix,
        offset=canonical_offset,
        stability_abscissa=float(np.max(np.real(np.linalg.eigvals(canonical_matrix)))),
        canonical_parameters=canonical_parameters,
        active_rate_constraints=active,
        canonical_structure_residual_to_affine=canonical_structure_residual(
            affine_matrix, affine_offset, canonical_parameters
        ),
        **shared,
    )
    return MatchedDynamicsBenchmarkResult(
        authority_fingerprint=authority.authority_fingerprint,
        data_sha256=data.data_sha256,
        fit_transition_sha256=fit_sha,
        evaluation_transition_sha256=evaluation_sha,
        affine=affine,
        canonical=canonical,
        canonical_minus_affine_one_step_rmse=(
            canonical_eval_metrics.one_step_rmse - affine_eval_metrics.one_step_rmse
        ),
        canonical_minus_affine_rollout_rmse=(
            canonical_eval_metrics.rollout_rmse - affine_eval_metrics.rollout_rmse
        ),
        parameter_reduction=affine.parameter_count - canonical.parameter_count,
    )
