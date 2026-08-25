"""Authority-bound classical dynamics controls for E002.

This module extends the v0.9 matched baseline with genuinely distinct predictive
control classes on the exact same ``TrajectoryEvidenceAuthority``:

* persistence, with no fitted parameters;
* diagonal AR(1) with intercept, six parameters in three dimensions;
* full affine VAR(1), twelve parameters in three dimensions.

For fully observed fixed-step states, the full affine VAR(1) forecast mean is also
the one-step discrete LDS forecast mean with identity observation. Calling those
two separate benchmark models would double-count one model class. A probabilistic
Kalman control only becomes distinct after an observation/noise likelihood contract
is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .dynamics_fitting import DynamicsMetrics, PHYSICALITY_TOLERANCE
from .trajectory_authority import TrajectoryEvidenceAuthority, TrajectoryEvidenceData

Array = np.ndarray
DIRECT_DISCRETE_ESTIMATOR_ID = "direct_fixed_step_transition_least_squares_v1"
DISCRETE_ROLLOUT_ID = "iterated_discrete_transition_v1"


def _transition_sha256(pairs: Array) -> str:
    """Use the same transition identity convention as the v0.9 baseline."""

    import hashlib

    values = np.ascontiguousarray(np.asarray(pairs, dtype=np.int64).reshape(-1, 2))
    digest = hashlib.sha256()
    digest.update(b"quantumbci.trajectory-transitions.v1\0")
    digest.update(str(values.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def _pairs_xy(data: TrajectoryEvidenceData, pairs: Array) -> tuple[Array, Array]:
    values = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if len(values) == 0:
        raise ValueError("classical dynamics fitting requires at least one transition")
    return (
        np.asarray(data.states[values[:, 0]], dtype=float),
        np.asarray(data.states[values[:, 1]], dtype=float),
    )


def _least_squares(design: Array, target: Array) -> tuple[Array, int]:
    x = np.asarray(design, dtype=float)
    y = np.asarray(target, dtype=float)
    if x.ndim != 2 or y.ndim not in {1, 2} or len(x) != len(y):
        raise ValueError("least-squares design/target shapes are incompatible")
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    return coefficients, int(np.linalg.matrix_rank(x))


def fit_full_affine_var1(
    data: TrajectoryEvidenceData,
    authority: TrajectoryEvidenceAuthority,
) -> tuple[Array, Array, int]:
    """Fit ``x[t+1] = F x[t] + c`` on legal source-fit transitions."""

    authority.restore(data)
    x, y = _pairs_xy(data, authority.transition_pairs(data, "fit"))
    design = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    coefficients, predictor_rank = _least_squares(design, y)
    transition = coefficients[: data.state_dimension].T
    intercept = np.asarray(coefficients[data.state_dimension], dtype=float).reshape(-1)
    return transition, intercept, predictor_rank * data.state_dimension


def fit_diagonal_ar1(
    data: TrajectoryEvidenceData,
    authority: TrajectoryEvidenceAuthority,
) -> tuple[Array, Array, int]:
    """Fit independent ``x_j[t+1] = a_j x_j[t] + c_j`` controls."""

    authority.restore(data)
    x, y = _pairs_xy(data, authority.transition_pairs(data, "fit"))
    dimension = data.state_dimension
    transition = np.zeros((dimension, dimension), dtype=float)
    intercept = np.zeros(dimension, dtype=float)
    parameter_rank = 0
    for index in range(dimension):
        design = np.column_stack([x[:, index], np.ones(len(x))])
        coefficients, rank = _least_squares(design, y[:, index])
        transition[index, index] = float(coefficients[0])
        intercept[index] = float(coefficients[1])
        parameter_rank += rank
    return transition, intercept, int(parameter_rank)


def persistence_transition(dimension: int) -> tuple[Array, Array]:
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    return np.eye(dimension, dtype=float), np.zeros(dimension, dtype=float)


def _predict(states: Array, transition: Array, intercept: Array) -> Array:
    values = np.asarray(states, dtype=float)
    matrix = np.asarray(transition, dtype=float)
    bias = np.asarray(intercept, dtype=float).reshape(-1)
    if values.ndim != 2 or matrix.shape != (values.shape[1], values.shape[1]):
        raise ValueError("transition shape is incompatible with states")
    if len(bias) != values.shape[1]:
        raise ValueError("intercept shape is incompatible with states")
    return values @ matrix.T + bias


def _rollout_predictions(
    data: TrajectoryEvidenceData,
    pairs: Array,
    transition: Array,
    intercept: Array,
) -> tuple[Array, Array]:
    transitions = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    next_by_left = {int(left): int(right) for left, right in transitions.tolist()}
    right_nodes = {int(right) for right in transitions[:, 1].tolist()}
    starts = sorted(left for left in next_by_left if left not in right_nodes)
    predictions: list[Array] = []
    targets: list[Array] = []
    visited = 0
    for start in starts:
        current_index = int(start)
        current_state = np.asarray(data.states[current_index], dtype=float).reshape(1, -1)
        while current_index in next_by_left:
            next_index = next_by_left[current_index]
            current_state = _predict(current_state, transition, intercept)
            predictions.append(current_state[0].copy())
            targets.append(np.asarray(data.states[next_index], dtype=float).copy())
            current_index = next_index
            visited += 1
    if visited != len(transitions):
        raise RuntimeError("transition graph is not a set of simple forward chains")
    return np.asarray(predictions, dtype=float), np.asarray(targets, dtype=float)


def _qubit_metrics(prediction: Array, target: Array) -> dict[str, float | None]:
    pred = np.asarray(prediction, dtype=float)
    truth = np.asarray(target, dtype=float)
    if pred.ndim != 2 or pred.shape != truth.shape or pred.shape[1] != 3:
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
        "mean_valid_qubit_trace_distance": float(np.mean(half_l2[valid])) if np.any(valid) else None,
    }


def evaluate_discrete_transition(
    data: TrajectoryEvidenceData,
    authority: TrajectoryEvidenceAuthority,
    transition: Array,
    intercept: Array,
    *,
    role: str = "evaluation",
) -> DynamicsMetrics:
    """Score a discrete transition using the same metric semantics as v0.9."""

    authority.restore(data)
    if role not in {"fit", "evaluation"}:
        raise ValueError("classical control scoring supports fit or evaluation roles only")
    pairs = authority.transition_pairs(data, role)  # type: ignore[arg-type]
    x, y = _pairs_xy(data, pairs)
    one_step_prediction = _predict(x, transition, intercept)
    one_step_error = one_step_prediction - y
    rollout_prediction, rollout_target = _rollout_predictions(
        data, pairs, transition, intercept
    )
    rollout_error = rollout_prediction - rollout_target
    one_step_qubit = _qubit_metrics(one_step_prediction, y)
    rollout_qubit = _qubit_metrics(rollout_prediction, rollout_target)
    return DynamicsMetrics(
        n_transitions=int(len(pairs)),
        one_step_rmse=float(np.sqrt(np.mean(one_step_error**2))),
        one_step_mae=float(np.mean(np.abs(one_step_error))),
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
class ClassicalControlLaneResult:
    model_id: str
    model_class: str
    estimator_id: str
    rollout_id: str
    parameter_count: int
    effective_parameter_rank: int
    authority_fingerprint: str
    data_sha256: str
    fit_transition_sha256: str
    evaluation_transition_sha256: str
    transition: Array
    intercept: Array
    spectral_radius: float
    fit_metrics: DynamicsMetrics
    evaluation_metrics: DynamicsMetrics

    def to_mapping(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_class": self.model_class,
            "claim_class": "classical_control",
            "estimator_id": self.estimator_id,
            "rollout_id": self.rollout_id,
            "parameter_count": int(self.parameter_count),
            "effective_parameter_rank": int(self.effective_parameter_rank),
            "authority_fingerprint": self.authority_fingerprint,
            "data_sha256": self.data_sha256,
            "fit_transition_sha256": self.fit_transition_sha256,
            "evaluation_transition_sha256": self.evaluation_transition_sha256,
            "transition": np.asarray(self.transition, dtype=float).tolist(),
            "intercept": np.asarray(self.intercept, dtype=float).tolist(),
            "spectral_radius": float(self.spectral_radius),
            "fit_metrics": self.fit_metrics.to_mapping(),
            "evaluation_metrics": self.evaluation_metrics.to_mapping(),
        }


@dataclass(frozen=True)
class ExtendedClassicalControlsResult:
    authority_fingerprint: str
    data_sha256: str
    fit_transition_sha256: str
    evaluation_transition_sha256: str
    persistence: ClassicalControlLaneResult
    diagonal_ar1: ClassicalControlLaneResult
    full_var1: ClassicalControlLaneResult
    best_one_step_model: str
    best_rollout_model: str
    calibration_used: bool = False

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "experiment": "E002",
            "claim_class": "classical_control",
            "authority_fingerprint": self.authority_fingerprint,
            "data_sha256": self.data_sha256,
            "fit_transition_sha256": self.fit_transition_sha256,
            "evaluation_transition_sha256": self.evaluation_transition_sha256,
            "calibration_used": bool(self.calibration_used),
            "controls": {
                "persistence": self.persistence.to_mapping(),
                "diagonal_ar1": self.diagonal_ar1.to_mapping(),
                "full_var1": self.full_var1.to_mapping(),
            },
            "best_one_step_model": self.best_one_step_model,
            "best_rollout_model": self.best_rollout_model,
            "equivalence_notes": {
                "full_var1_aliases": [
                    "direct discrete affine transition",
                    "VAR(1) with intercept",
                    "fully observed one-step discrete LDS mean with identity observation",
                ],
                "aliases_count_as_one_model_class": True,
                "kalman_forecast_mean_distinct_under_current_contract": False,
                "kalman_reason": (
                    "With the frozen state tensor treated as the fully observed state and no "
                    "probabilistic observation/noise likelihood, an identity-observation "
                    "Kalman forecast mean reduces to the same linear transition mean. A "
                    "distinct Kalman control requires an explicit latent observation/noise "
                    "contract and probabilistic scoring."
                ),
            },
            "interpretation": (
                "These controls expand the classical predictive ladder without double-counting "
                "equivalent model names. They remain bound to the same temporal authority as "
                "the v0.9 canonical and affine baseline."
            ),
        }


def _lane(
    *,
    model_id: str,
    model_class: str,
    parameter_count: int,
    effective_parameter_rank: int,
    transition: Array,
    intercept: Array,
    data: TrajectoryEvidenceData,
    authority: TrajectoryEvidenceAuthority,
    fit_sha: str,
    evaluation_sha: str,
) -> ClassicalControlLaneResult:
    return ClassicalControlLaneResult(
        model_id=model_id,
        model_class=model_class,
        estimator_id=DIRECT_DISCRETE_ESTIMATOR_ID,
        rollout_id=DISCRETE_ROLLOUT_ID,
        parameter_count=int(parameter_count),
        effective_parameter_rank=int(effective_parameter_rank),
        authority_fingerprint=authority.authority_fingerprint,
        data_sha256=data.data_sha256,
        fit_transition_sha256=fit_sha,
        evaluation_transition_sha256=evaluation_sha,
        transition=np.asarray(transition, dtype=float),
        intercept=np.asarray(intercept, dtype=float),
        spectral_radius=float(np.max(np.abs(np.linalg.eigvals(transition)))),
        fit_metrics=evaluate_discrete_transition(
            data, authority, transition, intercept, role="fit"
        ),
        evaluation_metrics=evaluate_discrete_transition(
            data, authority, transition, intercept, role="evaluation"
        ),
    )


def run_extended_classical_controls(
    data: TrajectoryEvidenceData,
    authority: TrajectoryEvidenceAuthority,
) -> ExtendedClassicalControlsResult:
    """Fit the v0.10 classical hierarchy under one frozen trajectory authority."""

    authority.restore(data)
    fit_pairs = authority.transition_pairs(data, "fit")
    evaluation_pairs = authority.transition_pairs(data, "evaluation")
    fit_sha = _transition_sha256(fit_pairs)
    evaluation_sha = _transition_sha256(evaluation_pairs)
    dimension = data.state_dimension

    persistence_matrix, persistence_intercept = persistence_transition(dimension)
    diagonal_matrix, diagonal_intercept, diagonal_rank = fit_diagonal_ar1(data, authority)
    full_matrix, full_intercept, full_rank = fit_full_affine_var1(data, authority)

    persistence = _lane(
        model_id="persistence",
        model_class="identity_transition",
        parameter_count=0,
        effective_parameter_rank=0,
        transition=persistence_matrix,
        intercept=persistence_intercept,
        data=data,
        authority=authority,
        fit_sha=fit_sha,
        evaluation_sha=evaluation_sha,
    )
    diagonal = _lane(
        model_id="diagonal_ar1",
        model_class="independent_ar1_with_intercept",
        parameter_count=2 * dimension,
        effective_parameter_rank=diagonal_rank,
        transition=diagonal_matrix,
        intercept=diagonal_intercept,
        data=data,
        authority=authority,
        fit_sha=fit_sha,
        evaluation_sha=evaluation_sha,
    )
    full = _lane(
        model_id="full_var1_affine",
        model_class="full_var1_with_intercept",
        parameter_count=dimension * dimension + dimension,
        effective_parameter_rank=full_rank,
        transition=full_matrix,
        intercept=full_intercept,
        data=data,
        authority=authority,
        fit_sha=fit_sha,
        evaluation_sha=evaluation_sha,
    )
    lanes = [persistence, diagonal, full]
    best_one_step = min(lanes, key=lambda lane: lane.evaluation_metrics.one_step_rmse)
    best_rollout = min(lanes, key=lambda lane: lane.evaluation_metrics.rollout_rmse)
    return ExtendedClassicalControlsResult(
        authority_fingerprint=authority.authority_fingerprint,
        data_sha256=data.data_sha256,
        fit_transition_sha256=fit_sha,
        evaluation_transition_sha256=evaluation_sha,
        persistence=persistence,
        diagonal_ar1=diagonal,
        full_var1=full,
        best_one_step_model=best_one_step.model_id,
        best_rollout_model=best_rollout.model_id,
    )
