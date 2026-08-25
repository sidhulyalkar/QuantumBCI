"""Dependency-light nonlinear classical dynamics control for E002.

v0.13 adds an affine-plus-random-Fourier-feature residual control. The exact
v0.10 full-VAR affine mean is supplied externally and frozen. Only a nonlinear
residual map is fit here.

All preprocessing and residual weights use source-fit authority only. Model
complexity is selected on calibration transitions only. Final evaluation is
read-only. One-step Gaussian predictive density and deterministic mean rollout
are reported separately because nonlinear uncertainty propagation is not
silently approximated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .dynamics_fitting import PHYSICALITY_TOLERANCE
from .trajectory_authority import TrajectoryEvidenceAuthority, TrajectoryEvidenceData

Array = np.ndarray
NONLINEAR_MODEL_ID = "affine_plus_rff_residual_v1"
NONLINEAR_SCORE_ID = "fit_variance_gaussian_one_step_plus_mean_rollout_v1"
RFF_SEED = 1301
FEATURE_COUNTS = (16, 32, 64)
LENGTH_SCALE_MULTIPLIERS = (0.5, 1.0, 2.0)
RIDGES = (1e-4, 1e-2, 1.0)
MIN_FIT_TRANSITIONS_PER_FEATURE = 4
VARIANCE_FLOOR_RELATIVE = 1e-8
VARIANCE_FLOOR_ABSOLUTE = 1e-12


def _transition_sha256(pairs: Array) -> str:
    values = np.ascontiguousarray(np.asarray(pairs, dtype=np.int64).reshape(-1, 2))
    digest = hashlib.sha256()
    digest.update(b"quantumbci.trajectory-transitions.v1\0")
    digest.update(str(values.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def _model_sha256(
    transition: Array,
    intercept: Array,
    state_mean: Array,
    state_scale: Array,
    frequencies: Array,
    phases: Array,
    residual_weights: Array,
    innovation_variance: Array,
) -> str:
    payload = {
        "transition": np.asarray(transition, dtype=float).tolist(),
        "intercept": np.asarray(intercept, dtype=float).tolist(),
        "state_mean": np.asarray(state_mean, dtype=float).tolist(),
        "state_scale": np.asarray(state_scale, dtype=float).tolist(),
        "frequencies": np.asarray(frequencies, dtype=float).tolist(),
        "phases": np.asarray(phases, dtype=float).tolist(),
        "residual_weights": np.asarray(residual_weights, dtype=float).tolist(),
        "innovation_variance": np.asarray(innovation_variance, dtype=float).tolist(),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(b"quantumbci.e002.nonlinear-rff.v1\0" + canonical).hexdigest()


def _validate_affine_mean(
    transition: Array,
    intercept: Array,
    *,
    dimension: int,
) -> tuple[Array, Array]:
    matrix = np.asarray(transition, dtype=float)
    bias = np.asarray(intercept, dtype=float).reshape(-1)
    if matrix.shape != (dimension, dimension):
        raise ValueError("affine transition shape does not match trajectory dimension")
    if bias.shape != (dimension,):
        raise ValueError("affine intercept shape does not match trajectory dimension")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(bias)):
        raise ValueError("affine mean must be finite")
    return matrix, bias


def _pairs_xy(data: TrajectoryEvidenceData, pairs: Array) -> tuple[Array, Array]:
    values = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if len(values) == 0:
        raise ValueError("nonlinear dynamics require at least one transition")
    return (
        np.asarray(data.states[values[:, 0]], dtype=float),
        np.asarray(data.states[values[:, 1]], dtype=float),
    )


def _transition_chains(pairs: Array) -> list[list[int]]:
    values = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if len(values) == 0:
        raise ValueError("nonlinear dynamics require at least one transition")
    next_by_left: dict[int, int] = {}
    predecessors: dict[int, int] = {}
    for left, right in values.tolist():
        left_i, right_i = int(left), int(right)
        if left_i in next_by_left:
            raise RuntimeError("transition graph contains branching left nodes")
        next_by_left[left_i] = right_i
        if right_i in predecessors and predecessors[right_i] != left_i:
            raise RuntimeError("transition graph contains merging right nodes")
        predecessors[right_i] = left_i
    starts = sorted(left for left in next_by_left if left not in predecessors)
    chains: list[list[int]] = []
    visited_edges = 0
    for start in starts:
        chain = [int(start)]
        current = int(start)
        seen: set[int] = set()
        while current in next_by_left:
            if current in seen:
                raise RuntimeError("transition graph contains a cycle")
            seen.add(current)
            current = next_by_left[current]
            chain.append(current)
            visited_edges += 1
        chains.append(chain)
    if visited_edges != len(values):
        raise RuntimeError("transition graph is not a set of simple forward chains")
    return chains


def _fit_standardization(x: Array) -> tuple[Array, Array]:
    values = np.asarray(x, dtype=float)
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    fallback = np.maximum(np.sqrt(np.mean(values**2, axis=0)) * 1e-8, 1e-12)
    scale = np.where(scale > fallback, scale, 1.0)
    return np.asarray(mean, dtype=float), np.asarray(scale, dtype=float)


def _rff_bank(
    *,
    dimension: int,
    max_features: int,
    length_scale: float,
) -> tuple[Array, Array]:
    if dimension <= 0 or max_features <= 0 or length_scale <= 0:
        raise ValueError("RFF dimensions and length scale must be positive")
    rng = np.random.default_rng(RFF_SEED)
    frequencies = rng.normal(
        loc=0.0,
        scale=1.0 / float(length_scale),
        size=(max_features, dimension),
    )
    phases = rng.uniform(0.0, 2.0 * np.pi, size=max_features)
    return np.asarray(frequencies, dtype=float), np.asarray(phases, dtype=float)


def _features(
    x: Array,
    *,
    state_mean: Array,
    state_scale: Array,
    frequencies: Array,
    phases: Array,
) -> Array:
    values = np.asarray(x, dtype=float)
    mean = np.asarray(state_mean, dtype=float)
    scale = np.asarray(state_scale, dtype=float)
    freq = np.asarray(frequencies, dtype=float)
    phase = np.asarray(phases, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(mean) or len(scale) != len(mean):
        raise ValueError("RFF input shape is invalid")
    if freq.ndim != 2 or freq.shape[1] != len(mean) or phase.shape != (len(freq),):
        raise ValueError("RFF parameter shape is invalid")
    standardized = (values - mean) / scale
    return np.sqrt(2.0 / len(freq)) * np.cos(standardized @ freq.T + phase)


def _fit_residual_weights(
    features: Array,
    residual: Array,
    *,
    ridge: float,
) -> tuple[Array, int]:
    phi = np.asarray(features, dtype=float)
    target = np.asarray(residual, dtype=float)
    if phi.ndim != 2 or target.ndim != 2 or len(phi) != len(target):
        raise ValueError("nonlinear residual design/target shapes are incompatible")
    if ridge <= 0 or not np.isfinite(ridge):
        raise ValueError("nonlinear ridge must be finite and positive")
    gram = phi.T @ phi + float(ridge) * np.eye(phi.shape[1])
    weights = np.linalg.solve(gram, phi.T @ target)
    return np.asarray(weights, dtype=float), int(np.linalg.matrix_rank(phi))


def _fit_innovation_variance(
    x: Array,
    y: Array,
    prediction: Array,
) -> Array:
    residual = np.asarray(y, dtype=float) - np.asarray(prediction, dtype=float)
    state_scale = np.var(np.asarray(x, dtype=float), axis=0)
    floor = np.maximum(
        state_scale * VARIANCE_FLOOR_RELATIVE,
        VARIANCE_FLOOR_ABSOLUTE,
    )
    variance = np.maximum(np.mean(residual**2, axis=0), floor)
    if np.any(variance <= 0) or not np.all(np.isfinite(variance)):
        raise RuntimeError("nonlinear innovation variance is not finite and positive")
    return np.asarray(variance, dtype=float)


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
        "mean_valid_qubit_trace_distance": (
            float(np.mean(half_l2[valid])) if np.any(valid) else None
        ),
    }


def _gaussian_mean_nll(target: Array, mean: Array, variance: Array) -> float:
    truth = np.asarray(target, dtype=float)
    prediction = np.asarray(mean, dtype=float)
    var = np.asarray(variance, dtype=float).reshape(-1)
    if truth.shape != prediction.shape or truth.ndim != 2 or len(var) != truth.shape[1]:
        raise ValueError("Gaussian nonlinear score shapes are incompatible")
    if np.any(var <= 0) or not np.all(np.isfinite(var)):
        raise ValueError("Gaussian nonlinear variance must be finite and positive")
    error = truth - prediction
    values = 0.5 * (
        truth.shape[1] * np.log(2.0 * np.pi)
        + np.sum(np.log(var))
        + np.sum(error**2 / var, axis=1)
    )
    return float(np.mean(values))


@dataclass(frozen=True)
class NonlinearCandidateResult:
    feature_count: int
    length_scale_multiplier: float
    ridge: float
    effective_feature_rank: int
    parameter_count: int
    calibration_mean_nll: float
    calibration_rmse: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "feature_count": int(self.feature_count),
            "length_scale_multiplier": float(self.length_scale_multiplier),
            "ridge": float(self.ridge),
            "effective_feature_rank": int(self.effective_feature_rank),
            "parameter_count": int(self.parameter_count),
            "calibration_mean_nll": float(self.calibration_mean_nll),
            "calibration_rmse": float(self.calibration_rmse),
        }


@dataclass(frozen=True)
class NonlinearMetrics:
    n_transitions: int
    one_step_mean_nll: float
    one_step_rmse: float
    one_step_mae: float
    rollout_rmse: float
    rollout_mae: float
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
            "score_id": NONLINEAR_SCORE_ID,
            "n_transitions": int(self.n_transitions),
            "one_step_mean_nll": float(self.one_step_mean_nll),
            "one_step_rmse": float(self.one_step_rmse),
            "one_step_mae": float(self.one_step_mae),
            "rollout_rmse": float(self.rollout_rmse),
            "rollout_mae": float(self.rollout_mae),
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


@dataclass(frozen=True)
class NonlinearResidualModel:
    transition: Array
    intercept: Array
    state_mean: Array
    state_scale: Array
    frequencies: Array
    phases: Array
    residual_weights: Array
    innovation_variance: Array
    feature_count: int
    length_scale_multiplier: float
    ridge: float
    effective_feature_rank: int

    @property
    def nonlinear_parameter_count(self) -> int:
        return int(self.residual_weights.size)

    @property
    def model_sha256(self) -> str:
        return _model_sha256(
            self.transition,
            self.intercept,
            self.state_mean,
            self.state_scale,
            self.frequencies,
            self.phases,
            self.residual_weights,
            self.innovation_variance,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "model_id": NONLINEAR_MODEL_ID,
            "transition": np.asarray(self.transition, dtype=float).tolist(),
            "intercept": np.asarray(self.intercept, dtype=float).tolist(),
            "affine_mean_refit": False,
            "state_standardization_fit_only": True,
            "state_mean": np.asarray(self.state_mean, dtype=float).tolist(),
            "state_scale": np.asarray(self.state_scale, dtype=float).tolist(),
            "feature_count": int(self.feature_count),
            "length_scale_multiplier": float(self.length_scale_multiplier),
            "ridge": float(self.ridge),
            "rff_seed": RFF_SEED,
            "effective_feature_rank": int(self.effective_feature_rank),
            "nonlinear_parameter_count": self.nonlinear_parameter_count,
            "innovation_variance": np.asarray(self.innovation_variance, dtype=float).tolist(),
            "model_sha256": self.model_sha256,
        }


def _predict(model: NonlinearResidualModel, states: Array) -> Array:
    x = np.asarray(states, dtype=float)
    phi = _features(
        x,
        state_mean=model.state_mean,
        state_scale=model.state_scale,
        frequencies=model.frequencies,
        phases=model.phases,
    )
    return (
        x @ model.transition.T
        + model.intercept
        + phi @ model.residual_weights
    )


def evaluate_nonlinear_model(
    data: TrajectoryEvidenceData,
    authority: TrajectoryEvidenceAuthority,
    model: NonlinearResidualModel,
    *,
    role: str,
) -> NonlinearMetrics:
    authority.restore(data)
    if role not in {"calibration", "evaluation"}:
        raise ValueError("nonlinear evaluation supports calibration or evaluation roles")
    pairs = authority.transition_pairs(data, role)  # type: ignore[arg-type]
    x, y = _pairs_xy(data, pairs)
    one_step = _predict(model, x)
    one_error = one_step - y
    one_qubit = _qubit_metrics(one_step, y)

    rollout_predictions: list[Array] = []
    rollout_targets: list[Array] = []
    for chain in _transition_chains(pairs):
        state = np.asarray(data.states[chain[0]], dtype=float).reshape(1, -1)
        for index in chain[1:]:
            state = _predict(model, state)
            rollout_predictions.append(state[0].copy())
            rollout_targets.append(np.asarray(data.states[index], dtype=float).copy())
    rollout = np.asarray(rollout_predictions, dtype=float)
    rollout_target = np.asarray(rollout_targets, dtype=float)
    rollout_error = rollout - rollout_target
    rollout_qubit = _qubit_metrics(rollout, rollout_target)

    return NonlinearMetrics(
        n_transitions=int(len(pairs)),
        one_step_mean_nll=_gaussian_mean_nll(y, one_step, model.innovation_variance),
        one_step_rmse=float(np.sqrt(np.mean(one_error**2))),
        one_step_mae=float(np.mean(np.abs(one_error))),
        rollout_rmse=float(np.sqrt(np.mean(rollout_error**2))),
        rollout_mae=float(np.mean(np.abs(rollout_error))),
        one_step_mean_bloch_half_l2=one_qubit["mean_bloch_half_l2"],
        rollout_mean_bloch_half_l2=rollout_qubit["mean_bloch_half_l2"],
        one_step_prediction_physical_fraction=one_qubit["prediction_physical_fraction"],
        rollout_prediction_physical_fraction=rollout_qubit["prediction_physical_fraction"],
        target_physical_fraction=one_qubit["target_physical_fraction"],
        one_step_valid_qubit_pair_fraction=one_qubit["valid_qubit_pair_fraction"],
        rollout_valid_qubit_pair_fraction=rollout_qubit["valid_qubit_pair_fraction"],
        one_step_mean_valid_qubit_trace_distance=one_qubit["mean_valid_qubit_trace_distance"],
        rollout_mean_valid_qubit_trace_distance=rollout_qubit["mean_valid_qubit_trace_distance"],
    )


@dataclass(frozen=True)
class NonlinearControlResult:
    authority_fingerprint: str
    data_sha256: str
    fit_transition_sha256: str
    calibration_transition_sha256: str
    evaluation_transition_sha256: str
    model: NonlinearResidualModel
    candidates: tuple[NonlinearCandidateResult, ...]
    calibration_metrics: NonlinearMetrics
    evaluation_metrics: NonlinearMetrics

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "experiment": "E002",
            "claim_class": "classical_control",
            "authority_fingerprint": self.authority_fingerprint,
            "data_sha256": self.data_sha256,
            "fit_transition_sha256": self.fit_transition_sha256,
            "calibration_transition_sha256": self.calibration_transition_sha256,
            "evaluation_transition_sha256": self.evaluation_transition_sha256,
            "model": self.model.to_mapping(),
            "candidate_grid": {
                "feature_counts": list(FEATURE_COUNTS),
                "length_scale_multipliers": list(LENGTH_SCALE_MULTIPLIERS),
                "ridges": list(RIDGES),
                "rff_seed": RFF_SEED,
                "minimum_fit_transitions_per_feature": MIN_FIT_TRANSITIONS_PER_FEATURE,
            },
            "candidates": [candidate.to_mapping() for candidate in self.candidates],
            "calibration_metrics": self.calibration_metrics.to_mapping(),
            "evaluation_metrics": self.evaluation_metrics.to_mapping(),
            "fit_authority_only_for_weights_and_standardization": True,
            "calibration_used_for_complexity_selection": True,
            "evaluation_used_for_model_selection": False,
            "affine_mean_refit": False,
            "one_step_gaussian_density_complete": True,
            "deterministic_mean_rollout_complete": True,
            "nonlinear_uncertainty_rollout_complete": False,
            "rollout_likelihood_promotion_eligible": False,
            "interpretation": (
                "The selected model is a flexible classical nonlinear residual around the "
                "frozen affine baseline. Predictive gains support nonlinear classical dynamics, "
                "not a quantum mechanism or microscopic substrate."
            ),
        }


def run_nonlinear_residual_control(
    data: TrajectoryEvidenceData,
    authority: TrajectoryEvidenceAuthority,
    transition: Array,
    intercept: Array,
) -> NonlinearControlResult:
    """Fit/calibrate/evaluate the preregistered v0.13 nonlinear residual control."""

    authority.restore(data)
    matrix, bias = _validate_affine_mean(
        transition,
        intercept,
        dimension=data.state_dimension,
    )
    fit_pairs = authority.transition_pairs(data, "fit")
    calibration_pairs = authority.transition_pairs(data, "calibration")
    evaluation_pairs = authority.transition_pairs(data, "evaluation")
    if len(calibration_pairs) == 0:
        raise ValueError(
            "nonlinear residual control requires calibration transitions for complexity selection"
        )
    x_fit, y_fit = _pairs_xy(data, fit_pairs)
    x_cal, y_cal = _pairs_xy(data, calibration_pairs)
    state_mean, state_scale = _fit_standardization(x_fit)
    base_fit = x_fit @ matrix.T + bias
    residual_fit = y_fit - base_fit

    max_features = max(FEATURE_COUNTS)
    candidates: list[
        tuple[
            tuple[float, int, float, float],
            NonlinearCandidateResult,
            NonlinearResidualModel,
        ]
    ] = []
    for length_scale in LENGTH_SCALE_MULTIPLIERS:
        bank_frequencies, bank_phases = _rff_bank(
            dimension=data.state_dimension,
            max_features=max_features,
            length_scale=length_scale,
        )
        for feature_count in FEATURE_COUNTS:
            if len(fit_pairs) < MIN_FIT_TRANSITIONS_PER_FEATURE * feature_count:
                continue
            frequencies = bank_frequencies[:feature_count]
            phases = bank_phases[:feature_count]
            phi_fit = _features(
                x_fit,
                state_mean=state_mean,
                state_scale=state_scale,
                frequencies=frequencies,
                phases=phases,
            )
            for ridge in RIDGES:
                weights, feature_rank = _fit_residual_weights(
                    phi_fit,
                    residual_fit,
                    ridge=ridge,
                )
                fit_prediction = base_fit + phi_fit @ weights
                variance = _fit_innovation_variance(x_fit, y_fit, fit_prediction)
                model = NonlinearResidualModel(
                    transition=matrix,
                    intercept=bias,
                    state_mean=state_mean,
                    state_scale=state_scale,
                    frequencies=frequencies,
                    phases=phases,
                    residual_weights=weights,
                    innovation_variance=variance,
                    feature_count=feature_count,
                    length_scale_multiplier=length_scale,
                    ridge=ridge,
                    effective_feature_rank=feature_rank,
                )
                cal_prediction = _predict(model, x_cal)
                cal_nll = _gaussian_mean_nll(y_cal, cal_prediction, variance)
                cal_rmse = float(np.sqrt(np.mean((cal_prediction - y_cal) ** 2)))
                candidate = NonlinearCandidateResult(
                    feature_count=feature_count,
                    length_scale_multiplier=length_scale,
                    ridge=ridge,
                    effective_feature_rank=feature_rank,
                    parameter_count=int(weights.size),
                    calibration_mean_nll=cal_nll,
                    calibration_rmse=cal_rmse,
                )
                key = (cal_nll, feature_count, length_scale, ridge)
                candidates.append((key, candidate, model))

    if not candidates:
        raise ValueError(
            "no nonlinear candidate satisfies the fit-transition sample-size authority"
        )
    candidates.sort(key=lambda item: item[0])
    _, _, selected_model = candidates[0]
    calibration_metrics = evaluate_nonlinear_model(
        data,
        authority,
        selected_model,
        role="calibration",
    )
    evaluation_metrics = evaluate_nonlinear_model(
        data,
        authority,
        selected_model,
        role="evaluation",
    )
    return NonlinearControlResult(
        authority_fingerprint=authority.authority_fingerprint,
        data_sha256=data.data_sha256,
        fit_transition_sha256=_transition_sha256(fit_pairs),
        calibration_transition_sha256=_transition_sha256(calibration_pairs),
        evaluation_transition_sha256=_transition_sha256(evaluation_pairs),
        model=selected_model,
        candidates=tuple(item[1] for item in candidates),
        calibration_metrics=calibration_metrics,
        evaluation_metrics=evaluation_metrics,
    )
