"""Probabilistic state-space controls for E002 under frozen temporal evidence.

v0.11 makes the first Kalman-family control scientifically distinct from the
observed-state VAR baseline. The v0.10 full VAR mean transition is frozen and is
not refit here. The only new model structure is a same-coordinate latent state
observed through a fixed identity observation matrix with diagonal process and
measurement noise.

Noise scales are selected on calibration evidence only. Final evaluation is never
used for hyperparameter selection. Both sequential filtered prediction and open-loop
prediction are reported because they consume different amounts of within-role
observation history.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .dynamics_fitting import PHYSICALITY_TOLERANCE
from .trajectory_authority import TrajectoryEvidenceAuthority, TrajectoryEvidenceData

Array = np.ndarray
PROBABILISTIC_MODEL_ID = "identity_observation_diagonal_noise_kalman_v1"
DIRECT_GAUSSIAN_BASELINE_ID = "gaussian_var1_diagonal_innovation_v1"
NOISE_CALIBRATION_OBJECTIVE = "sequential_predictive_gaussian_nll_v1"
Q_SCALE_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
R_SCALE_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
VARIANCE_FLOOR_RELATIVE = 1e-8
VARIANCE_FLOOR_ABSOLUTE = 1e-12
NORMAL_95 = 1.959963984540054


def _transition_sha256(pairs: Array) -> str:
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
        raise ValueError("probabilistic state-space scoring requires at least one transition")
    return (
        np.asarray(data.states[values[:, 0]], dtype=float),
        np.asarray(data.states[values[:, 1]], dtype=float),
    )


def _transition_chains(pairs: Array) -> list[list[int]]:
    """Convert a legal transition graph into deterministic simple forward chains."""

    values = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if len(values) == 0:
        raise ValueError("probabilistic state-space scoring requires at least one transition")
    next_by_left = {int(left): int(right) for left, right in values.tolist()}
    if len(next_by_left) != len(values):
        raise RuntimeError("transition graph contains branching left nodes")
    predecessors: dict[int, int] = {}
    for left, right in values.tolist():
        left_i, right_i = int(left), int(right)
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


def _positive_definite(matrix: Array, *, floor: float = VARIANCE_FLOOR_ABSOLUTE) -> Array:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("covariance must be square")
    symmetric = (values + values.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, float(floor))
    repaired = (eigenvectors * clipped) @ eigenvectors.T
    return (repaired + repaired.T) / 2.0


def _validate_mean_model(
    transition: Array,
    intercept: Array,
    *,
    dimension: int,
) -> tuple[Array, Array]:
    matrix = np.asarray(transition, dtype=float)
    bias = np.asarray(intercept, dtype=float).reshape(-1)
    if matrix.shape != (dimension, dimension):
        raise ValueError("mean transition shape does not match trajectory state dimension")
    if bias.shape != (dimension,):
        raise ValueError("mean intercept shape does not match trajectory state dimension")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(bias)):
        raise ValueError("mean transition and intercept must be finite")
    return matrix, bias


def _qubit_mean_metrics(prediction: Array, target: Array) -> dict[str, float | None]:
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


def _gaussian_terms(target: Array, mean: Array, covariance: Array) -> tuple[float, float, float]:
    cov = _positive_definite(covariance)
    sign, log_det = np.linalg.slogdet(cov)
    if sign <= 0 or not np.isfinite(log_det):
        raise np.linalg.LinAlgError("predictive covariance is not positive definite")
    error = np.asarray(target, dtype=float) - np.asarray(mean, dtype=float)
    mahalanobis = float(error @ np.linalg.solve(cov, error))
    nll = 0.5 * (
        len(error) * np.log(2.0 * np.pi)
        + float(log_det)
        + mahalanobis
    )
    return float(nll), mahalanobis, float(log_det)


@dataclass(frozen=True)
class PredictiveDensityMetrics:
    n_scored: int
    mean_nll: float
    total_nll: float
    predictive_mean_rmse: float
    predictive_mean_mae: float
    mean_mahalanobis_sq: float
    mean_log_det_covariance: float
    marginal_95_coverage: float
    mean_bloch_half_l2: float | None
    prediction_physical_fraction: float | None
    target_physical_fraction: float | None
    valid_qubit_pair_fraction: float | None
    mean_valid_qubit_trace_distance: float | None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "n_scored": int(self.n_scored),
            "mean_nll": float(self.mean_nll),
            "total_nll": float(self.total_nll),
            "predictive_mean_rmse": float(self.predictive_mean_rmse),
            "predictive_mean_mae": float(self.predictive_mean_mae),
            "mean_mahalanobis_sq": float(self.mean_mahalanobis_sq),
            "mean_log_det_covariance": float(self.mean_log_det_covariance),
            "marginal_95_coverage": float(self.marginal_95_coverage),
            "mean_bloch_half_l2": self.mean_bloch_half_l2,
            "prediction_physical_fraction": self.prediction_physical_fraction,
            "target_physical_fraction": self.target_physical_fraction,
            "valid_qubit_pair_fraction": self.valid_qubit_pair_fraction,
            "mean_valid_qubit_trace_distance": self.mean_valid_qubit_trace_distance,
        }


def _summarize_density_predictions(
    predictions: Iterable[Array],
    targets: Iterable[Array],
    covariances: Iterable[Array],
) -> PredictiveDensityMetrics:
    pred = np.asarray(list(predictions), dtype=float)
    truth = np.asarray(list(targets), dtype=float)
    covs = np.asarray(list(covariances), dtype=float)
    if pred.ndim != 2 or pred.shape != truth.shape or len(pred) == 0:
        raise ValueError("predictive-density inputs must contain aligned non-empty vectors")
    if covs.shape != (len(pred), pred.shape[1], pred.shape[1]):
        raise ValueError("one predictive covariance is required per prediction")

    nll_values: list[float] = []
    mahalanobis_values: list[float] = []
    log_det_values: list[float] = []
    covered = 0
    coverage_total = pred.size
    for prediction, target, covariance in zip(pred, truth, covs):
        covariance = _positive_definite(covariance)
        nll, mahalanobis, log_det = _gaussian_terms(target, prediction, covariance)
        nll_values.append(nll)
        mahalanobis_values.append(mahalanobis)
        log_det_values.append(log_det)
        standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        covered += int(
            np.count_nonzero(
                np.abs(target - prediction) <= NORMAL_95 * standard_deviation
            )
        )

    error = pred - truth
    qubit = _qubit_mean_metrics(pred, truth)
    return PredictiveDensityMetrics(
        n_scored=int(len(pred)),
        mean_nll=float(np.mean(nll_values)),
        total_nll=float(np.sum(nll_values)),
        predictive_mean_rmse=float(np.sqrt(np.mean(error**2))),
        predictive_mean_mae=float(np.mean(np.abs(error))),
        mean_mahalanobis_sq=float(np.mean(mahalanobis_values)),
        mean_log_det_covariance=float(np.mean(log_det_values)),
        marginal_95_coverage=float(covered / coverage_total),
        mean_bloch_half_l2=qubit["mean_bloch_half_l2"],
        prediction_physical_fraction=qubit["prediction_physical_fraction"],
        target_physical_fraction=qubit["target_physical_fraction"],
        valid_qubit_pair_fraction=qubit["valid_qubit_pair_fraction"],
        mean_valid_qubit_trace_distance=qubit["mean_valid_qubit_trace_distance"],
    )


def fit_base_innovation_variance(
    data: TrajectoryEvidenceData,
    authority: TrajectoryEvidenceAuthority,
    transition: Array,
    intercept: Array,
) -> Array:
    """Estimate a diagonal innovation scale using source-fit transitions only."""

    authority.restore(data)
    matrix, bias = _validate_mean_model(
        transition, intercept, dimension=data.state_dimension
    )
    pairs = authority.transition_pairs(data, "fit")
    x, y = _pairs_xy(data, pairs)
    residual = y - (x @ matrix.T + bias)
    state_variance = np.var(x, axis=0)
    floor = np.maximum(
        state_variance * VARIANCE_FLOOR_RELATIVE,
        VARIANCE_FLOOR_ABSOLUTE,
    )
    variance = np.maximum(np.mean(residual**2, axis=0), floor)
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0):
        raise RuntimeError("failed to construct a finite positive innovation variance")
    return np.asarray(variance, dtype=float)


def score_direct_gaussian_var(
    data: TrajectoryEvidenceData,
    authority: TrajectoryEvidenceAuthority,
    transition: Array,
    intercept: Array,
    innovation_variance: Array,
    *,
    role: str,
    open_loop: bool,
) -> PredictiveDensityMetrics:
    """Score the frozen observed-state VAR mean with diagonal Gaussian innovations."""

    authority.restore(data)
    if role not in {"fit", "calibration", "evaluation"}:
        raise ValueError("role must be fit, calibration, or evaluation")
    matrix, bias = _validate_mean_model(
        transition, intercept, dimension=data.state_dimension
    )
    variance = np.asarray(innovation_variance, dtype=float).reshape(-1)
    if variance.shape != (data.state_dimension,) or np.any(variance <= 0):
        raise ValueError("innovation variance must be positive with one value per state coordinate")
    innovation_covariance = np.diag(variance)
    pairs = authority.transition_pairs(data, role)  # type: ignore[arg-type]

    predictions: list[Array] = []
    targets: list[Array] = []
    covariances: list[Array] = []
    if not open_loop:
        x, y = _pairs_xy(data, pairs)
        prediction = x @ matrix.T + bias
        predictions.extend(prediction)
        targets.extend(y)
        covariances.extend([innovation_covariance] * len(prediction))
    else:
        for chain in _transition_chains(pairs):
            mean = np.asarray(data.states[chain[0]], dtype=float).copy()
            covariance = np.zeros((data.state_dimension, data.state_dimension), dtype=float)
            for index in chain[1:]:
                mean = matrix @ mean + bias
                covariance = matrix @ covariance @ matrix.T + innovation_covariance
                covariance = _positive_definite(covariance)
                predictions.append(mean.copy())
                targets.append(np.asarray(data.states[index], dtype=float).copy())
                covariances.append(covariance.copy())
    return _summarize_density_predictions(predictions, targets, covariances)


def score_identity_observation_kalman(
    data: TrajectoryEvidenceData,
    authority: TrajectoryEvidenceAuthority,
    transition: Array,
    intercept: Array,
    process_variance: Array,
    measurement_variance: Array,
    *,
    role: str,
    open_loop: bool,
) -> PredictiveDensityMetrics:
    """Score a same-coordinate latent state with fixed identity observation matrix.

    Each role-local chain is initialized from its first observation. Hidden state is
    never carried from fit into calibration or from calibration into evaluation.
    """

    authority.restore(data)
    if role not in {"fit", "calibration", "evaluation"}:
        raise ValueError("role must be fit, calibration, or evaluation")
    matrix, bias = _validate_mean_model(
        transition, intercept, dimension=data.state_dimension
    )
    q_diag = np.asarray(process_variance, dtype=float).reshape(-1)
    r_diag = np.asarray(measurement_variance, dtype=float).reshape(-1)
    if q_diag.shape != (data.state_dimension,) or r_diag.shape != (data.state_dimension,):
        raise ValueError("Q/R diagonals must match the state dimension")
    if np.any(q_diag <= 0) or np.any(r_diag <= 0):
        raise ValueError("Q/R diagonals must be strictly positive")
    q = np.diag(q_diag)
    r = np.diag(r_diag)
    identity = np.eye(data.state_dimension)
    pairs = authority.transition_pairs(data, role)  # type: ignore[arg-type]

    predictions: list[Array] = []
    targets: list[Array] = []
    covariances: list[Array] = []
    for chain in _transition_chains(pairs):
        state = np.asarray(data.states[chain[0]], dtype=float).copy()
        covariance = r.copy()
        for index in chain[1:]:
            predicted_state = matrix @ state + bias
            predicted_covariance = matrix @ covariance @ matrix.T + q
            innovation_covariance = _positive_definite(predicted_covariance + r)
            observation = np.asarray(data.states[index], dtype=float)
            predictions.append(predicted_state.copy())
            targets.append(observation.copy())
            covariances.append(innovation_covariance.copy())

            if open_loop:
                state = predicted_state
                covariance = _positive_definite(predicted_covariance)
                continue

            gain = np.linalg.solve(
                innovation_covariance,
                predicted_covariance.T,
            ).T
            residual = observation - predicted_state
            state = predicted_state + gain @ residual
            i_minus_k = identity - gain
            covariance = (
                i_minus_k @ predicted_covariance @ i_minus_k.T
                + gain @ r @ gain.T
            )
            covariance = _positive_definite(covariance)
    return _summarize_density_predictions(predictions, targets, covariances)


@dataclass(frozen=True)
class NoiseCalibrationCandidate:
    q_scale: float
    r_scale: float
    mean_nll: float
    predictive_mean_rmse: float
    marginal_95_coverage: float

    def to_mapping(self) -> dict[str, float]:
        return {
            "q_scale": float(self.q_scale),
            "r_scale": float(self.r_scale),
            "mean_nll": float(self.mean_nll),
            "predictive_mean_rmse": float(self.predictive_mean_rmse),
            "marginal_95_coverage": float(self.marginal_95_coverage),
        }


@dataclass(frozen=True)
class ProbabilisticStateSpaceResult:
    authority_fingerprint: str
    data_sha256: str
    fit_transition_sha256: str
    calibration_transition_sha256: str
    evaluation_transition_sha256: str
    transition: Array
    intercept: Array
    base_innovation_variance: Array
    selected_q_scale: float
    selected_r_scale: float
    process_variance: Array
    measurement_variance: Array
    calibration_candidates: tuple[NoiseCalibrationCandidate, ...]
    direct_calibration_sequential: PredictiveDensityMetrics
    kalman_calibration_sequential: PredictiveDensityMetrics
    direct_evaluation_sequential: PredictiveDensityMetrics
    direct_evaluation_open_loop: PredictiveDensityMetrics
    kalman_evaluation_sequential: PredictiveDensityMetrics
    kalman_evaluation_open_loop: PredictiveDensityMetrics

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "experiment": "E002",
            "claim_class": "classical_control",
            "model_id": PROBABILISTIC_MODEL_ID,
            "matched_direct_baseline_id": DIRECT_GAUSSIAN_BASELINE_ID,
            "authority_fingerprint": self.authority_fingerprint,
            "data_sha256": self.data_sha256,
            "fit_transition_sha256": self.fit_transition_sha256,
            "calibration_transition_sha256": self.calibration_transition_sha256,
            "evaluation_transition_sha256": self.evaluation_transition_sha256,
            "observation_matrix": np.eye(len(self.intercept)).tolist(),
            "observation_matrix_fixed": True,
            "latent_dimension": int(len(self.intercept)),
            "observation_dimension": int(len(self.intercept)),
            "latent_coordinate_gauge_fixed": True,
            "latent_state_semantics": "denoised_same_coordinate_state",
            "mean_transition_refit": False,
            "transition": np.asarray(self.transition, dtype=float).tolist(),
            "intercept": np.asarray(self.intercept, dtype=float).tolist(),
            "base_innovation_variance": np.asarray(
                self.base_innovation_variance, dtype=float
            ).tolist(),
            "variance_shape": "diagonal",
            "noise_calibration_objective": NOISE_CALIBRATION_OBJECTIVE,
            "q_scale_grid": [float(v) for v in Q_SCALE_GRID],
            "r_scale_grid": [float(v) for v in R_SCALE_GRID],
            "selected_q_scale": float(self.selected_q_scale),
            "selected_r_scale": float(self.selected_r_scale),
            "process_variance": np.asarray(self.process_variance, dtype=float).tolist(),
            "measurement_variance": np.asarray(
                self.measurement_variance, dtype=float
            ).tolist(),
            "calibration_candidates": [
                candidate.to_mapping() for candidate in self.calibration_candidates
            ],
            "direct_gaussian_var": {
                "calibration_sequential": self.direct_calibration_sequential.to_mapping(),
                "evaluation_sequential": self.direct_evaluation_sequential.to_mapping(),
                "evaluation_open_loop": self.direct_evaluation_open_loop.to_mapping(),
            },
            "identity_observation_kalman": {
                "calibration_sequential": self.kalman_calibration_sequential.to_mapping(),
                "evaluation_sequential": self.kalman_evaluation_sequential.to_mapping(),
                "evaluation_open_loop": self.kalman_evaluation_open_loop.to_mapping(),
            },
            "calibration_used": True,
            "evaluation_used_for_hyperparameter_selection": False,
            "role_boundary_filter_reset": True,
            "first_observation_per_role_chain_is_context_only": True,
            "mean_forecast_parameter_count_upstream": int(
                len(self.intercept) ** 2 + len(self.intercept)
            ),
            "noise_scale_hyperparameter_count": 2,
            "aic_parameter_count": None,
            "aic_reason": (
                "Q/R scales are calibration-selected hyperparameters around fit-derived "
                "innovation scales rather than jointly maximum-likelihood parameters."
            ),
            "evaluation_comparison": {
                "direct_minus_kalman_sequential_mean_nll": float(
                    self.direct_evaluation_sequential.mean_nll
                    - self.kalman_evaluation_sequential.mean_nll
                ),
                "direct_minus_kalman_sequential_rmse": float(
                    self.direct_evaluation_sequential.predictive_mean_rmse
                    - self.kalman_evaluation_sequential.predictive_mean_rmse
                ),
                "direct_minus_kalman_open_loop_mean_nll": float(
                    self.direct_evaluation_open_loop.mean_nll
                    - self.kalman_evaluation_open_loop.mean_nll
                ),
                "direct_minus_kalman_open_loop_rmse": float(
                    self.direct_evaluation_open_loop.predictive_mean_rmse
                    - self.kalman_evaluation_open_loop.predictive_mean_rmse
                ),
            },
            "interpretation": (
                "The Kalman lane differs from direct observed-state VAR only through an "
                "identity-observation latent-noise model, role-local filtering, and calibrated "
                "predictive covariance. The upstream mean transition is frozen."
            ),
        }


def _validated_grid(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    grid = tuple(float(value) for value in values)
    if not grid or any((not np.isfinite(value) or value <= 0) for value in grid):
        raise ValueError(f"{name} must contain finite positive values")
    if len(set(grid)) != len(grid):
        raise ValueError(f"{name} contains duplicate values")
    return tuple(sorted(grid))


def run_probabilistic_state_space_control(
    data: TrajectoryEvidenceData,
    authority: TrajectoryEvidenceAuthority,
    transition: Array,
    intercept: Array,
    *,
    q_scale_grid: Iterable[float] = Q_SCALE_GRID,
    r_scale_grid: Iterable[float] = R_SCALE_GRID,
) -> ProbabilisticStateSpaceResult:
    """Calibrate and evaluate the v0.11 identity-observation Kalman control."""

    authority.restore(data)
    matrix, bias = _validate_mean_model(
        transition, intercept, dimension=data.state_dimension
    )
    q_grid = _validated_grid(q_scale_grid, name="q_scale_grid")
    r_grid = _validated_grid(r_scale_grid, name="r_scale_grid")
    fit_pairs = authority.transition_pairs(data, "fit")
    calibration_pairs = authority.transition_pairs(data, "calibration")
    evaluation_pairs = authority.transition_pairs(data, "evaluation")
    if len(calibration_pairs) == 0:
        raise ValueError(
            "probabilistic state-space control requires calibration transitions; "
            "noise hyperparameters may not be selected from final evaluation"
        )

    base_variance = fit_base_innovation_variance(
        data, authority, matrix, bias
    )
    direct_calibration = score_direct_gaussian_var(
        data,
        authority,
        matrix,
        bias,
        base_variance,
        role="calibration",
        open_loop=False,
    )

    candidates: list[
        tuple[tuple[float, float, float, float], NoiseCalibrationCandidate, PredictiveDensityMetrics]
    ] = []
    for q_scale in q_grid:
        for r_scale in r_grid:
            metrics = score_identity_observation_kalman(
                data,
                authority,
                matrix,
                bias,
                q_scale * base_variance,
                r_scale * base_variance,
                role="calibration",
                open_loop=False,
            )
            candidate = NoiseCalibrationCandidate(
                q_scale=q_scale,
                r_scale=r_scale,
                mean_nll=metrics.mean_nll,
                predictive_mean_rmse=metrics.predictive_mean_rmse,
                marginal_95_coverage=metrics.marginal_95_coverage,
            )
            key = (
                metrics.mean_nll,
                q_scale + r_scale,
                q_scale,
                r_scale,
            )
            candidates.append((key, candidate, metrics))
    if not candidates:
        raise RuntimeError("noise calibration produced no candidates")
    candidates.sort(key=lambda item: item[0])
    _, selected, calibration_metrics = candidates[0]
    process_variance = selected.q_scale * base_variance
    measurement_variance = selected.r_scale * base_variance

    direct_evaluation_sequential = score_direct_gaussian_var(
        data,
        authority,
        matrix,
        bias,
        base_variance,
        role="evaluation",
        open_loop=False,
    )
    direct_evaluation_open_loop = score_direct_gaussian_var(
        data,
        authority,
        matrix,
        bias,
        base_variance,
        role="evaluation",
        open_loop=True,
    )
    kalman_evaluation_sequential = score_identity_observation_kalman(
        data,
        authority,
        matrix,
        bias,
        process_variance,
        measurement_variance,
        role="evaluation",
        open_loop=False,
    )
    kalman_evaluation_open_loop = score_identity_observation_kalman(
        data,
        authority,
        matrix,
        bias,
        process_variance,
        measurement_variance,
        role="evaluation",
        open_loop=True,
    )

    return ProbabilisticStateSpaceResult(
        authority_fingerprint=authority.authority_fingerprint,
        data_sha256=data.data_sha256,
        fit_transition_sha256=_transition_sha256(fit_pairs),
        calibration_transition_sha256=_transition_sha256(calibration_pairs),
        evaluation_transition_sha256=_transition_sha256(evaluation_pairs),
        transition=matrix,
        intercept=bias,
        base_innovation_variance=base_variance,
        selected_q_scale=selected.q_scale,
        selected_r_scale=selected.r_scale,
        process_variance=process_variance,
        measurement_variance=measurement_variance,
        calibration_candidates=tuple(item[1] for item in candidates),
        direct_calibration_sequential=direct_calibration,
        kalman_calibration_sequential=calibration_metrics,
        direct_evaluation_sequential=direct_evaluation_sequential,
        direct_evaluation_open_loop=direct_evaluation_open_loop,
        kalman_evaluation_sequential=kalman_evaluation_sequential,
        kalman_evaluation_open_loop=kalman_evaluation_open_loop,
    )
