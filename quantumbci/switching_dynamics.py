"""Dependency-light Markov-switching affine dynamics for E002.

v0.12 adds a two-regime classical switching-state adversary. Regimes are latent
labels on transition dynamics, not biological states. The model is fit only on
source-fit transitions and scored sequentially after resetting regime belief at
every evidence-role trajectory boundary.

The first release intentionally does not claim an exact open-loop switching
forecast. Exact state/regime path mixtures grow exponentially with horizon, so
sequential Gaussian-mixture predictive density is the qualified surface here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .dynamics_fitting import PHYSICALITY_TOLERANCE
from .trajectory_authority import TrajectoryEvidenceAuthority, TrajectoryEvidenceData

Array = np.ndarray
SWITCHING_MODEL_ID = "two_regime_markov_switching_affine_var1_v1"
SWITCHING_SCORE_ID = "sequential_gaussian_mixture_nll_v1"
REGIME_COUNT = 2
EM_MAX_ITERATIONS = 100
EM_TOLERANCE = 1e-7
TRANSITION_PSEUDOCOUNT = 1e-3
MIN_EFFECTIVE_TRANSITIONS = 8.0
VARIANCE_FLOOR_RELATIVE = 1e-8
VARIANCE_FLOOR_ABSOLUTE = 1e-12
LABEL_CANONICALIZATION_ID = "lexicographic_rounded_regime_parameter_vector_v1"
INITIALIZATION_IDS = (
    "residual_pc_median",
    "state_pc_median",
    "delta_pc_median",
    "temporal_half",
)


def _logsumexp(values: Array, axis: int | None = None) -> Array:
    x = np.asarray(values, dtype=float)
    maximum = np.max(x, axis=axis, keepdims=True)
    shifted = np.exp(x - maximum)
    result = maximum + np.log(np.sum(shifted, axis=axis, keepdims=True))
    if axis is None:
        return np.asarray(result.squeeze(), dtype=float)
    return np.asarray(np.squeeze(result, axis=axis), dtype=float)


def _pairs_xy(data: TrajectoryEvidenceData, pairs: Array) -> tuple[Array, Array]:
    values = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if len(values) == 0:
        raise ValueError("switching-state dynamics require at least one transition")
    return (
        np.asarray(data.states[values[:, 0]], dtype=float),
        np.asarray(data.states[values[:, 1]], dtype=float),
    )


def _transition_row_chains(pairs: Array) -> list[np.ndarray]:
    """Return row-index chains for a legal transition-pair graph."""

    values = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if len(values) == 0:
        raise ValueError("switching-state dynamics require at least one transition")
    row_by_left: dict[int, int] = {}
    predecessor_row: dict[int, int] = {}
    next_row: dict[int, int] = {}
    for row, (left, right) in enumerate(values.tolist()):
        left_i, right_i = int(left), int(right)
        if left_i in row_by_left:
            raise RuntimeError("transition graph contains branching left nodes")
        row_by_left[left_i] = row
    for row, (_, right) in enumerate(values.tolist()):
        right_i = int(right)
        if right_i in row_by_left:
            following = row_by_left[right_i]
            if following in predecessor_row:
                raise RuntimeError("transition graph contains merging transition rows")
            next_row[row] = following
            predecessor_row[following] = row

    starts = sorted(row for row in range(len(values)) if row not in predecessor_row)
    chains: list[np.ndarray] = []
    visited: set[int] = set()
    for start in starts:
        rows: list[int] = []
        current = start
        while True:
            if current in visited:
                raise RuntimeError("transition graph contains a cycle")
            visited.add(current)
            rows.append(current)
            if current not in next_row:
                break
            current = next_row[current]
        chains.append(np.asarray(rows, dtype=np.int64))
    if len(visited) != len(values):
        raise RuntimeError("transition graph is not a set of simple forward chains")
    return chains


def _design(states: Array) -> Array:
    x = np.asarray(states, dtype=float)
    if x.ndim != 2 or len(x) == 0:
        raise ValueError("states must be a non-empty 2D array")
    return np.concatenate([x, np.ones((len(x), 1), dtype=float)], axis=1)


def _weighted_affine_fit(x: Array, y: Array, weights: Array) -> tuple[Array, Array, Array, float]:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if x_values.shape != y_values.shape or len(x_values) != len(w):
        raise ValueError("weighted affine fit inputs are misaligned")
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite and non-negative")
    effective = float(np.sum(w))
    if effective < MIN_EFFECTIVE_TRANSITIONS:
        raise ValueError("regime has too few effective fit transitions")
    design = _design(x_values)
    sqrt_w = np.sqrt(w)[:, None]
    coefficients = np.linalg.lstsq(design * sqrt_w, y_values * sqrt_w, rcond=None)[0]
    transition = coefficients[:-1].T
    intercept = np.asarray(coefficients[-1], dtype=float).reshape(-1)
    residual = y_values - (x_values @ transition.T + intercept)
    state_scale = np.average(x_values**2, axis=0, weights=w)
    variance_floor = np.maximum(
        state_scale * VARIANCE_FLOOR_RELATIVE,
        VARIANCE_FLOOR_ABSOLUTE,
    )
    variance = np.maximum(
        np.average(residual**2, axis=0, weights=w),
        variance_floor,
    )
    return transition, intercept, variance, effective


def _emission_log_probabilities(
    x: Array,
    y: Array,
    transitions: Array,
    intercepts: Array,
    variances: Array,
) -> Array:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    n_regimes = int(len(transitions))
    output = np.empty((len(x_values), n_regimes), dtype=float)
    normalizer = x_values.shape[1] * np.log(2.0 * np.pi)
    for regime in range(n_regimes):
        variance = np.asarray(variances[regime], dtype=float)
        if np.any(variance <= 0) or not np.all(np.isfinite(variance)):
            raise ValueError("regime innovation variances must be finite and positive")
        mean = x_values @ transitions[regime].T + intercepts[regime]
        error = y_values - mean
        output[:, regime] = -0.5 * (
            normalizer
            + float(np.sum(np.log(variance)))
            + np.sum(error**2 / variance, axis=1)
        )
    return output


def _forward_backward(
    emission_log_prob: Array,
    chains: Iterable[Array],
    initial_probabilities: Array,
    regime_transition: Array,
) -> tuple[float, Array, Array, Array]:
    emissions = np.asarray(emission_log_prob, dtype=float)
    pi = np.asarray(initial_probabilities, dtype=float).reshape(-1)
    transition = np.asarray(regime_transition, dtype=float)
    n_regimes = len(pi)
    if transition.shape != (n_regimes, n_regimes):
        raise ValueError("regime transition matrix shape is invalid")
    if np.any(pi <= 0) or np.any(transition <= 0):
        raise ValueError("initial and regime-transition probabilities must be positive")
    log_pi = np.log(pi)
    log_transition = np.log(transition)
    gamma = np.zeros_like(emissions)
    transition_counts = np.zeros_like(transition)
    start_counts = np.zeros(n_regimes, dtype=float)
    total_log_likelihood = 0.0

    for chain in chains:
        rows = np.asarray(chain, dtype=np.int64).reshape(-1)
        sequence = emissions[rows]
        length = len(rows)
        log_alpha = np.empty((length, n_regimes), dtype=float)
        log_beta = np.empty((length, n_regimes), dtype=float)
        log_alpha[0] = log_pi + sequence[0]
        for time in range(1, length):
            log_alpha[time] = sequence[time] + _logsumexp(
                log_alpha[time - 1][:, None] + log_transition,
                axis=0,
            )
        sequence_log_likelihood = float(_logsumexp(log_alpha[-1]))
        total_log_likelihood += sequence_log_likelihood

        log_beta[-1] = 0.0
        for time in range(length - 2, -1, -1):
            log_beta[time] = _logsumexp(
                log_transition
                + sequence[time + 1][None, :]
                + log_beta[time + 1][None, :],
                axis=1,
            )
        sequence_gamma = np.exp(
            log_alpha + log_beta - sequence_log_likelihood
        )
        sequence_gamma /= np.sum(sequence_gamma, axis=1, keepdims=True)
        gamma[rows] = sequence_gamma
        start_counts += sequence_gamma[0]

        for time in range(length - 1):
            log_xi = (
                log_alpha[time][:, None]
                + log_transition
                + sequence[time + 1][None, :]
                + log_beta[time + 1][None, :]
                - sequence_log_likelihood
            )
            xi = np.exp(log_xi)
            xi /= np.sum(xi)
            transition_counts += xi

    return total_log_likelihood, gamma, transition_counts, start_counts


def _principal_score(values: Array) -> Array:
    x = np.asarray(values, dtype=float)
    centered = x - np.mean(x, axis=0, keepdims=True)
    if not np.any(np.abs(centered) > 1e-15):
        return np.arange(len(x), dtype=float)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    # SVD signs are arbitrary. Fix the sign from the largest-magnitude loading.
    anchor = int(np.argmax(np.abs(direction)))
    if direction[anchor] < 0:
        direction = -direction
    return centered @ direction


def _initial_responsibilities(
    initialization_id: str,
    x: Array,
    y: Array,
    chains: list[Array],
) -> Array:
    n = len(x)
    if initialization_id == "residual_pc_median":
        global_transition, global_intercept, _, _ = _weighted_affine_fit(
            x, y, np.ones(n, dtype=float)
        )
        residual = y - (x @ global_transition.T + global_intercept)
        score = _principal_score(residual)
    elif initialization_id == "state_pc_median":
        score = _principal_score(x)
    elif initialization_id == "delta_pc_median":
        score = _principal_score(y - x)
    elif initialization_id == "temporal_half":
        score = np.zeros(n, dtype=float)
        for chain in chains:
            rows = np.asarray(chain, dtype=np.int64)
            midpoint = max(1, len(rows) // 2)
            score[rows[:midpoint]] = -1.0
            score[rows[midpoint:]] = 1.0
    else:
        raise ValueError(f"unknown switching initialization: {initialization_id}")

    threshold = float(np.median(score))
    hard = score >= threshold
    if np.all(hard) or not np.any(hard):
        order = np.argsort(score, kind="stable")
        hard = np.zeros(n, dtype=bool)
        hard[order[n // 2 :]] = True
    responsibilities = np.empty((n, REGIME_COUNT), dtype=float)
    responsibilities[:, 0] = np.where(hard, 0.1, 0.9)
    responsibilities[:, 1] = 1.0 - responsibilities[:, 0]
    return responsibilities


def _m_step(
    x: Array,
    y: Array,
    gamma: Array,
    transition_counts: Array | None,
    start_counts: Array | None,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    transitions: list[Array] = []
    intercepts: list[Array] = []
    variances: list[Array] = []
    effective: list[float] = []
    for regime in range(REGIME_COUNT):
        transition, intercept, variance, count = _weighted_affine_fit(
            x,
            y,
            gamma[:, regime],
        )
        transitions.append(transition)
        intercepts.append(intercept)
        variances.append(variance)
        effective.append(count)

    if transition_counts is None:
        regime_transition = np.asarray([[0.95, 0.05], [0.05, 0.95]], dtype=float)
    else:
        counts = np.asarray(transition_counts, dtype=float) + TRANSITION_PSEUDOCOUNT
        regime_transition = counts / np.sum(counts, axis=1, keepdims=True)
    if start_counts is None:
        initial_probabilities = np.asarray([0.5, 0.5], dtype=float)
    else:
        starts = np.asarray(start_counts, dtype=float) + TRANSITION_PSEUDOCOUNT
        initial_probabilities = starts / np.sum(starts)
    return (
        np.asarray(transitions, dtype=float),
        np.asarray(intercepts, dtype=float),
        np.asarray(variances, dtype=float),
        regime_transition,
        initial_probabilities,
        np.asarray(effective, dtype=float),
    )


def _regime_key(transition: Array, intercept: Array, variance: Array) -> tuple[float, ...]:
    vector = np.concatenate(
        [
            np.asarray(transition, dtype=float).reshape(-1),
            np.asarray(intercept, dtype=float).reshape(-1),
            np.asarray(variance, dtype=float).reshape(-1),
        ]
    )
    return tuple(float(value) for value in np.round(vector, decimals=12))


def _canonicalize_labels(
    transitions: Array,
    intercepts: Array,
    variances: Array,
    regime_transition: Array,
    initial_probabilities: Array,
    effective_counts: Array,
) -> tuple[Array, Array, Array, Array, Array, Array, tuple[int, ...]]:
    order = tuple(
        sorted(
            range(REGIME_COUNT),
            key=lambda regime: _regime_key(
                transitions[regime], intercepts[regime], variances[regime]
            ),
        )
    )
    index = np.asarray(order, dtype=int)
    return (
        transitions[index],
        intercepts[index],
        variances[index],
        regime_transition[np.ix_(index, index)],
        initial_probabilities[index],
        effective_counts[index],
        order,
    )


@dataclass(frozen=True)
class SwitchingFitResult:
    transitions: Array
    intercepts: Array
    variances: Array
    regime_transition: Array
    initial_probabilities: Array
    effective_transition_counts: Array
    fit_log_likelihood: float
    fit_mean_nll: float
    iterations: int
    converged: bool
    initialization_id: str
    initialization_log_likelihoods: tuple[tuple[str, float], ...]
    canonicalization_permutation: tuple[int, ...]

    @property
    def parameter_count(self) -> int:
        dimension = int(self.intercepts.shape[1])
        emission = REGIME_COUNT * (dimension * dimension + dimension + dimension)
        markov = REGIME_COUNT * (REGIME_COUNT - 1) + (REGIME_COUNT - 1)
        return int(emission + markov)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "model_id": SWITCHING_MODEL_ID,
            "regime_count": REGIME_COUNT,
            "transitions": np.asarray(self.transitions, dtype=float).tolist(),
            "intercepts": np.asarray(self.intercepts, dtype=float).tolist(),
            "innovation_variances": np.asarray(self.variances, dtype=float).tolist(),
            "regime_transition": np.asarray(self.regime_transition, dtype=float).tolist(),
            "initial_probabilities": np.asarray(
                self.initial_probabilities, dtype=float
            ).tolist(),
            "effective_transition_counts": np.asarray(
                self.effective_transition_counts, dtype=float
            ).tolist(),
            "fit_log_likelihood": float(self.fit_log_likelihood),
            "fit_mean_nll": float(self.fit_mean_nll),
            "iterations": int(self.iterations),
            "converged": bool(self.converged),
            "selected_initialization": self.initialization_id,
            "initialization_log_likelihoods": [
                {"initialization_id": name, "fit_log_likelihood": float(value)}
                for name, value in self.initialization_log_likelihoods
            ],
            "parameter_count": self.parameter_count,
            "label_canonicalization_id": LABEL_CANONICALIZATION_ID,
            "canonicalization_permutation": [
                int(value) for value in self.canonicalization_permutation
            ],
            "regime_labels_mechanistically_identifiable": False,
        }


def _fit_from_initialization(
    x: Array,
    y: Array,
    chains: list[Array],
    initialization_id: str,
) -> SwitchingFitResult:
    gamma = _initial_responsibilities(initialization_id, x, y, chains)
    (
        transitions,
        intercepts,
        variances,
        regime_transition,
        initial_probabilities,
        effective_counts,
    ) = _m_step(x, y, gamma, None, None)

    previous_log_likelihood = -np.inf
    converged = False
    iterations = 0
    for iteration in range(1, EM_MAX_ITERATIONS + 1):
        emissions = _emission_log_probabilities(
            x, y, transitions, intercepts, variances
        )
        (
            log_likelihood,
            gamma,
            transition_counts,
            start_counts,
        ) = _forward_backward(
            emissions,
            chains,
            initial_probabilities,
            regime_transition,
        )
        if np.isfinite(previous_log_likelihood):
            improvement = log_likelihood - previous_log_likelihood
            tolerance = EM_TOLERANCE * (1.0 + abs(previous_log_likelihood))
            if improvement < -max(1e-7, 10.0 * tolerance):
                raise RuntimeError("switching EM log likelihood decreased unexpectedly")
            if abs(improvement) <= tolerance:
                converged = True
                iterations = iteration
                break
        previous_log_likelihood = log_likelihood
        (
            transitions,
            intercepts,
            variances,
            regime_transition,
            initial_probabilities,
            effective_counts,
        ) = _m_step(
            x,
            y,
            gamma,
            transition_counts,
            start_counts,
        )
        iterations = iteration

    emissions = _emission_log_probabilities(x, y, transitions, intercepts, variances)
    final_log_likelihood, gamma, transition_counts, start_counts = _forward_backward(
        emissions,
        chains,
        initial_probabilities,
        regime_transition,
    )
    # Synchronize Markov probabilities with the final posterior once more.
    (
        transitions,
        intercepts,
        variances,
        regime_transition,
        initial_probabilities,
        effective_counts,
    ) = _m_step(x, y, gamma, transition_counts, start_counts)
    emissions = _emission_log_probabilities(x, y, transitions, intercepts, variances)
    final_log_likelihood, _, _, _ = _forward_backward(
        emissions,
        chains,
        initial_probabilities,
        regime_transition,
    )

    (
        transitions,
        intercepts,
        variances,
        regime_transition,
        initial_probabilities,
        effective_counts,
        permutation,
    ) = _canonicalize_labels(
        transitions,
        intercepts,
        variances,
        regime_transition,
        initial_probabilities,
        effective_counts,
    )
    return SwitchingFitResult(
        transitions=transitions,
        intercepts=intercepts,
        variances=variances,
        regime_transition=regime_transition,
        initial_probabilities=initial_probabilities,
        effective_transition_counts=effective_counts,
        fit_log_likelihood=float(final_log_likelihood),
        fit_mean_nll=float(-final_log_likelihood / len(x)),
        iterations=int(iterations),
        converged=bool(converged),
        initialization_id=initialization_id,
        initialization_log_likelihoods=(),
        canonicalization_permutation=permutation,
    )


def fit_switching_affine_var(
    data: TrajectoryEvidenceData,
    authority: TrajectoryEvidenceAuthority,
) -> SwitchingFitResult:
    """Fit a deterministic multi-start two-regime switching affine VAR on fit authority."""

    authority.restore(data)
    pairs = authority.transition_pairs(data, "fit")
    x, y = _pairs_xy(data, pairs)
    chains = _transition_row_chains(pairs)
    candidates: list[SwitchingFitResult] = []
    failures: list[str] = []
    for initialization_id in INITIALIZATION_IDS:
        try:
            candidates.append(
                _fit_from_initialization(x, y, chains, initialization_id)
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            failures.append(f"{initialization_id}: {exc}")
    if not candidates:
        raise RuntimeError(
            "all switching-state initializations failed: " + "; ".join(failures)
        )
    candidates.sort(
        key=lambda result: (-result.fit_log_likelihood, result.initialization_id)
    )
    best = candidates[0]
    ledger = tuple(
        (result.initialization_id, result.fit_log_likelihood)
        for result in sorted(candidates, key=lambda result: result.initialization_id)
    )
    return SwitchingFitResult(
        transitions=best.transitions,
        intercepts=best.intercepts,
        variances=best.variances,
        regime_transition=best.regime_transition,
        initial_probabilities=best.initial_probabilities,
        effective_transition_counts=best.effective_transition_counts,
        fit_log_likelihood=best.fit_log_likelihood,
        fit_mean_nll=best.fit_mean_nll,
        iterations=best.iterations,
        converged=best.converged,
        initialization_id=best.initialization_id,
        initialization_log_likelihoods=ledger,
        canonicalization_permutation=best.canonicalization_permutation,
    )


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


@dataclass(frozen=True)
class SwitchingPredictiveMetrics:
    n_transitions: int
    mean_nll: float
    total_nll: float
    predictive_mean_rmse: float
    predictive_mean_mae: float
    mean_predictive_regime_entropy: float
    mean_max_predictive_regime_probability: float
    mean_bloch_half_l2: float | None
    prediction_physical_fraction: float | None
    target_physical_fraction: float | None
    valid_qubit_pair_fraction: float | None
    mean_valid_qubit_trace_distance: float | None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "score_id": SWITCHING_SCORE_ID,
            "n_transitions": int(self.n_transitions),
            "mean_nll": float(self.mean_nll),
            "total_nll": float(self.total_nll),
            "predictive_mean_rmse": float(self.predictive_mean_rmse),
            "predictive_mean_mae": float(self.predictive_mean_mae),
            "mean_predictive_regime_entropy": float(
                self.mean_predictive_regime_entropy
            ),
            "mean_max_predictive_regime_probability": float(
                self.mean_max_predictive_regime_probability
            ),
            "mean_bloch_half_l2": self.mean_bloch_half_l2,
            "prediction_physical_fraction": self.prediction_physical_fraction,
            "target_physical_fraction": self.target_physical_fraction,
            "valid_qubit_pair_fraction": self.valid_qubit_pair_fraction,
            "mean_valid_qubit_trace_distance": self.mean_valid_qubit_trace_distance,
        }


def _single_emission_log_probability(
    target: Array,
    mean: Array,
    variance: Array,
) -> float:
    error = np.asarray(target, dtype=float) - np.asarray(mean, dtype=float)
    var = np.asarray(variance, dtype=float)
    return float(
        -0.5
        * (
            len(error) * np.log(2.0 * np.pi)
            + np.sum(np.log(var))
            + np.sum(error**2 / var)
        )
    )


def score_switching_affine_var(
    data: TrajectoryEvidenceData,
    authority: TrajectoryEvidenceAuthority,
    model: SwitchingFitResult,
    *,
    role: str,
) -> SwitchingPredictiveMetrics:
    """Sequentially score a fitted switching model on one evidence role."""

    authority.restore(data)
    if role not in {"fit", "calibration", "evaluation"}:
        raise ValueError("role must be fit, calibration, or evaluation")
    pairs = authority.transition_pairs(data, role)  # type: ignore[arg-type]
    x, y = _pairs_xy(data, pairs)
    chains = _transition_row_chains(pairs)
    predictions: list[Array] = []
    targets: list[Array] = []
    nll_values: list[float] = []
    entropies: list[float] = []
    max_probabilities: list[float] = []

    for chain in chains:
        predictive_probabilities = np.asarray(
            model.initial_probabilities, dtype=float
        ).copy()
        for row in np.asarray(chain, dtype=np.int64):
            means = np.asarray(
                [
                    model.transitions[regime] @ x[row] + model.intercepts[regime]
                    for regime in range(REGIME_COUNT)
                ],
                dtype=float,
            )
            log_emissions = np.asarray(
                [
                    _single_emission_log_probability(
                        y[row], means[regime], model.variances[regime]
                    )
                    for regime in range(REGIME_COUNT)
                ],
                dtype=float,
            )
            log_joint = np.log(predictive_probabilities) + log_emissions
            log_normalizer = float(_logsumexp(log_joint))
            nll_values.append(-log_normalizer)
            predictions.append(predictive_probabilities @ means)
            targets.append(y[row].copy())
            safe_probabilities = np.maximum(predictive_probabilities, 1e-300)
            entropies.append(
                float(-np.sum(safe_probabilities * np.log(safe_probabilities)))
            )
            max_probabilities.append(float(np.max(predictive_probabilities)))

            posterior = np.exp(log_joint - log_normalizer)
            posterior /= np.sum(posterior)
            predictive_probabilities = posterior @ model.regime_transition
            predictive_probabilities = np.maximum(predictive_probabilities, 1e-300)
            predictive_probabilities /= np.sum(predictive_probabilities)

    prediction_array = np.asarray(predictions, dtype=float)
    target_array = np.asarray(targets, dtype=float)
    error = prediction_array - target_array
    qubit = _qubit_mean_metrics(prediction_array, target_array)
    return SwitchingPredictiveMetrics(
        n_transitions=int(len(prediction_array)),
        mean_nll=float(np.mean(nll_values)),
        total_nll=float(np.sum(nll_values)),
        predictive_mean_rmse=float(np.sqrt(np.mean(error**2))),
        predictive_mean_mae=float(np.mean(np.abs(error))),
        mean_predictive_regime_entropy=float(np.mean(entropies)),
        mean_max_predictive_regime_probability=float(np.mean(max_probabilities)),
        mean_bloch_half_l2=qubit["mean_bloch_half_l2"],
        prediction_physical_fraction=qubit["prediction_physical_fraction"],
        target_physical_fraction=qubit["target_physical_fraction"],
        valid_qubit_pair_fraction=qubit["valid_qubit_pair_fraction"],
        mean_valid_qubit_trace_distance=qubit["mean_valid_qubit_trace_distance"],
    )


@dataclass(frozen=True)
class SwitchingStateControlResult:
    authority_fingerprint: str
    data_sha256: str
    model: SwitchingFitResult
    calibration_metrics: SwitchingPredictiveMetrics | None
    evaluation_metrics: SwitchingPredictiveMetrics

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "experiment": "E002",
            "claim_class": "classical_control",
            "authority_fingerprint": self.authority_fingerprint,
            "data_sha256": self.data_sha256,
            "model": self.model.to_mapping(),
            "calibration_metrics": (
                None
                if self.calibration_metrics is None
                else self.calibration_metrics.to_mapping()
            ),
            "evaluation_metrics": self.evaluation_metrics.to_mapping(),
            "fit_authority_only": True,
            "calibration_used_for_model_selection": False,
            "evaluation_used_for_model_selection": False,
            "role_boundary_regime_belief_reset": True,
            "sequential_predictive_density_complete": True,
            "exact_open_loop_switching_forecast_complete": False,
            "open_loop_promotion_eligible": False,
            "regime_labels_mechanistically_identifiable": False,
            "interpretation": (
                "The two latent regime labels are exchangeable statistical components. "
                "Sequential predictive gains support a classical switching-dynamics "
                "adversary; they do not identify biological states or quantum mechanisms."
            ),
        }


def run_switching_state_control(
    data: TrajectoryEvidenceData,
    authority: TrajectoryEvidenceAuthority,
) -> SwitchingStateControlResult:
    """Fit on source authority and evaluate the v0.12 switching-state control."""

    authority.restore(data)
    model = fit_switching_affine_var(data, authority)
    calibration_pairs = authority.transition_pairs(data, "calibration")
    calibration_metrics = (
        score_switching_affine_var(data, authority, model, role="calibration")
        if len(calibration_pairs) > 0
        else None
    )
    evaluation_metrics = score_switching_affine_var(
        data, authority, model, role="evaluation"
    )
    return SwitchingStateControlResult(
        authority_fingerprint=authority.authority_fingerprint,
        data_sha256=data.data_sha256,
        model=model,
        calibration_metrics=calibration_metrics,
        evaluation_metrics=evaluation_metrics,
    )
