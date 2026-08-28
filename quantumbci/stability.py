"""Trajectory-block bootstrap stability evidence for E002.

This module deliberately separates single-case bootstrap stability from repeated-
case reliability. A bootstrap distribution from one trajectory case is *not* an
intraclass correlation coefficient (ICC). v0.14 therefore reports percentile
intervals, sign consistency, selection frequencies, predictive-gain survival and
explicit failure ledgers. ICC is reserved for genuinely repeated participant/case
estimates in a later cross-case layer.

Source fit and calibration trajectories are resampled independently at the
trajectory-block level. Final evaluation trajectories are copied exactly once and
remain read-only. Every bootstrap replicate reconstructs the fitted canonical,
affine and nonlinear models from its resampled source evidence, then scores on the
same untouched evaluation values.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from .classical_dynamics import fit_full_affine_var1
from .dynamics_fitting import run_matched_qubit_dynamics_benchmark
from .nonlinear_dynamics import run_nonlinear_residual_control
from .probabilistic_ssm import fit_base_innovation_variance, score_direct_gaussian_var
from .trajectory_authority import TrajectoryEvidenceAuthority, TrajectoryEvidenceData

Array = np.ndarray
BOOTSTRAP_METHOD_ID = "role_stratified_trajectory_block_bootstrap_v1"
DEFAULT_BOOTSTRAP_SEED = 1401
DEFAULT_BOOTSTRAP_REPLICATES = 200
DEFAULT_MIN_SUCCESS_FRACTION = 0.90
PERCENTILE_INTERVAL = (2.5, 97.5)


def _role_groups(
    data: TrajectoryEvidenceData,
    indices: tuple[int, ...],
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index in indices:
        groups.setdefault(str(data.trajectory_ids[index]), []).append(int(index))
    for values in groups.values():
        values.sort(key=lambda i: float(data.start_times_s[i]))
    return groups


def _draw_sha256(fit_ids: list[str], calibration_ids: list[str]) -> str:
    payload = {
        "fit": fit_ids,
        "calibration": calibration_ids,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(b"quantumbci.e002.bootstrap-draw.v1\0" + canonical).hexdigest()


def _append_rows(
    *,
    data: TrajectoryEvidenceData,
    source_rows: list[int],
    trajectory_id: str,
    states: list[Array],
    trajectory_ids: list[str],
    starts: list[float],
    stops: list[float],
    valid: list[bool],
) -> list[int]:
    output: list[int] = []
    for original in source_rows:
        output.append(len(states))
        states.append(np.asarray(data.states[original]).copy())
        trajectory_ids.append(trajectory_id)
        starts.append(float(data.start_times_s[original]))
        stops.append(float(data.stop_times_s[original]))
        valid.append(bool(data.valid_mask[original]))
    return output


def _bootstrap_case(
    data: TrajectoryEvidenceData,
    authority: TrajectoryEvidenceAuthority,
    *,
    rng: np.random.Generator,
    replicate: int,
) -> tuple[TrajectoryEvidenceData, TrajectoryEvidenceAuthority, str]:
    """Construct one role-stratified trajectory-block bootstrap case.

    Fit and calibration trajectory IDs are sampled independently with replacement.
    Evaluation rows are copied exactly once. New role-specific trajectory IDs prevent
    artificial cross-role adjacency while preserving every within-role timestamp and
    gap from the frozen parent evidence.
    """

    authority.restore(data)
    fit_groups = _role_groups(data, authority.fit_indices)
    calibration_groups = _role_groups(data, authority.calibration_indices)
    evaluation_groups = _role_groups(data, authority.evaluation_indices)
    if len(fit_groups) < 2:
        raise ValueError("bootstrap stability requires at least two fit trajectories")
    if len(calibration_groups) < 2:
        raise ValueError("bootstrap stability requires at least two calibration trajectories")
    if not evaluation_groups:
        raise ValueError("bootstrap stability requires evaluation trajectories")

    fit_keys = sorted(fit_groups)
    calibration_keys = sorted(calibration_groups)
    fit_draw = [
        fit_keys[int(index)]
        for index in rng.integers(0, len(fit_keys), size=len(fit_keys)).tolist()
    ]
    calibration_draw = [
        calibration_keys[int(index)]
        for index in rng.integers(
            0, len(calibration_keys), size=len(calibration_keys)
        ).tolist()
    ]
    draw_sha = _draw_sha256(fit_draw, calibration_draw)

    states: list[Array] = []
    trajectory_ids: list[str] = []
    starts: list[float] = []
    stops: list[float] = []
    valid: list[bool] = []
    fit_indices: list[int] = []
    calibration_indices: list[int] = []
    evaluation_indices: list[int] = []
    representation_indices: list[int] = []
    representation_parent = set(authority.representation_fit_indices)

    for draw_index, original_id in enumerate(fit_draw):
        rows = fit_groups[original_id]
        new_rows = _append_rows(
            data=data,
            source_rows=rows,
            trajectory_id=f"bootstrap-fit-{replicate}-{draw_index}",
            states=states,
            trajectory_ids=trajectory_ids,
            starts=starts,
            stops=stops,
            valid=valid,
        )
        fit_indices.extend(new_rows)
        representation_indices.extend(
            new_index
            for new_index, original in zip(new_rows, rows)
            if original in representation_parent
        )

    for draw_index, original_id in enumerate(calibration_draw):
        rows = calibration_groups[original_id]
        calibration_indices.extend(
            _append_rows(
                data=data,
                source_rows=rows,
                trajectory_id=f"bootstrap-calibration-{replicate}-{draw_index}",
                states=states,
                trajectory_ids=trajectory_ids,
                starts=starts,
                stops=stops,
                valid=valid,
            )
        )

    for eval_index, original_id in enumerate(sorted(evaluation_groups)):
        evaluation_indices.extend(
            _append_rows(
                data=data,
                source_rows=evaluation_groups[original_id],
                trajectory_id=f"fixed-evaluation-{eval_index}",
                states=states,
                trajectory_ids=trajectory_ids,
                starts=starts,
                stops=stops,
                valid=valid,
            )
        )

    if not representation_indices:
        raise RuntimeError("bootstrap replicate lost all representation-fit authority")

    boot_data = TrajectoryEvidenceData(
        dataset_id=f"{data.dataset_id}::bootstrap",
        states=np.asarray(states, dtype=float),
        trajectory_ids=np.asarray(trajectory_ids),
        start_times_s=np.asarray(starts, dtype=float),
        stop_times_s=np.asarray(stops, dtype=float),
        valid_mask=np.asarray(valid, dtype=bool),
        metadata={
            **dict(data.metadata),
            "bootstrap_method_id": BOOTSTRAP_METHOD_ID,
            "parent_data_sha256": data.data_sha256,
            "replicate": int(replicate),
            "source_draw_sha256": draw_sha,
        },
    )
    boot_authority = TrajectoryEvidenceAuthority.from_data(
        boot_data,
        case_id=f"{authority.case_id}::bootstrap-{replicate}",
        fit_indices=fit_indices,
        calibration_indices=calibration_indices,
        evaluation_indices=evaluation_indices,
        representation_fit_indices=representation_indices,
        latent_dimension=authority.latent_dimension,
        time_step_policy=authority.time_step_policy,
        expected_window_seconds=authority.expected_window_seconds,
        expected_step_seconds=authority.expected_step_seconds,
        step_tolerance_seconds=authority.step_tolerance_seconds,
        purge_seconds=authority.purge_seconds,
        upstream_authority_fingerprint=authority.authority_fingerprint,
        source_revisions={
            **dict(authority.source_revisions),
            "bootstrap_method": BOOTSTRAP_METHOD_ID,
        },
        case_metadata={
            "bootstrap_parent_case_id": authority.case_id,
            "bootstrap_replicate": int(replicate),
        },
    )
    return boot_data, boot_authority, draw_sha


@dataclass(frozen=True)
class BootstrapScalarSummary:
    point_estimate: float
    bootstrap_mean: float
    bootstrap_median: float
    bootstrap_std: float
    ci_low: float
    ci_high: float
    finite_fraction: float
    sign_consistency: float
    positive_fraction: float
    relative_ci_width: float | None
    zero_excluded: bool

    def to_mapping(self) -> dict[str, Any]:
        return {
            "point_estimate": float(self.point_estimate),
            "bootstrap_mean": float(self.bootstrap_mean),
            "bootstrap_median": float(self.bootstrap_median),
            "bootstrap_std": float(self.bootstrap_std),
            "ci_percentiles": list(PERCENTILE_INTERVAL),
            "ci_low": float(self.ci_low),
            "ci_high": float(self.ci_high),
            "finite_fraction": float(self.finite_fraction),
            "sign_consistency": float(self.sign_consistency),
            "positive_fraction": float(self.positive_fraction),
            "relative_ci_width": self.relative_ci_width,
            "zero_excluded": bool(self.zero_excluded),
        }


def _scalar_summary(point: float, values: list[float]) -> BootstrapScalarSummary:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        raise ValueError("cannot summarize an all-nonfinite bootstrap quantity")
    low, high = np.percentile(finite, PERCENTILE_INTERVAL)
    tolerance = max(1e-12, abs(float(point)) * 1e-12)
    if point > tolerance:
        sign_consistency = float(np.mean(finite > 0.0))
    elif point < -tolerance:
        sign_consistency = float(np.mean(finite < 0.0))
    else:
        sign_consistency = float(np.mean(np.abs(finite) <= tolerance))
    relative_width = (
        float((high - low) / abs(point)) if abs(point) > tolerance else None
    )
    return BootstrapScalarSummary(
        point_estimate=float(point),
        bootstrap_mean=float(np.mean(finite)),
        bootstrap_median=float(np.median(finite)),
        bootstrap_std=float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        ci_low=float(low),
        ci_high=float(high),
        finite_fraction=float(len(finite) / len(array)),
        sign_consistency=sign_consistency,
        positive_fraction=float(np.mean(finite > 0.0)),
        relative_ci_width=relative_width,
        zero_excluded=bool(low > 0.0 or high < 0.0),
    )


@dataclass(frozen=True)
class NonlinearSelectionStability:
    mode_feature_count: int
    mode_length_scale_multiplier: float
    mode_ridge: float
    mode_frequency: float
    unique_configurations: int
    configuration_frequencies: tuple[tuple[str, float], ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "mode": {
                "feature_count": int(self.mode_feature_count),
                "length_scale_multiplier": float(self.mode_length_scale_multiplier),
                "ridge": float(self.mode_ridge),
            },
            "mode_frequency": float(self.mode_frequency),
            "unique_configurations": int(self.unique_configurations),
            "configuration_frequencies": [
                {"configuration": key, "frequency": float(value)}
                for key, value in self.configuration_frequencies
            ],
        }


def _selection_key(feature_count: int, length_scale: float, ridge: float) -> str:
    return f"features={int(feature_count)}|length_scale={float(length_scale):.12g}|ridge={float(ridge):.12g}"


def _selection_summary(configurations: list[tuple[int, float, float]]) -> NonlinearSelectionStability:
    if not configurations:
        raise ValueError("nonlinear selection stability requires successful configurations")
    keys = [_selection_key(*configuration) for configuration in configurations]
    counts = Counter(keys)
    mode_key, mode_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    mode_configuration = configurations[keys.index(mode_key)]
    total = len(configurations)
    frequencies = tuple(
        (key, count / total) for key, count in sorted(counts.items())
    )
    return NonlinearSelectionStability(
        mode_feature_count=int(mode_configuration[0]),
        mode_length_scale_multiplier=float(mode_configuration[1]),
        mode_ridge=float(mode_configuration[2]),
        mode_frequency=float(mode_count / total),
        unique_configurations=len(counts),
        configuration_frequencies=frequencies,
    )


@dataclass(frozen=True)
class BootstrapReplicate:
    replicate: int
    source_draw_sha256: str
    status: str
    failure_reason: str | None
    omega_x: float | None
    omega_z: float | None
    gamma_dephasing: float | None
    gamma_relaxation: float | None
    canonical_structure_residual: float | None
    canonical_minus_affine_one_step_rmse: float | None
    canonical_minus_affine_rollout_rmse: float | None
    direct_minus_nonlinear_mean_nll: float | None
    direct_minus_nonlinear_one_step_rmse: float | None
    nonlinear_feature_count: int | None
    nonlinear_length_scale_multiplier: float | None
    nonlinear_ridge: float | None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "replicate": int(self.replicate),
            "source_draw_sha256": self.source_draw_sha256,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "canonical_parameters": (
                None
                if self.omega_x is None
                else {
                    "omega_x": self.omega_x,
                    "omega_z": self.omega_z,
                    "gamma_dephasing": self.gamma_dephasing,
                    "gamma_relaxation": self.gamma_relaxation,
                }
            ),
            "canonical_structure_residual": self.canonical_structure_residual,
            "canonical_minus_affine_one_step_rmse": self.canonical_minus_affine_one_step_rmse,
            "canonical_minus_affine_rollout_rmse": self.canonical_minus_affine_rollout_rmse,
            "direct_minus_nonlinear_mean_nll": self.direct_minus_nonlinear_mean_nll,
            "direct_minus_nonlinear_one_step_rmse": self.direct_minus_nonlinear_one_step_rmse,
            "nonlinear_selection": (
                None
                if self.nonlinear_feature_count is None
                else {
                    "feature_count": self.nonlinear_feature_count,
                    "length_scale_multiplier": self.nonlinear_length_scale_multiplier,
                    "ridge": self.nonlinear_ridge,
                }
            ),
        }


@dataclass(frozen=True)
class E002BootstrapStabilityResult:
    authority_fingerprint: str
    data_sha256: str
    bootstrap_seed: int
    requested_replicates: int
    minimum_success_fraction: float
    point_estimates: dict[str, float]
    parameter_summaries: dict[str, BootstrapScalarSummary]
    predictive_summaries: dict[str, BootstrapScalarSummary]
    nonlinear_selection_stability: NonlinearSelectionStability
    replicates: tuple[BootstrapReplicate, ...]

    @property
    def success_count(self) -> int:
        return sum(replicate.status == "success" for replicate in self.replicates)

    @property
    def failure_count(self) -> int:
        return len(self.replicates) - self.success_count

    @property
    def success_fraction(self) -> float:
        return self.success_count / len(self.replicates)

    @property
    def stability_evidence_complete(self) -> bool:
        return self.success_fraction >= self.minimum_success_fraction

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "experiment": "E002",
            "claim_class": "quantum_inspired",
            "artifact_role": "bootstrap_stability_evidence",
            "bootstrap_method_id": BOOTSTRAP_METHOD_ID,
            "bootstrap_seed": int(self.bootstrap_seed),
            "requested_replicates": int(self.requested_replicates),
            "minimum_success_fraction": float(self.minimum_success_fraction),
            "success_count": int(self.success_count),
            "failure_count": int(self.failure_count),
            "success_fraction": float(self.success_fraction),
            "authority_fingerprint": self.authority_fingerprint,
            "data_sha256": self.data_sha256,
            "evaluation_resampled": False,
            "fit_trajectory_blocks_resampled": True,
            "calibration_trajectory_blocks_resampled": True,
            "point_estimates": dict(self.point_estimates),
            "parameter_summaries": {
                key: summary.to_mapping()
                for key, summary in self.parameter_summaries.items()
            },
            "predictive_summaries": {
                key: summary.to_mapping()
                for key, summary in self.predictive_summaries.items()
            },
            "nonlinear_selection_stability": self.nonlinear_selection_stability.to_mapping(),
            "replicates": [replicate.to_mapping() for replicate in self.replicates],
            "single_case_bootstrap_is_icc": False,
            "participant_icc_computed": False,
            "stability_evidence_complete": bool(self.stability_evidence_complete),
            "intervention_direction_evidence_required": True,
            "intervention_stage_eligible": False,
            "physical_quantum_promotion_eligible": False,
            "interpretation": (
                "This artifact measures sensitivity of fitted E002 quantities to role-stratified "
                "trajectory-block resampling while final evaluation remains fixed. Percentile "
                "intervals and selection frequencies are single-case bootstrap evidence, not ICC. "
                "Stability can strengthen or falsify a proposed parameterization but cannot by "
                "itself identify a biological or physical-quantum mechanism; intervention-direction "
                "evidence remains required."
            ),
        }


def run_e002_bootstrap_stability(
    data: TrajectoryEvidenceData,
    authority: TrajectoryEvidenceAuthority,
    *,
    n_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    minimum_success_fraction: float = DEFAULT_MIN_SUCCESS_FRACTION,
) -> E002BootstrapStabilityResult:
    """Run deterministic role-stratified trajectory bootstrap stability evidence."""

    authority.restore(data)
    if data.state_dimension != 3:
        raise ValueError("E002 bootstrap stability currently requires 3D state trajectories")
    if len(authority.transition_pairs(data, "calibration")) == 0:
        raise ValueError("E002 bootstrap stability requires calibration transitions")
    if n_replicates < 2:
        raise ValueError("n_replicates must be at least 2")
    if not (0.0 < minimum_success_fraction <= 1.0):
        raise ValueError("minimum_success_fraction must lie in (0, 1]")

    point_matched = run_matched_qubit_dynamics_benchmark(data, authority, ridge=0.0)
    point_transition, point_intercept, _ = fit_full_affine_var1(data, authority)
    point_variance = fit_base_innovation_variance(
        data, authority, point_transition, point_intercept
    )
    point_direct = score_direct_gaussian_var(
        data,
        authority,
        point_transition,
        point_intercept,
        point_variance,
        role="evaluation",
        open_loop=False,
    )
    point_nonlinear = run_nonlinear_residual_control(
        data, authority, point_transition, point_intercept
    )
    point_parameters = point_matched.canonical.canonical_parameters
    if point_parameters is None:
        raise RuntimeError("canonical point fit did not expose canonical parameters")
    point_estimates = {
        **point_parameters.to_mapping(),
        "canonical_structure_residual": float(
            point_matched.canonical.canonical_structure_residual_to_affine
        ),
        "canonical_minus_affine_one_step_rmse": float(
            point_matched.canonical_minus_affine_one_step_rmse
        ),
        "canonical_minus_affine_rollout_rmse": float(
            point_matched.canonical_minus_affine_rollout_rmse
        ),
        "direct_minus_nonlinear_mean_nll": float(
            point_direct.mean_nll - point_nonlinear.evaluation_metrics.one_step_mean_nll
        ),
        "direct_minus_nonlinear_one_step_rmse": float(
            point_direct.predictive_mean_rmse
            - point_nonlinear.evaluation_metrics.one_step_rmse
        ),
    }

    rng = np.random.default_rng(seed)
    replicates: list[BootstrapReplicate] = []
    for replicate in range(n_replicates):
        draw_sha = ""
        try:
            boot_data, boot_authority, draw_sha = _bootstrap_case(
                data, authority, rng=rng, replicate=replicate
            )
            matched = run_matched_qubit_dynamics_benchmark(
                boot_data, boot_authority, ridge=0.0
            )
            parameters = matched.canonical.canonical_parameters
            if parameters is None:
                raise RuntimeError("bootstrap canonical fit omitted parameters")
            transition, intercept, _ = fit_full_affine_var1(boot_data, boot_authority)
            variance = fit_base_innovation_variance(
                boot_data, boot_authority, transition, intercept
            )
            direct = score_direct_gaussian_var(
                boot_data,
                boot_authority,
                transition,
                intercept,
                variance,
                role="evaluation",
                open_loop=False,
            )
            nonlinear = run_nonlinear_residual_control(
                boot_data, boot_authority, transition, intercept
            )
            replicates.append(
                BootstrapReplicate(
                    replicate=replicate,
                    source_draw_sha256=draw_sha,
                    status="success",
                    failure_reason=None,
                    omega_x=float(parameters.omega_x),
                    omega_z=float(parameters.omega_z),
                    gamma_dephasing=float(parameters.gamma_dephasing),
                    gamma_relaxation=float(parameters.gamma_relaxation),
                    canonical_structure_residual=float(
                        matched.canonical.canonical_structure_residual_to_affine
                    ),
                    canonical_minus_affine_one_step_rmse=float(
                        matched.canonical_minus_affine_one_step_rmse
                    ),
                    canonical_minus_affine_rollout_rmse=float(
                        matched.canonical_minus_affine_rollout_rmse
                    ),
                    direct_minus_nonlinear_mean_nll=float(
                        direct.mean_nll
                        - nonlinear.evaluation_metrics.one_step_mean_nll
                    ),
                    direct_minus_nonlinear_one_step_rmse=float(
                        direct.predictive_mean_rmse
                        - nonlinear.evaluation_metrics.one_step_rmse
                    ),
                    nonlinear_feature_count=int(nonlinear.model.feature_count),
                    nonlinear_length_scale_multiplier=float(
                        nonlinear.model.length_scale_multiplier
                    ),
                    nonlinear_ridge=float(nonlinear.model.ridge),
                )
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            replicates.append(
                BootstrapReplicate(
                    replicate=replicate,
                    source_draw_sha256=draw_sha,
                    status="failure",
                    failure_reason=str(exc),
                    omega_x=None,
                    omega_z=None,
                    gamma_dephasing=None,
                    gamma_relaxation=None,
                    canonical_structure_residual=None,
                    canonical_minus_affine_one_step_rmse=None,
                    canonical_minus_affine_rollout_rmse=None,
                    direct_minus_nonlinear_mean_nll=None,
                    direct_minus_nonlinear_one_step_rmse=None,
                    nonlinear_feature_count=None,
                    nonlinear_length_scale_multiplier=None,
                    nonlinear_ridge=None,
                )
            )

    successful = [replicate for replicate in replicates if replicate.status == "success"]
    success_fraction = len(successful) / len(replicates)
    if not successful:
        reasons = sorted(
            {replicate.failure_reason for replicate in replicates if replicate.failure_reason}
        )
        raise RuntimeError("all E002 bootstrap replicates failed: " + "; ".join(reasons))

    def values(name: str) -> list[float]:
        output: list[float] = []
        for replicate in successful:
            value = getattr(replicate, name)
            if value is None:
                raise RuntimeError(f"successful bootstrap replicate omitted {name}")
            output.append(float(value))
        return output

    parameter_summaries = {
        name: _scalar_summary(point_estimates[name], values(name))
        for name in (
            "omega_x",
            "omega_z",
            "gamma_dephasing",
            "gamma_relaxation",
            "canonical_structure_residual",
        )
    }
    predictive_summaries = {
        name: _scalar_summary(point_estimates[name], values(name))
        for name in (
            "canonical_minus_affine_one_step_rmse",
            "canonical_minus_affine_rollout_rmse",
            "direct_minus_nonlinear_mean_nll",
            "direct_minus_nonlinear_one_step_rmse",
        )
    }
    configurations = [
        (
            int(replicate.nonlinear_feature_count),
            float(replicate.nonlinear_length_scale_multiplier),
            float(replicate.nonlinear_ridge),
        )
        for replicate in successful
        if replicate.nonlinear_feature_count is not None
        and replicate.nonlinear_length_scale_multiplier is not None
        and replicate.nonlinear_ridge is not None
    ]
    result = E002BootstrapStabilityResult(
        authority_fingerprint=authority.authority_fingerprint,
        data_sha256=data.data_sha256,
        bootstrap_seed=int(seed),
        requested_replicates=int(n_replicates),
        minimum_success_fraction=float(minimum_success_fraction),
        point_estimates=point_estimates,
        parameter_summaries=parameter_summaries,
        predictive_summaries=predictive_summaries,
        nonlinear_selection_stability=_selection_summary(configurations),
        replicates=tuple(replicates),
    )
    if abs(result.success_fraction - success_fraction) > 1e-15:
        raise RuntimeError("bootstrap success accounting is inconsistent")
    return result
