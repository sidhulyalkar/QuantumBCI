"""Read-side verification for BMRB operating-characteristics artifacts.

Fingerprints in QuantumBCI are integrity checks, not signatures. This verifier therefore
checks both serialized fingerprints and the internal scientific invariants needed before
an operating-study artifact is reused downstream.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .bmrb_validation import default_validation_scenarios
from .bmrb_validation_operating import (
    BMRB_OPERATING_CHARACTERISTICS_BENCHMARK,
    BMRB_OPERATING_CHARACTERISTICS_METHOD,
    NORMAL_95,
    _scientific_fingerprint,
)


def _required_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _required_text(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _finite(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive_int(name: str, value: Any) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if normalized <= 0 or normalized != value:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def _fraction(name: str, value: Any) -> float:
    number = _finite(name, value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return number


def _close(name: str, observed: Any, expected: float, *, atol: float = 1e-12) -> None:
    value = _finite(name, observed)
    if not np.isclose(value, expected, rtol=0.0, atol=atol):
        raise ValueError(f"{name} is inconsistent with reconstructed operating evidence")


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    proportion = successes / trials
    z2 = NORMAL_95**2
    denominator = 1.0 + z2 / trials
    center = (proportion + z2 / (2.0 * trials)) / denominator
    half_width = (
        NORMAL_95
        * np.sqrt(
            proportion * (1.0 - proportion) / trials
            + z2 / (4.0 * trials * trials)
        )
        / denominator
    )
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def _verify_grid(grid: Mapping[str, Any], claimed_fingerprint: Any) -> tuple[
    tuple[str, ...],
    tuple[int, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    scenarios_raw = grid.get("scenario_ids")
    participants_raw = grid.get("participant_counts")
    effects_raw = grid.get("effect_scales")
    heterogeneity_raw = grid.get("heterogeneity_scales")
    noise_raw = grid.get("measurement_noise_scales")
    for name, values in (
        ("grid.scenario_ids", scenarios_raw),
        ("grid.participant_counts", participants_raw),
        ("grid.effect_scales", effects_raw),
        ("grid.heterogeneity_scales", heterogeneity_raw),
        ("grid.measurement_noise_scales", noise_raw),
    ):
        if not isinstance(values, list) or not values:
            raise ValueError(f"{name} must be a non-empty list")

    scenarios = tuple(_required_text("grid.scenario_id", value) for value in scenarios_raw)
    if len(set(scenarios)) != len(scenarios):
        raise ValueError("grid.scenario_ids must be unique")
    known = {scenario.scenario_id for scenario in default_validation_scenarios()}
    unknown = sorted(set(scenarios) - known)
    if unknown:
        raise ValueError(f"grid contains unknown scenario ids: {unknown}")

    participants = tuple(_positive_int("grid.participant_count", value) for value in participants_raw)
    if len(set(participants)) != len(participants) or any(value < 4 for value in participants):
        raise ValueError("grid participant counts must be unique and at least four")

    def positive_unique(name: str, raw: list[Any]) -> tuple[float, ...]:
        values = tuple(_finite(name, value) for value in raw)
        if any(value <= 0.0 for value in values) or len(set(values)) != len(values):
            raise ValueError(f"{name} values must be positive and unique")
        return values

    effects = positive_unique("grid.effect_scale", effects_raw)
    heterogeneity = positive_unique("grid.heterogeneity_scale", heterogeneity_raw)
    noise = positive_unique("grid.measurement_noise_scale", noise_raw)
    expected_cell_count = (
        len(scenarios)
        * len(participants)
        * len(effects)
        * len(heterogeneity)
        * len(noise)
    )
    if _positive_int("grid.cell_count", grid.get("cell_count")) != expected_cell_count:
        raise ValueError("grid.cell_count does not match the declared Cartesian grid")
    expected_fingerprint = _scientific_fingerprint(
        "quantumbci.bmrb-operating-grid.v1",
        dict(grid),
    )
    if _required_text("grid_fingerprint", claimed_fingerprint) != expected_fingerprint:
        raise ValueError("operating grid fingerprint mismatch")
    return scenarios, participants, effects, heterogeneity, noise


def _verify_seed_partition(
    policy: Mapping[str, Any],
    *,
    cell_count: int,
    replicates_per_cell: int,
) -> tuple[int, int]:
    seed_partition = _required_mapping("policy.seed_partition", policy.get("seed_partition"))
    if seed_partition.get("method") != "disjoint_arithmetic_seed_partitions_v1":
        raise ValueError("unknown operating seed-partition method")
    expected_fingerprint = _scientific_fingerprint(
        "quantumbci.bmrb-operating-seed-partition.v1",
        dict(seed_partition),
    )
    if _required_text(
        "policy.seed_partition_fingerprint",
        policy.get("seed_partition_fingerprint"),
    ) != expected_fingerprint:
        raise ValueError("operating seed-partition fingerprint mismatch")

    development_offset = _positive_int(
        "seed_partition.development_offset", seed_partition.get("development_offset")
    )
    evaluation_offset = _positive_int(
        "seed_partition.evaluation_offset", seed_partition.get("evaluation_offset")
    )
    cell_stride = _positive_int("seed_partition.cell_stride", seed_partition.get("cell_stride"))
    replicate_stride = _positive_int(
        "seed_partition.replicate_stride", seed_partition.get("replicate_stride")
    )
    max_replicates = _positive_int(
        "seed_partition.max_replicates_per_cell",
        seed_partition.get("max_replicates_per_cell"),
    )
    if development_offset >= evaluation_offset:
        raise ValueError("development seed authority must precede evaluation authority")
    if replicates_per_cell > max_replicates:
        raise ValueError("operating artifact exceeds seed-partition replicate capacity")
    if cell_stride <= (max_replicates - 1) * replicate_stride:
        raise ValueError("operating seed cell_stride permits within-partition collisions")

    development_maximum = (
        development_offset
        + (cell_count - 1) * cell_stride
        + (replicates_per_cell - 1) * replicate_stride
    )
    if development_maximum >= evaluation_offset:
        raise ValueError("development and evaluation seed authorities overlap")

    partition = _required_text("policy.partition", policy.get("partition"))
    if partition == "development":
        selected_offset = development_offset
    elif partition == "evaluation":
        selected_offset = evaluation_offset
    else:
        raise ValueError("policy.partition must be development or evaluation")
    return selected_offset, cell_stride


def verify_bmrb_operating_characteristics_mapping(payload: Mapping[str, Any]) -> None:
    """Verify fingerprints and scientific invariants before reusing an operating artifact."""

    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("operating artifact schema_version must be 1")
    if payload.get("benchmark") != BMRB_OPERATING_CHARACTERISTICS_BENCHMARK:
        raise ValueError("artifact is not a BMRB operating-characteristics result")
    if payload.get("method") != BMRB_OPERATING_CHARACTERISTICS_METHOD:
        raise ValueError("operating artifact method mismatch")
    if payload.get("qualification_defined") is not False:
        raise ValueError("operating artifact must not invent a universal qualification gate")

    claimed_artifact = _required_text("artifact_fingerprint", payload.get("artifact_fingerprint"))
    core = {key: value for key, value in payload.items() if key != "artifact_fingerprint"}
    expected_artifact = _scientific_fingerprint(
        "quantumbci.bmrb-operating-result.v1",
        core,
    )
    if claimed_artifact != expected_artifact:
        raise ValueError("operating artifact fingerprint mismatch")

    policy = _required_mapping("policy", payload.get("policy"))
    claimed_policy = _required_text("policy.policy_fingerprint", policy.get("policy_fingerprint"))
    policy_core = {key: value for key, value in policy.items() if key != "policy_fingerprint"}
    expected_policy = _scientific_fingerprint(
        "quantumbci.bmrb-operating-policy.v1",
        policy_core,
    )
    if claimed_policy != expected_policy:
        raise ValueError("operating policy fingerprint mismatch")

    if policy.get("benchmark") != BMRB_OPERATING_CHARACTERISTICS_BENCHMARK:
        raise ValueError("operating policy benchmark mismatch")
    if policy.get("method") != BMRB_OPERATING_CHARACTERISTICS_METHOD:
        raise ValueError("operating policy method mismatch")
    _required_text("policy.study_id", policy.get("study_id"))
    _required_text("policy.source_sha", policy.get("source_sha"))
    replicates = _positive_int("policy.replicates_per_cell", policy.get("replicates_per_cell"))
    if _positive_int("policy.bootstrap_resamples", policy.get("bootstrap_resamples")) < 100:
        raise ValueError("policy.bootstrap_resamples must be at least 100")
    primary_budget = int(policy.get("primary_calibration_per_class", -1))
    if primary_budget < 0:
        raise ValueError("policy.primary_calibration_per_class must be non-negative")

    grid = _required_mapping("policy.grid", policy.get("grid"))
    scenarios, participants, effects, heterogeneity, noise = _verify_grid(
        grid,
        policy.get("grid_fingerprint"),
    )
    cell_count = _positive_int("grid.cell_count", grid.get("cell_count"))
    selected_offset, cell_stride = _verify_seed_partition(
        policy,
        cell_count=cell_count,
        replicates_per_cell=replicates,
    )

    raw_cells = payload.get("cells")
    if not isinstance(raw_cells, list) or len(raw_cells) != cell_count:
        raise ValueError("operating artifact cells do not cover the complete frozen grid")
    scenario_contracts = {scenario.scenario_id: scenario for scenario in default_validation_scenarios()}
    combinations = tuple(product(scenarios, participants, effects, heterogeneity, noise))

    for index, (raw_cell, expected_combo) in enumerate(zip(raw_cells, combinations, strict=True)):
        cell = _required_mapping(f"cells[{index}]", raw_cell)
        if int(cell.get("cell_index", -1)) != index:
            raise ValueError("operating cell indices are incomplete or out of order")
        scenario_id, participant_count, effect_scale, heterogeneity_scale, noise_scale = expected_combo
        scenario = scenario_contracts[scenario_id]
        if cell.get("scenario_id") != scenario_id:
            raise ValueError("operating cell scenario order does not match the frozen grid")
        if int(cell.get("participant_count", -1)) != participant_count:
            raise ValueError("operating cell participant count does not match the frozen grid")
        _close("cell.effect_scale", cell.get("effect_scale"), effect_scale)
        _close("cell.heterogeneity_scale", cell.get("heterogeneity_scale"), heterogeneity_scale)
        _close("cell.measurement_noise_scale", cell.get("measurement_noise_scale"), noise_scale)
        if cell.get("truth_class") != scenario.truth_class:
            raise ValueError("operating cell truth_class disagrees with the declared DGM")
        if cell.get("expected_scientific_pass") is not scenario.expected_scientific_pass:
            raise ValueError("operating cell expected scientific result disagrees with the DGM")
        if cell.get("expected_failure_component") != scenario.expected_failure_component:
            raise ValueError("operating cell expected failure component disagrees with the DGM")

        expected_seed = selected_offset + index * cell_stride
        if int(cell.get("base_seed", -1)) != expected_seed:
            raise ValueError("operating cell base seed does not match frozen seed authority")
        if _positive_int("cell.replicates", cell.get("replicates")) != replicates:
            raise ValueError("operating cell replicate count does not match the policy")
        successes = int(cell.get("observed_passes", -1))
        if not 0 <= successes <= replicates:
            raise ValueError("operating cell observed_passes is outside the replicate range")
        pass_rate = successes / replicates
        _close("cell.observed_pass_rate", cell.get("observed_pass_rate"), pass_rate)
        expected_error = 1.0 - pass_rate if scenario.expected_scientific_pass else pass_rate
        _close("cell.decision_error_rate", cell.get("decision_error_rate"), expected_error)
        expected_se = float(np.sqrt(pass_rate * (1.0 - pass_rate) / replicates))
        _close("cell.monte_carlo_se", cell.get("monte_carlo_se"), expected_se)
        ci_lower, ci_upper = _wilson_interval(successes, replicates)
        _close("cell.pass_rate_ci_lower", cell.get("pass_rate_ci_lower"), ci_lower)
        _close("cell.pass_rate_ci_upper", cell.get("pass_rate_ci_upper"), ci_upper)
        _fraction(
            "cell.expected_failure_localization_rate",
            cell.get("expected_failure_localization_rate"),
        )
        _fraction("cell.reference_ci_coverage", cell.get("reference_ci_coverage"))
        _finite("cell.mean_reference_effect_bias", cell.get("mean_reference_effect_bias"))
        rmse = _finite("cell.reference_effect_rmse", cell.get("reference_effect_rmse"))
        if rmse < 0.0:
            raise ValueError("cell.reference_effect_rmse must be non-negative")

    _required_mapping("aggregate", payload.get("aggregate"))
    _required_text("interpretation", payload.get("interpretation"))


def load_bmrb_operating_characteristics(path: str | Path) -> dict[str, Any]:
    """Load and verify one serialized operating-characteristics artifact."""

    artifact_path = Path(path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("operating-characteristics artifact must contain a JSON object")
    verify_bmrb_operating_characteristics_mapping(payload)
    return payload