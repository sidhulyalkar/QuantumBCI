"""Read-side verification for BMRB study-level operating artifacts.

Fingerprints are integrity checks rather than signatures. Reusing a study-level operating
artifact therefore requires reconstructing the declared grid and checking the scientific
invariants that can be derived from the serialized evidence. The v1 operating result binds
the seed-partition fingerprint but does not serialize the partition parameters; a future
evaluation seal must supply that full authority separately and prove the fingerprints match.
"""

from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path
from typing import Any, Mapping

from .bmrb_study_operating import (
    BMRB_STUDY_OPERATING_BENCHMARK,
    BMRB_STUDY_OPERATING_METHOD,
    BMRBStudyOperatingGrid,
    default_study_operating_scenarios,
)
from .preregistration import canonical_scientific_fingerprint


def _required_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _required_text(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _sha256(name: str, value: Any) -> str:
    text = _required_text(name, value).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a 64-character SHA-256 hexadecimal digest")
    return text


def _strict_bool(name: str, value: Any) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a JSON/Python boolean")
    return value


def _positive_int(name: str, value: Any) -> int:
    if type(value) is bool:
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if number < 1 or number != value:
        raise ValueError(f"{name} must be a positive integer")
    return number


def _finite(name: str, value: Any) -> float:
    if type(value) is bool:
        raise ValueError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _fraction(name: str, value: Any) -> float:
    number = _finite(name, value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return number


def _close(name: str, observed: Any, expected: float, *, atol: float = 1e-12) -> None:
    value = _finite(name, observed)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=atol):
        raise ValueError(f"{name} is inconsistent with reconstructed study operating evidence")


def _wilson_interval(successes: int, trials: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    proportion = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (proportion + z2 / (2.0 * trials)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z2 / (4.0 * trials * trials)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _unique_tuple(name: str, raw: Any, normalize: Any) -> tuple[Any, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{name} must be a non-empty list")
    values = tuple(normalize(value) for value in raw)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate values")
    return values


def _verify_grid(policy: Mapping[str, Any]) -> BMRBStudyOperatingGrid:
    grid_payload = _required_mapping("policy.grid", policy.get("grid"))
    scenarios = _unique_tuple(
        "grid.scenario_ids",
        grid_payload.get("scenario_ids"),
        lambda value: _required_text("scenario_id", value),
    )
    known = {scenario.scenario_id for scenario in default_study_operating_scenarios()}
    unknown = sorted(set(scenarios) - known)
    if unknown:
        raise ValueError(f"grid contains unknown study operating scenarios: {unknown}")
    participants = _unique_tuple(
        "grid.participant_counts",
        grid_payload.get("participant_counts"),
        lambda value: _positive_int("participant_count", value),
    )
    if any(value < 4 for value in participants):
        raise ValueError("study operating participant counts must be at least four")

    def nonnegative(name: str, value: Any) -> float:
        number = _finite(name, value)
        if number < 0.0:
            raise ValueError(f"{name} must be non-negative")
        return number

    within = _unique_tuple(
        "grid.within_study_heterogeneity_scales",
        grid_payload.get("within_study_heterogeneity_scales"),
        lambda value: nonnegative("within_study_heterogeneity_scale", value),
    )
    noise = _unique_tuple(
        "grid.measurement_noise_scales",
        grid_payload.get("measurement_noise_scales"),
        lambda value: nonnegative("measurement_noise_scale", value),
    )
    cross = _unique_tuple(
        "grid.cross_study_effect_scales",
        grid_payload.get("cross_study_effect_scales"),
        lambda value: nonnegative("cross_study_effect_scale", value),
    )
    grid = BMRBStudyOperatingGrid(
        scenario_ids=scenarios,
        participant_counts=participants,
        within_study_heterogeneity_scales=within,
        measurement_noise_scales=noise,
        cross_study_effect_scales=cross,
    )
    if grid.to_mapping() != dict(grid_payload):
        raise ValueError("study operating grid is noncanonical")
    if _sha256("policy.grid_fingerprint", policy.get("grid_fingerprint")) != grid.fingerprint:
        raise ValueError("study operating grid fingerprint mismatch")
    return grid


def verify_bmrb_study_operating_mapping(payload: Mapping[str, Any]) -> None:
    """Verify one serialized study-level operating artifact before downstream reuse."""

    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("study operating artifact schema_version must be 1")
    if payload.get("benchmark") != BMRB_STUDY_OPERATING_BENCHMARK:
        raise ValueError("artifact is not a BMRB study operating result")
    if payload.get("method") != BMRB_STUDY_OPERATING_METHOD:
        raise ValueError("study operating artifact method mismatch")
    if payload.get("qualification_defined") is not False:
        raise ValueError("development operating evidence must not define qualification post hoc")
    if payload.get("evaluation_partition_executed") is not False:
        raise ValueError("study operating development evidence must not claim evaluation execution")
    if payload.get("physical_quantum_promotion_eligible") is not False:
        raise ValueError("study operating evidence cannot authorize physical-quantum promotion")

    claimed_artifact = _sha256("artifact_fingerprint", payload.get("artifact_fingerprint"))
    core = {key: value for key, value in payload.items() if key != "artifact_fingerprint"}
    expected_artifact = canonical_scientific_fingerprint(
        "quantumbci.bmrb-study-operating-result.v1", core
    )
    if claimed_artifact != expected_artifact:
        raise ValueError("study operating artifact fingerprint mismatch")

    policy = _required_mapping("policy", payload.get("policy"))
    claimed_policy = _sha256("policy.policy_fingerprint", policy.get("policy_fingerprint"))
    policy_core = {key: value for key, value in policy.items() if key != "policy_fingerprint"}
    expected_policy = canonical_scientific_fingerprint(
        "quantumbci.bmrb-study-operating-policy.v1", policy_core
    )
    if claimed_policy != expected_policy:
        raise ValueError("study operating policy fingerprint mismatch")
    if policy.get("benchmark") != BMRB_STUDY_OPERATING_BENCHMARK:
        raise ValueError("study operating policy benchmark mismatch")
    if policy.get("method") != BMRB_STUDY_OPERATING_METHOD:
        raise ValueError("study operating policy method mismatch")
    if policy.get("partition") != "development":
        raise ValueError("reusable study operating development evidence must use development partition")
    if policy.get("evaluation_partition_executable") is not False:
        raise ValueError("study operating v1 must keep evaluation partition non-executable")
    _required_text("policy.study_id", policy.get("study_id"))
    _required_text("policy.source_sha", policy.get("source_sha"))
    replicates = _positive_int("policy.replicates_per_cell", policy.get("replicates_per_cell"))
    _positive_int("policy.bootstrap_resamples", policy.get("bootstrap_resamples"))
    _sha256("policy.seed_partition_fingerprint", policy.get("seed_partition_fingerprint"))
    _fraction(
        "policy.sensitivity_min_direction_agreement",
        policy.get("sensitivity_min_direction_agreement"),
    )
    for name in (
        "sensitivity_max_effect_range",
        "sensitivity_max_leave_one_out_mean_shift",
    ):
        if _finite(f"policy.{name}", policy.get(name)) < 0.0:
            raise ValueError(f"policy.{name} must be non-negative")
    grid = _verify_grid(policy)

    scenario_contracts = {
        scenario.scenario_id: scenario for scenario in default_study_operating_scenarios()
    }
    expected_cells = tuple(
        product(
            grid.scenario_ids,
            grid.participant_counts,
            grid.within_study_heterogeneity_scales,
            grid.measurement_noise_scales,
            grid.cross_study_effect_scales,
        )
    )
    raw_cells = payload.get("cells")
    if not isinstance(raw_cells, list) or len(raw_cells) != len(expected_cells):
        raise ValueError("study operating cells do not cover the complete frozen grid")

    normalized_cells: list[dict[str, Any]] = []
    for index, (raw_cell, expected) in enumerate(zip(raw_cells, expected_cells, strict=True)):
        cell = _required_mapping(f"cells[{index}]", raw_cell)
        scenario_id, participants, within, noise, cross = expected
        scenario = scenario_contracts[scenario_id]
        if cell.get("scenario_id") != scenario_id:
            raise ValueError("study operating cell scenario order disagrees with frozen grid")
        if _positive_int("cell.study_count", cell.get("study_count")) != scenario.study_count:
            raise ValueError("study operating cell study_count disagrees with scenario authority")
        if _positive_int("cell.participant_count", cell.get("participant_count")) != participants:
            raise ValueError("study operating participant count disagrees with frozen grid")
        _close("cell.within_study_heterogeneity_scale", cell.get("within_study_heterogeneity_scale"), within)
        _close("cell.measurement_noise_scale", cell.get("measurement_noise_scale"), noise)
        _close("cell.cross_study_effect_scale", cell.get("cross_study_effect_scale"), cross)
        if _strict_bool("cell.expected_replication_pass", cell.get("expected_replication_pass")) is not scenario.expected_replication_pass:
            raise ValueError("cell expected replication result disagrees with scenario authority")
        if _strict_bool("cell.expected_context_specific_only", cell.get("expected_context_specific_only")) is not scenario.expected_context_specific_only:
            raise ValueError("cell context-specific truth disagrees with scenario authority")
        if _strict_bool("cell.expected_sensitivity_warning", cell.get("expected_sensitivity_warning")) is not scenario.expected_sensitivity_warning:
            raise ValueError("cell sensitivity truth disagrees with scenario authority")
        if _positive_int("cell.replicates", cell.get("replicates")) != replicates:
            raise ValueError("study operating cell replicate count disagrees with policy")

        pass_rate = _fraction("cell.observed_replication_pass_rate", cell.get("observed_replication_pass_rate"))
        successes = int(round(pass_rate * replicates))
        if not 0 <= successes <= replicates or not math.isclose(
            successes / replicates, pass_rate, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("observed replication pass rate is not attainable at declared replicate count")
        expected_error = 1.0 - pass_rate if scenario.expected_replication_pass else pass_rate
        _close("cell.decision_error_rate", cell.get("decision_error_rate"), expected_error)
        lower, upper = _wilson_interval(successes, replicates)
        _close("cell.pass_rate_ci_lower", cell.get("pass_rate_ci_lower"), lower)
        _close("cell.pass_rate_ci_upper", cell.get("pass_rate_ci_upper"), upper)
        for name in (
            "context_specific_match_rate",
            "sensitivity_warning_match_rate",
            "primary_role_protection_rate",
            "fragile_claim_detection_rate",
        ):
            _fraction(f"cell.{name}", cell.get(name))
        _finite("cell.mean_successful_replication_margin", cell.get("mean_successful_replication_margin"))
        if _finite("cell.mean_study_effect_range", cell.get("mean_study_effect_range")) < 0.0:
            raise ValueError("cell.mean_study_effect_range must be non-negative")
        normalized_cells.append(dict(cell))

    aggregate = _required_mapping("aggregate", payload.get("aggregate"))
    negative = [cell for cell in normalized_cells if not cell["expected_replication_pass"]]
    positive = [cell for cell in normalized_cells if cell["expected_replication_pass"]]
    warning = [cell for cell in normalized_cells if cell["expected_sensitivity_warning"]]
    no_warning = [cell for cell in normalized_cells if not cell["expected_sensitivity_warning"]]
    expected_aggregate = {
        "mean_false_promotion_rate": sum(cell["observed_replication_pass_rate"] for cell in negative) / len(negative),
        "mean_known_positive_recovery_rate": sum(cell["observed_replication_pass_rate"] for cell in positive) / len(positive),
        "mean_context_semantics_match_rate": sum(cell["context_specific_match_rate"] for cell in normalized_cells) / len(normalized_cells),
        "mean_expected_warning_match_rate": sum(cell["sensitivity_warning_match_rate"] for cell in warning) / len(warning),
        "mean_expected_no_warning_match_rate": sum(cell["sensitivity_warning_match_rate"] for cell in no_warning) / len(no_warning),
    }
    for name, expected in expected_aggregate.items():
        _close(f"aggregate.{name}", aggregate.get(name), expected)
    if aggregate.get("qualification_defined") is not False:
        raise ValueError("study operating aggregate must keep qualification undefined")
    if set(aggregate) != {*expected_aggregate, "qualification_defined"}:
        raise ValueError("study operating aggregate contains unexpected or missing fields")
    _required_text("interpretation", payload.get("interpretation"))


def load_bmrb_study_operating(path: str | Path) -> dict[str, Any]:
    """Load and verify one serialized study-level operating artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("study operating artifact must contain a JSON object")
    verify_bmrb_study_operating_mapping(payload)
    return payload
