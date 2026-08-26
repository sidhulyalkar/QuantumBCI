"""Lightweight verification for JSON-only neuros-mechint scientific artifacts.

QuantumBCI's base install intentionally does not depend on PyTorch or neuros-mechint.
This module reproduces neuros-mechint's stable hashing semantics for JSON-compatible
scientific payloads so BMRB can reject hand-edited causal evidence before promotion.

For evidence packs, the study fingerprint intentionally binds the frozen candidate,
policies and per-example candidate cases. Convenience aggregates and promotion
summaries are derived fields. BMRB therefore recomputes the validation summary from
the fingerprint-bound cases instead of trusting those convenience fields directly.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from statistics import median
from typing import Any


DOSE_RESPONSE_SCHEMA = "neuros-mechint.dose-response-study.v1"
EVIDENCE_PACK_SCHEMA = "neuros-mechint.evidence-pack.v1"


def _stable_hash_update(hasher: Any, value: Any) -> None:
    """Mirror neuros_mechint.core.manifest._update_hash for JSON-compatible values."""

    if value is None:
        hasher.update(b"none")
    elif isinstance(value, bool):
        hasher.update(b"bool:1" if value else b"bool:0")
    elif isinstance(value, int):
        hasher.update(f"int:{value}".encode())
    elif isinstance(value, float):
        hasher.update(f"float:{value.hex()}".encode())
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        hasher.update(f"str:{len(encoded)}:".encode())
        hasher.update(encoded)
    elif isinstance(value, Mapping):
        hasher.update(b"mapping:")
        for key in sorted(value, key=lambda item: str(item)):
            _stable_hash_update(hasher, key)
            _stable_hash_update(hasher, value[key])
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        hasher.update(b"sequence:")
        hasher.update(str(len(value)).encode())
        for item in value:
            _stable_hash_update(hasher, item)
    else:
        raise TypeError(
            "QuantumBCI lightweight neuros-mechint verification only supports "
            f"JSON-compatible scientific payloads; observed {type(value)!r}"
        )


def neuros_mechint_stable_hash(value: Any) -> str:
    hasher = hashlib.sha256()
    _stable_hash_update(hasher, value)
    return hasher.hexdigest()


def unwrap_artifact_result(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a scientific result from either an artifact envelope or raw result."""

    result = payload.get("result")
    if isinstance(result, Mapping):
        return result
    return payload


def verify_dose_response_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(unwrap_artifact_result(payload))
    schema = result.get("schema_version")
    if schema != DOSE_RESPONSE_SCHEMA:
        raise ValueError(f"unsupported neuros-mechint dose-response schema: {schema!r}")
    fingerprint = result.get("study_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("dose-response result is missing study_fingerprint")
    scientific_payload = {
        key: value
        for key, value in result.items()
        if key not in {"schema_version", "study_fingerprint"}
    }
    expected = neuros_mechint_stable_hash(scientific_payload)
    if expected != fingerprint:
        raise ValueError("dose-response scientific fingerprint mismatch")
    return result


def _required_pack_identity(result: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "candidate",
        "candidate_cases",
        "discovery_example_ids",
        "faithfulness_policy",
        "magnitude_candidate",
        "magnitude_cases",
        "mean_ablation_references",
        "policy",
        "spec",
        "validation_example_ids",
    )
    missing = [key for key in required if key not in result]
    if missing:
        raise ValueError(f"evidence-pack result is missing fingerprint field(s): {missing}")
    return {key: result[key] for key in required}


def _finite_number(label: str, value: Any) -> float:
    number = float(value)
    if number != number or number in {float("inf"), float("-inf")}:
        raise ValueError(f"{label} must be finite")
    return number


def _valid_reports(
    cases: Any,
    *,
    split: str,
) -> tuple[list[tuple[Mapping[str, Any], Mapping[str, Any]]], int, set[str], set[str]]:
    if not isinstance(cases, list):
        raise ValueError("evidence-pack cases must be a list")
    reports: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    total = 0
    example_ids: set[str] = set()
    baselines: set[str] = set()
    for case in cases:
        if not isinstance(case, Mapping) or str(case.get("split", "")) != split:
            continue
        total += 1
        example_id = str(case.get("example_id", "")).strip()
        baseline = str(case.get("intervention_baseline", "")).strip()
        if example_id:
            example_ids.add(example_id)
        if baseline:
            baselines.add(baseline)
        if not bool(case.get("valid", False)):
            continue
        report = case.get("report")
        if not isinstance(report, Mapping):
            raise ValueError("valid evidence-pack case is missing report")
        reports.append((case, report))
    return reports, total, example_ids, baselines


def _joint_faithfulness(report: Mapping[str, Any]) -> float:
    if report.get("joint_faithfulness") is not None:
        return _finite_number("joint_faithfulness", report.get("joint_faithfulness"))
    return min(
        _finite_number("sufficiency_fraction", report.get("sufficiency_fraction")),
        _finite_number("necessity_fraction", report.get("necessity_fraction")),
    )


def derive_evidence_pack_validation(result: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute BMRB-consumed evidence from fingerprint-bound evidence-pack cases."""

    candidate_cases = result.get("candidate_cases")
    validation, validation_total, validation_examples, validation_baselines = _valid_reports(
        candidate_cases, split="validation"
    )
    discovery, _, _, _ = _valid_reports(candidate_cases, split="discovery")
    magnitude_validation, magnitude_total, _, _ = _valid_reports(
        result.get("magnitude_cases"), split="validation"
    )
    if not validation:
        raise ValueError("evidence pack has no valid fingerprint-bound validation cases")

    sufficiency = [
        _finite_number("sufficiency_fraction", report.get("sufficiency_fraction"))
        for _, report in validation
    ]
    necessity = [
        _finite_number("necessity_fraction", report.get("necessity_fraction"))
        for _, report in validation
    ]
    joint = [_joint_faithfulness(report) for _, report in validation]
    random_percentiles: list[float] = []
    passed: list[bool] = []
    for _, report in validation:
        raw_random = report.get("joint_random_percentile")
        if raw_random is None:
            raise ValueError("validation faithfulness report lacks joint_random_percentile")
        random_percentiles.append(_finite_number("joint_random_percentile", raw_random))
        passed.append(bool(report.get("passed", False)))

    validation_pass_rate = sum(passed) / len(passed)
    valid_case_rate = len(validation) / validation_total if validation_total else 0.0
    validation_joint_median = float(median(joint))
    discovery_joint = [_joint_faithfulness(report) for _, report in discovery]
    discovery_joint_median = (
        float(median(discovery_joint)) if discovery_joint else validation_joint_median
    )
    generalization_drop = max(0.0, discovery_joint_median - validation_joint_median)

    magnitude_joint = [_joint_faithfulness(report) for _, report in magnitude_validation]
    magnitude_validation_median = (
        float(median(magnitude_joint)) if magnitude_joint else None
    )
    advantage_vs_magnitude = (
        None
        if magnitude_validation_median is None
        else validation_joint_median - magnitude_validation_median
    )

    policy = result.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("evidence pack is missing fingerprint-bound policy")
    min_examples = int(policy.get("min_validation_examples", 2))
    min_pass_rate = _finite_number(
        "policy.min_validation_pass_rate", policy.get("min_validation_pass_rate", 0.80)
    )
    min_joint = _finite_number(
        "policy.min_validation_joint_median", policy.get("min_validation_joint_median", 0.50)
    )
    max_drop = _finite_number(
        "policy.max_joint_generalization_drop", policy.get("max_joint_generalization_drop", 0.25)
    )
    min_advantage = _finite_number(
        "policy.min_validation_joint_advantage_vs_magnitude",
        policy.get("min_validation_joint_advantage_vs_magnitude", 0.0),
    )
    require_all_valid = bool(policy.get("require_all_cases_valid", True))
    require_multiple_baselines = bool(
        policy.get("require_multiple_intervention_baselines", True)
    )

    reasons: list[str] = []
    if len(validation_examples) < min_examples:
        reasons.append(f"validation examples {len(validation_examples)} < {min_examples}")
    if require_all_valid and len(validation) != validation_total:
        reasons.append("one or more validation cases are invalid")
    if validation_pass_rate < min_pass_rate:
        reasons.append(
            f"validation pass rate {validation_pass_rate:.3f} < {min_pass_rate:.3f}"
        )
    if validation_joint_median < min_joint:
        reasons.append(
            f"validation joint median {validation_joint_median:.3f} < {min_joint:.3f}"
        )
    if generalization_drop > max_drop:
        reasons.append(
            f"joint generalization drop {generalization_drop:.3f} > {max_drop:.3f}"
        )
    if require_multiple_baselines and len(validation_baselines) < 2:
        reasons.append("validation evidence does not span multiple intervention baselines")
    if result.get("magnitude_candidate") is not None:
        if magnitude_total and not magnitude_validation:
            reasons.append("magnitude-control validation cases are all invalid")
        if advantage_vs_magnitude is None:
            reasons.append("magnitude-control validation evidence is unavailable")
        elif advantage_vs_magnitude < min_advantage:
            reasons.append(
                "validation joint advantage versus magnitude control "
                f"{advantage_vs_magnitude:.3f} < {min_advantage:.3f}"
            )

    derived = {
        "n_cases": validation_total,
        "n_valid_cases": len(validation),
        "n_invalid_cases": validation_total - len(validation),
        "n_examples": len(validation_examples),
        "pass_rate": float(validation_pass_rate),
        "valid_case_rate": float(valid_case_rate),
        "mean_sufficiency": float(sum(sufficiency) / len(sufficiency)),
        "mean_necessity": float(sum(necessity) / len(necessity)),
        "mean_joint_faithfulness": float(sum(joint) / len(joint)),
        "median_joint_faithfulness": validation_joint_median,
        "mean_joint_random_percentile": float(
            sum(random_percentiles) / len(random_percentiles)
        ),
        "promotion_passed": not reasons,
        "promotion_reasons": tuple(reasons),
        "discovery_joint_median": discovery_joint_median,
        "joint_generalization_drop": float(generalization_drop),
        "validation_joint_advantage_vs_magnitude": advantage_vs_magnitude,
    }

    claimed_aggregate = result.get("validation_aggregate")
    if isinstance(claimed_aggregate, Mapping):
        for key in (
            "n_cases",
            "n_valid_cases",
            "n_invalid_cases",
            "n_examples",
            "pass_rate",
            "valid_case_rate",
            "mean_sufficiency",
            "mean_necessity",
            "mean_joint_faithfulness",
            "median_joint_faithfulness",
            "mean_joint_random_percentile",
        ):
            if key not in claimed_aggregate:
                raise ValueError(f"validation_aggregate is missing {key!r}")
            claimed = claimed_aggregate[key]
            expected = derived[key]
            if isinstance(expected, int):
                if int(claimed) != expected:
                    raise ValueError(f"validation_aggregate mismatch: {key}")
            elif abs(float(claimed) - float(expected)) > 1e-9:
                raise ValueError(f"validation_aggregate mismatch: {key}")

    claimed_promotion = result.get("promotion")
    if isinstance(claimed_promotion, Mapping):
        if bool(claimed_promotion.get("passed", False)) != derived["promotion_passed"]:
            raise ValueError("evidence-pack promotion summary mismatch")

    return derived


def verify_evidence_pack_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(unwrap_artifact_result(payload))
    schema = result.get("schema_version")
    if schema != EVIDENCE_PACK_SCHEMA:
        raise ValueError(f"unsupported neuros-mechint evidence-pack schema: {schema!r}")
    fingerprint = result.get("study_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("evidence-pack result is missing study_fingerprint")
    scientific_identity = _required_pack_identity(result)
    expected = neuros_mechint_stable_hash(scientific_identity)
    if expected != fingerprint:
        raise ValueError("evidence-pack scientific fingerprint mismatch")
    derive_evidence_pack_validation(result)
    return result
