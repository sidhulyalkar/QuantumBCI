"""Lightweight verification for JSON-only neuros-mechint scientific artifacts.

QuantumBCI's base install intentionally does not depend on PyTorch or neuros-mechint.
This module reproduces neuros-mechint's stable hashing semantics for JSON-compatible
scientific payloads so BMRB can reject hand-edited causal evidence before promotion.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
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


def verify_evidence_pack_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(unwrap_artifact_result(payload))
    schema = result.get("schema_version")
    if schema != EVIDENCE_PACK_SCHEMA:
        raise ValueError(f"unsupported neuros-mechint evidence-pack schema: {schema!r}")
    fingerprint = result.get("study_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("evidence-pack result is missing study_fingerprint")
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
    scientific_identity = {key: result[key] for key in required}
    expected = neuros_mechint_stable_hash(scientific_identity)
    if expected != fingerprint:
        raise ValueError("evidence-pack scientific fingerprint mismatch")
    return result
