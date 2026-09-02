from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from quantumbci.kumar2024_authority_artifacts import (
    fingerprint_kumar2024_authority_capsule_mapping,
    verify_kumar2024_authority_capsule_mapping,
)

ROOT = Path("evidence/kumar2024-authority-freeze-v1")
EXPECTED_CAPSULE_FINGERPRINT = (
    "1013358b419436a3a9592c8a48eec2372701b1977e7ced06f4c25cfd4ebae29d"
)
EXPECTED_COHORT_FINGERPRINT = (
    "36cdfdf42e5ac375999d4defa02554cf4d2d04472ed6c06a08c389b5ad02b81c"
)
EXPECTED_RAW_FINGERPRINT = (
    "c91c6dca34be880e688359e210686c1823461ad93923f71e947bb3d0725d6c8b"
)


def _load() -> dict:
    return json.loads(
        (ROOT / "authority-capsule.json").read_text(encoding="utf-8")
    )


def _canonical(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _pretty_hash(value: object) -> str:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return sha256(raw).hexdigest()


def _canonical_hash(value: object) -> str:
    return sha256(_canonical(value).encode("utf-8")).hexdigest()


def _semantic_files(payload: dict) -> dict[str, object]:
    files = {
        "authority-freeze.json": payload["authority_freeze"],
        "raw-source-fingerprint.json": payload["raw_source_fingerprint"],
    }
    for authority in payload["subject_authorities"]:
        subject = authority["case_metadata"]["subject"]
        files[f"authority-subject-{subject}-session-5.json"] = authority
    return files


def _refresh_integrity_layers(payload: dict) -> None:
    files = _semantic_files(payload)
    payload["canonical_component_sha256"] = {
        name: _canonical_hash(value)
        for name, value in sorted(files.items())
    }
    payload["source_artifact"]["original_file_sha256_manifest"] = {
        name: _pretty_hash(value) for name, value in sorted(files.items())
    }
    payload["capsule_fingerprint"] = (
        fingerprint_kumar2024_authority_capsule_mapping(payload)
    )


def test_committed_capsule_verifies_and_pins_exact_authority() -> None:
    payload = _load()
    verified = verify_kumar2024_authority_capsule_mapping(payload)
    assert verified["capsule_fingerprint"] == EXPECTED_CAPSULE_FINGERPRINT
    assert (
        verified["authority_freeze"]["cohort_authority_fingerprint"]
        == EXPECTED_COHORT_FINGERPRINT
    )
    assert (
        verified["raw_source_fingerprint"]["fingerprint"]
        == EXPECTED_RAW_FINGERPRINT
    )
    assert verified["authority_freeze"]["subject_count"] == 18
    assert len(verified["raw_source_fingerprint"]["files"]) == 360
    assert verified["authority_freeze"]["raw_total_bytes"] == 4_193_818_400


def test_every_subject_partition_is_complete_disjoint_and_real_geometry() -> None:
    payload = verify_kumar2024_authority_capsule_mapping(_load())
    calibration_sizes = []
    evaluation_sizes = []
    for authority in payload["subject_authorities"]:
        n_samples = authority["n_samples"]
        source = set(authority["source_train_indices"])
        evaluation = set(authority["evaluation_indices"])
        calibration = {
            value
            for values in authority["calibration_order_by_class"].values()
            for value in values
        }
        assert not source & calibration
        assert not source & evaluation
        assert not calibration & evaluation
        assert source | calibration | evaluation == set(range(n_samples))
        assert authority["input_shape"] == [n_samples, 22, 2561]
        calibration_sizes.append(len(calibration))
        evaluation_sizes.append(len(evaluation))
    assert min(calibration_sizes) == 28
    assert max(calibration_sizes) == 30
    assert min(evaluation_sizes) == 29
    assert max(evaluation_sizes) == 30


def test_partition_overlap_is_rejected_even_after_outer_integrity_is_refreshed() -> None:
    payload = deepcopy(_load())
    authority = payload["subject_authorities"][0]
    authority["evaluation_indices"][0] = authority["source_train_indices"][0]
    _refresh_integrity_layers(payload)
    with pytest.raises(ValueError, match="partitions overlap"):
        verify_kumar2024_authority_capsule_mapping(payload)


def test_outcome_field_injection_is_rejected_even_after_refingerprinting() -> None:
    payload = deepcopy(_load())
    payload["subject_authorities"][0]["candidate_metric"] = 0.9
    _refresh_integrity_layers(payload)
    with pytest.raises(ValueError, match="outcome-shaped fields leaked"):
        verify_kumar2024_authority_capsule_mapping(payload)


def test_source_manifest_must_be_reproducible_from_embedded_mappings() -> None:
    payload = deepcopy(_load())
    payload["source_artifact"]["original_file_sha256_manifest"][
        "authority-subject-1-session-5.json"
    ] = "0" * 64
    payload["capsule_fingerprint"] = (
        fingerprint_kumar2024_authority_capsule_mapping(payload)
    )
    with pytest.raises(ValueError, match="do not reproduce"):
        verify_kumar2024_authority_capsule_mapping(payload)


def test_claim_boundary_cannot_be_promoted_by_refingerprinting() -> None:
    payload = deepcopy(_load())
    payload["claim_boundary"]["physical_quantum_promotion_eligible"] = True
    payload["capsule_fingerprint"] = (
        fingerprint_kumar2024_authority_capsule_mapping(payload)
    )
    with pytest.raises(ValueError, match="claim boundary drifted"):
        verify_kumar2024_authority_capsule_mapping(payload)


def test_moabb_to_raw_subject_mapping_is_explicit_and_nontrivial() -> None:
    payload = verify_kumar2024_authority_capsule_mapping(_load())
    mapping = payload["raw_source_fingerprint"]["selection"][
        "moabb_subject_to_raw_subject"
    ]
    assert mapping["1"] == 1
    assert mapping["9"] == 9
    assert mapping["10"] == 11
    assert mapping["18"] == 19


def test_repository_file_manifest_pins_exact_capsule_bytes() -> None:
    manifest = json.loads(
        (ROOT / "sha256-manifest.json").read_text(encoding="utf-8")
    )
    capsule = ROOT / "authority-capsule.json"
    assert manifest == {
        "authority-capsule.json": sha256(capsule.read_bytes()).hexdigest()
    }
