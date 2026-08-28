from __future__ import annotations

import pytest

from quantumbci.preregistration import (
    PreregistrationEvidence,
    canonical_scientific_fingerprint,
)


def test_preregistration_evidence_requires_timestamped_hash_bound_record() -> None:
    evidence = PreregistrationEvidence(
        registration_uri="https://osf.io/example/register/123",
        registered_at="2026-08-01T12:30:00Z",
        registration_document_sha256="a" * 64,
        registered_policy_sha256="b" * 64,
        registry="OSF",
    )
    payload = evidence.to_mapping()
    assert payload["artifact_role"] == "external_preregistration_evidence"
    assert evidence.matches_policy("b" * 64) is True
    assert evidence.matches_policy("c" * 64) is False


def test_preregistration_rejects_naive_timestamp_and_malformed_hashes() -> None:
    with pytest.raises(ValueError, match="timezone"):
        PreregistrationEvidence(
            registration_uri="https://example.org/registration",
            registered_at="2026-08-01T12:30:00",
            registration_document_sha256="a" * 64,
            registered_policy_sha256="b" * 64,
        )
    with pytest.raises(ValueError, match="SHA-256"):
        PreregistrationEvidence(
            registration_uri="https://example.org/registration",
            registered_at="2026-08-01T12:30:00Z",
            registration_document_sha256="not-a-hash",
            registered_policy_sha256="b" * 64,
        )


def test_scientific_fingerprint_is_order_invariant_but_content_sensitive() -> None:
    first = canonical_scientific_fingerprint("policy", {"a": 1, "b": [2, 3]})
    reordered = canonical_scientific_fingerprint("policy", {"b": [2, 3], "a": 1})
    changed = canonical_scientific_fingerprint("policy", {"a": 1, "b": [2, 4]})
    assert first == reordered
    assert first != changed
