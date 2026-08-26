"""Auditable preregistration evidence for confirmatory QuantumBCI studies.

A boolean saying ``preregistered=True`` is not evidence that a decision rule existed
before results were inspected.  This module provides a small dependency-free contract
that binds a current scientific policy to an externally timestamped registration
reference and immutable content hashes.

QuantumBCI deliberately does not contact OSF or another registry at runtime.  A
``PreregistrationEvidence`` record proves that the run bundle *declares and binds* an
external registration URI, timestamp, registered-document hash, and registered policy
fingerprint.  Authenticity of the external registry record remains independently
verifiable at publication/review time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
from typing import Any, Mapping


def _required_text(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _sha256(name: str, value: Any) -> str:
    text = _required_text(name, value).lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{name} must be a 64-character lowercase/uppercase SHA-256 hex digest")
    return text


def _timestamp(name: str, value: Any) -> str:
    text = _required_text(name, value)
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone offset or Z")
    return text


def canonical_scientific_fingerprint(domain: str, payload: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 fingerprint for a JSON-compatible scientific contract."""

    prefix = _required_text("domain", domain).encode("utf-8") + b"\0"
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(prefix + encoded).hexdigest()


@dataclass(frozen=True)
class PreregistrationEvidence:
    """Reference to an externally timestamped immutable preregistration.

    ``registered_policy_sha256`` must equal the fingerprint of the exact decision
    policy used by the confirmatory analysis. ``registration_document_sha256`` binds
    the complete externally registered document, which may contain hypotheses,
    exclusions, contingencies, power analysis, and other study details beyond the
    machine-readable QuantumBCI policy.
    """

    registration_uri: str
    registered_at: str
    registration_document_sha256: str
    registered_policy_sha256: str
    registry: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "registration_uri", _required_text("registration_uri", self.registration_uri)
        )
        object.__setattr__(self, "registered_at", _timestamp("registered_at", self.registered_at))
        object.__setattr__(
            self,
            "registration_document_sha256",
            _sha256("registration_document_sha256", self.registration_document_sha256),
        )
        object.__setattr__(
            self,
            "registered_policy_sha256",
            _sha256("registered_policy_sha256", self.registered_policy_sha256),
        )
        if self.registry is not None:
            object.__setattr__(self, "registry", _required_text("registry", self.registry))

    def matches_policy(self, policy_sha256: str) -> bool:
        return self.registered_policy_sha256 == _sha256("policy_sha256", policy_sha256)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_role": "external_preregistration_evidence",
            "registration_uri": self.registration_uri,
            "registered_at": self.registered_at,
            "registration_document_sha256": self.registration_document_sha256,
            "registered_policy_sha256": self.registered_policy_sha256,
            "registry": self.registry,
            "verification_scope": (
                "QuantumBCI binds the supplied external registration reference and hashes; "
                "registry authenticity/timestamp must be independently checked against the URI."
            ),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PreregistrationEvidence":
        role = payload.get("artifact_role")
        if role is not None and role != "external_preregistration_evidence":
            raise ValueError(f"unexpected preregistration artifact_role: {role!r}")
        return cls(
            registration_uri=_required_text("registration_uri", payload.get("registration_uri")),
            registered_at=_timestamp("registered_at", payload.get("registered_at")),
            registration_document_sha256=_sha256(
                "registration_document_sha256", payload.get("registration_document_sha256")
            ),
            registered_policy_sha256=_sha256(
                "registered_policy_sha256", payload.get("registered_policy_sha256")
            ),
            registry=(None if payload.get("registry") is None else str(payload.get("registry"))),
        )
