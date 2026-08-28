"""Matched-classical recovery evidence for BMRB causal-necessity studies.

A causal ablation is not sufficient evidence that a mechanism is necessary. BMRB
also asks whether the strongest declared classical alternative can recover the
ablation loss under the same information/evidence budget.

This module makes that recovery calculation a derived, fingerprinted artifact rather
than an inline number. The producer records the baseline, ablated and recovered
metrics plus evidence identities; QuantumBCI derives the recovery fraction and
verifies it again whenever the artifact is consumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping

import numpy as np

from .causal_recapitulation import MatchedClassicalRecovery

MATCHED_RECOVERY_SCHEMA = "quantumbci.matched-classical-recovery.v1"
MATCHED_RECOVERY_METHOD_ID = "matched_information_set_ablation_recovery_v1"


def _required_text(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _finite(name: str, value: Any) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fingerprint(value: Any) -> str:
    return sha256(
        b"quantumbci.matched-classical-recovery.v1\0"
        + _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _oriented_loss(
    *, baseline_metric: float, ablated_metric: float, higher_is_better: bool
) -> float:
    return (
        baseline_metric - ablated_metric
        if higher_is_better
        else ablated_metric - baseline_metric
    )


def _oriented_restoration(
    *, ablated_metric: float, recovered_metric: float, higher_is_better: bool
) -> float:
    return (
        recovered_metric - ablated_metric
        if higher_is_better
        else ablated_metric - recovered_metric
    )


@dataclass(frozen=True)
class MatchedClassicalRecoveryEvidence:
    study_id: str
    participant_id: str
    occasion_id: str
    case_id: str
    mechanism_id: str
    classical_model_id: str
    information_set_id: str
    metric_name: str
    higher_is_better: bool
    baseline_metric: float
    ablated_metric: float
    recovered_metric: float
    candidate_evidence_fingerprint: str
    classical_evidence_fingerprint: str
    source_fingerprint: str

    def __post_init__(self) -> None:
        for name in (
            "study_id",
            "participant_id",
            "occasion_id",
            "case_id",
            "mechanism_id",
            "classical_model_id",
            "information_set_id",
            "metric_name",
            "candidate_evidence_fingerprint",
            "classical_evidence_fingerprint",
            "source_fingerprint",
        ):
            _required_text(name, getattr(self, name))
        for name in ("baseline_metric", "ablated_metric", "recovered_metric"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if not isinstance(self.higher_is_better, bool):
            raise TypeError("higher_is_better must be a JSON boolean")
        if self.ablation_loss <= 0.0:
            raise ValueError(
                "matched recovery requires a strictly positive candidate ablation loss"
            )

    @property
    def ablation_loss(self) -> float:
        return float(
            _oriented_loss(
                baseline_metric=self.baseline_metric,
                ablated_metric=self.ablated_metric,
                higher_is_better=self.higher_is_better,
            )
        )

    @property
    def restored_loss(self) -> float:
        # A matched classical control that worsens the ablated state provides zero
        # recovery rather than a negative recovery credit.
        return float(
            max(
                0.0,
                _oriented_restoration(
                    ablated_metric=self.ablated_metric,
                    recovered_metric=self.recovered_metric,
                    higher_is_better=self.higher_is_better,
                ),
            )
        )

    @property
    def recovery_fraction(self) -> float:
        return float(self.restored_loss / self.ablation_loss)

    def scientific_identity(self) -> dict[str, Any]:
        return {
            "schema_version": MATCHED_RECOVERY_SCHEMA,
            "method_id": MATCHED_RECOVERY_METHOD_ID,
            "study_id": self.study_id,
            "participant_id": self.participant_id,
            "occasion_id": self.occasion_id,
            "case_id": self.case_id,
            "mechanism_id": self.mechanism_id,
            "classical_model_id": self.classical_model_id,
            "information_set_id": self.information_set_id,
            "metric_name": self.metric_name,
            "higher_is_better": self.higher_is_better,
            "baseline_metric": float(self.baseline_metric),
            "ablated_metric": float(self.ablated_metric),
            "recovered_metric": float(self.recovered_metric),
            "candidate_evidence_fingerprint": self.candidate_evidence_fingerprint,
            "classical_evidence_fingerprint": self.classical_evidence_fingerprint,
        }

    def to_mapping(self) -> dict[str, Any]:
        return {
            **self.scientific_identity(),
            "artifact_role": "matched_classical_recovery_evidence",
            "ablation_loss": self.ablation_loss,
            "restored_loss": self.restored_loss,
            "classical_recovery_fraction": self.recovery_fraction,
            "source_fingerprint": self.source_fingerprint,
            "interpretation": (
                "Recovery is the fraction of candidate-mechanism ablation loss restored by "
                "the declared classical model under the same information-set authority. "
                "Lower recovery is stronger evidence that the candidate mechanism is necessary "
                "under this benchmark."
            ),
        }

    def as_causal_recovery(self) -> MatchedClassicalRecovery:
        return MatchedClassicalRecovery(
            classical_model_id=self.classical_model_id,
            classical_recovery_fraction=self.recovery_fraction,
            information_set_id=self.information_set_id,
            source_fingerprint=self.source_fingerprint,
        )


def build_matched_classical_recovery_evidence(
    *,
    study_id: str,
    participant_id: str,
    occasion_id: str,
    case_id: str,
    mechanism_id: str,
    classical_model_id: str,
    information_set_id: str,
    metric_name: str,
    higher_is_better: bool,
    baseline_metric: float,
    ablated_metric: float,
    recovered_metric: float,
    candidate_evidence_fingerprint: str,
    classical_evidence_fingerprint: str,
) -> MatchedClassicalRecoveryEvidence:
    identity = {
        "schema_version": MATCHED_RECOVERY_SCHEMA,
        "method_id": MATCHED_RECOVERY_METHOD_ID,
        "study_id": _required_text("study_id", study_id),
        "participant_id": _required_text("participant_id", participant_id),
        "occasion_id": _required_text("occasion_id", occasion_id),
        "case_id": _required_text("case_id", case_id),
        "mechanism_id": _required_text("mechanism_id", mechanism_id),
        "classical_model_id": _required_text("classical_model_id", classical_model_id),
        "information_set_id": _required_text("information_set_id", information_set_id),
        "metric_name": _required_text("metric_name", metric_name),
        "higher_is_better": higher_is_better,
        "baseline_metric": _finite("baseline_metric", baseline_metric),
        "ablated_metric": _finite("ablated_metric", ablated_metric),
        "recovered_metric": _finite("recovered_metric", recovered_metric),
        "candidate_evidence_fingerprint": _required_text(
            "candidate_evidence_fingerprint", candidate_evidence_fingerprint
        ),
        "classical_evidence_fingerprint": _required_text(
            "classical_evidence_fingerprint", classical_evidence_fingerprint
        ),
    }
    if not isinstance(higher_is_better, bool):
        raise TypeError("higher_is_better must be a JSON boolean")
    return MatchedClassicalRecoveryEvidence(
        study_id=identity["study_id"],
        participant_id=identity["participant_id"],
        occasion_id=identity["occasion_id"],
        case_id=identity["case_id"],
        mechanism_id=identity["mechanism_id"],
        classical_model_id=identity["classical_model_id"],
        information_set_id=identity["information_set_id"],
        metric_name=identity["metric_name"],
        higher_is_better=identity["higher_is_better"],
        baseline_metric=identity["baseline_metric"],
        ablated_metric=identity["ablated_metric"],
        recovered_metric=identity["recovered_metric"],
        candidate_evidence_fingerprint=identity["candidate_evidence_fingerprint"],
        classical_evidence_fingerprint=identity["classical_evidence_fingerprint"],
        source_fingerprint=_fingerprint(identity),
    )


def matched_classical_recovery_from_mapping(
    payload: Mapping[str, Any],
) -> MatchedClassicalRecoveryEvidence:
    if payload.get("schema_version") != MATCHED_RECOVERY_SCHEMA:
        raise ValueError(
            "unsupported matched-classical recovery schema: "
            f"{payload.get('schema_version')!r}"
        )
    if payload.get("artifact_role") != "matched_classical_recovery_evidence":
        raise ValueError("matched recovery artifact has the wrong artifact_role")
    if payload.get("method_id") != MATCHED_RECOVERY_METHOD_ID:
        raise ValueError("matched recovery artifact has the wrong method_id")
    if not isinstance(payload.get("higher_is_better"), bool):
        raise TypeError("matched recovery higher_is_better must be a JSON boolean")

    evidence = build_matched_classical_recovery_evidence(
        study_id=payload.get("study_id"),
        participant_id=payload.get("participant_id"),
        occasion_id=payload.get("occasion_id"),
        case_id=payload.get("case_id"),
        mechanism_id=payload.get("mechanism_id"),
        classical_model_id=payload.get("classical_model_id"),
        information_set_id=payload.get("information_set_id"),
        metric_name=payload.get("metric_name"),
        higher_is_better=payload.get("higher_is_better"),
        baseline_metric=payload.get("baseline_metric"),
        ablated_metric=payload.get("ablated_metric"),
        recovered_metric=payload.get("recovered_metric"),
        candidate_evidence_fingerprint=payload.get("candidate_evidence_fingerprint"),
        classical_evidence_fingerprint=payload.get("classical_evidence_fingerprint"),
    )
    claimed_source = _required_text("source_fingerprint", payload.get("source_fingerprint"))
    if claimed_source != evidence.source_fingerprint:
        raise ValueError("matched-classical recovery source fingerprint mismatch")

    derived = {
        "ablation_loss": evidence.ablation_loss,
        "restored_loss": evidence.restored_loss,
        "classical_recovery_fraction": evidence.recovery_fraction,
    }
    for key, expected in derived.items():
        if key not in payload:
            raise ValueError(f"matched recovery artifact is missing derived field {key!r}")
        if abs(float(payload[key]) - float(expected)) > 1e-12:
            raise ValueError(f"matched recovery derived field mismatch: {key}")
    return evidence
