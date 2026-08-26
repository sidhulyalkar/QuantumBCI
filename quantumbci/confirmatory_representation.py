"""Confirmatory cross-representation protocol for publication-grade BMRB studies.

This module is deliberately a new scientific surface rather than a silent reinterpretation
of BMRB-Representation v1.  V1 remains a qualified exploratory/conservation artifact.
The confirmatory protocol fixes the primary calibration budget and primary classical
control before evaluation, binds the decision policy to external preregistration evidence,
and reports participant-level uncertainty without averaging distinct calibration budgets
into one primary estimand.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .claims import ClaimClass
from .exporting import verify_run_artifacts
from .inference import ParticipantEffectInference, participant_effect_inference
from .preregistration import PreregistrationEvidence, canonical_scientific_fingerprint
from .recapitulation import (
    EvidenceGate,
    EvidenceTier,
    GateStatus,
    MechanismNecessityProfile,
    RecapitulationSignature,
)
from .representation_studies import E001_REPRESENTATION_LANE_SCHEMA

CONFIRMATORY_REPRESENTATION_BENCHMARK = "BMRB_REPRESENTATION_CONFIRMATORY_V2"
CONFIRMATORY_REPRESENTATION_METHOD_ID = "participant_primary_budget_confirmatory_v2"


def _required_text(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _strict_bool(name: str, value: Any) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a JSON/Python boolean")
    return value


def _finite(name: str, value: Any) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _fraction(name: str, value: Any) -> float:
    number = _finite(name, value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return number


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric_balanced_accuracy(benchmark: Mapping[str, Any], method: str) -> float:
    metrics = benchmark.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("E001 benchmark is missing metrics mapping")
    metric = metrics.get(method)
    if not isinstance(metric, Mapping):
        raise ValueError(f"E001 benchmark is missing preregistered control/method {method!r}")
    return _finite(f"balanced_accuracy[{method}]", metric.get("balanced_accuracy"))


@dataclass(frozen=True)
class ConfirmatoryRepresentationPolicy:
    """Study-specific confirmatory policy with no universal biological thresholds."""

    policy_id: str
    reference_representation_id: str
    primary_calibration_per_class: int
    primary_classical_control: str
    min_participants: int
    min_representations: int
    min_representation_families: int
    min_candidate_advantage: float
    min_ablation_necessity: float
    min_reference_positive_fraction: float
    min_all_lane_positive_fraction: float
    min_direction_match_fraction: float
    min_ablation_direction_match_fraction: float
    min_information_novel_representation_fraction: float
    sample_size_rationale: str
    inference_seed: int = 1801
    bootstrap_resamples: int = 5000
    preregistration: PreregistrationEvidence | None = None

    def __post_init__(self) -> None:
        for name in (
            "policy_id",
            "reference_representation_id",
            "primary_classical_control",
            "sample_size_rationale",
        ):
            object.__setattr__(self, name, _required_text(name, getattr(self, name)))
        if int(self.primary_calibration_per_class) < 0:
            raise ValueError("primary_calibration_per_class must be non-negative")
        object.__setattr__(
            self, "primary_calibration_per_class", int(self.primary_calibration_per_class)
        )
        if int(self.min_participants) < 2:
            raise ValueError("min_participants must be at least 2")
        if int(self.min_representations) < 2:
            raise ValueError("min_representations must be at least 2")
        if int(self.min_representation_families) < 1:
            raise ValueError("min_representation_families must be positive")
        object.__setattr__(self, "min_participants", int(self.min_participants))
        object.__setattr__(self, "min_representations", int(self.min_representations))
        object.__setattr__(
            self, "min_representation_families", int(self.min_representation_families)
        )
        object.__setattr__(
            self, "min_candidate_advantage", _finite("min_candidate_advantage", self.min_candidate_advantage)
        )
        object.__setattr__(
            self, "min_ablation_necessity", _finite("min_ablation_necessity", self.min_ablation_necessity)
        )
        for name in (
            "min_reference_positive_fraction",
            "min_all_lane_positive_fraction",
            "min_direction_match_fraction",
            "min_ablation_direction_match_fraction",
            "min_information_novel_representation_fraction",
        ):
            object.__setattr__(self, name, _fraction(name, getattr(self, name)))
        if int(self.bootstrap_resamples) < 100:
            raise ValueError("bootstrap_resamples must be at least 100")
        object.__setattr__(self, "bootstrap_resamples", int(self.bootstrap_resamples))
        object.__setattr__(self, "inference_seed", int(self.inference_seed))

    def decision_payload(self) -> dict[str, Any]:
        """Exact policy content that must have been registered before confirmatory analysis."""

        return {
            "schema_version": 2,
            "benchmark": CONFIRMATORY_REPRESENTATION_BENCHMARK,
            "policy_id": self.policy_id,
            "reference_representation_id": self.reference_representation_id,
            "primary_calibration_per_class": self.primary_calibration_per_class,
            "primary_classical_control": self.primary_classical_control,
            "min_participants": self.min_participants,
            "min_representations": self.min_representations,
            "min_representation_families": self.min_representation_families,
            "min_candidate_advantage": self.min_candidate_advantage,
            "min_ablation_necessity": self.min_ablation_necessity,
            "min_reference_positive_fraction": self.min_reference_positive_fraction,
            "min_all_lane_positive_fraction": self.min_all_lane_positive_fraction,
            "min_direction_match_fraction": self.min_direction_match_fraction,
            "min_ablation_direction_match_fraction": self.min_ablation_direction_match_fraction,
            "min_information_novel_representation_fraction": self.min_information_novel_representation_fraction,
            "sample_size_rationale": self.sample_size_rationale,
            "inference_seed": self.inference_seed,
            "bootstrap_resamples": self.bootstrap_resamples,
        }

    @property
    def decision_fingerprint(self) -> str:
        return canonical_scientific_fingerprint(
            "quantumbci.confirmatory-representation-policy.v2", self.decision_payload()
        )

    @property
    def confirmatory_authority(self) -> bool:
        return bool(
            self.preregistration is not None
            and self.preregistration.matches_policy(self.decision_fingerprint)
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            **self.decision_payload(),
            "decision_fingerprint": self.decision_fingerprint,
            "preregistration": (
                None if self.preregistration is None else self.preregistration.to_mapping()
            ),
            "confirmatory_authority": self.confirmatory_authority,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ConfirmatoryRepresentationPolicy":
        registration_payload = payload.get("preregistration")
        if registration_payload is not None and not isinstance(registration_payload, Mapping):
            raise ValueError("policy.preregistration must be an object or null")
        policy = cls(
            policy_id=_required_text("policy_id", payload.get("policy_id")),
            reference_representation_id=_required_text(
                "reference_representation_id", payload.get("reference_representation_id")
            ),
            primary_calibration_per_class=int(payload.get("primary_calibration_per_class", -1)),
            primary_classical_control=_required_text(
                "primary_classical_control", payload.get("primary_classical_control")
            ),
            min_participants=int(payload.get("min_participants", 0)),
            min_representations=int(payload.get("min_representations", 0)),
            min_representation_families=int(payload.get("min_representation_families", 0)),
            min_candidate_advantage=float(payload.get("min_candidate_advantage")),
            min_ablation_necessity=float(payload.get("min_ablation_necessity")),
            min_reference_positive_fraction=float(payload.get("min_reference_positive_fraction")),
            min_all_lane_positive_fraction=float(payload.get("min_all_lane_positive_fraction")),
            min_direction_match_fraction=float(payload.get("min_direction_match_fraction")),
            min_ablation_direction_match_fraction=float(
                payload.get("min_ablation_direction_match_fraction")
            ),
            min_information_novel_representation_fraction=float(
                payload.get("min_information_novel_representation_fraction")
            ),
            sample_size_rationale=_required_text(
                "sample_size_rationale", payload.get("sample_size_rationale")
            ),
            inference_seed=int(payload.get("inference_seed", 1801)),
            bootstrap_resamples=int(payload.get("bootstrap_resamples", 5000)),
            preregistration=(
                None
                if registration_payload is None
                else PreregistrationEvidence.from_mapping(registration_payload)
            ),
        )
        supplied_fingerprint = payload.get("decision_fingerprint")
        if supplied_fingerprint is not None and str(supplied_fingerprint) != policy.decision_fingerprint:
            raise ValueError("policy decision_fingerprint does not match reconstructed policy")
        return policy


@dataclass(frozen=True)
class ConfirmatoryRepresentationObservation:
    participant_id: str
    occasion_id: str
    case_id: str
    calibration_per_class: int
    representation_id: str
    representation_family: str
    authority_fingerprint: str
    representation_sha256: str
    source_fingerprint: str
    candidate_metric: float
    primary_control_metric: float
    ablated_metric: float
    information_novel: bool
    model_id: str | None = None
    model_revision: str | None = None

    @property
    def key(self) -> tuple[str, str, str, int]:
        return (
            self.participant_id,
            self.occasion_id,
            self.case_id,
            self.calibration_per_class,
        )

    @property
    def candidate_advantage(self) -> float:
        return self.candidate_metric - self.primary_control_metric

    @property
    def ablation_necessity(self) -> float:
        return self.candidate_metric - self.ablated_metric


@dataclass(frozen=True)
class ConfirmatoryLaneSummary:
    representation_id: str
    representation_family: str
    participant_count: int
    case_count: int
    candidate: ParticipantEffectInference
    ablation: ParticipantEffectInference
    information_novel_fraction: float
    model_id: str | None = None
    model_revision: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "representation_id": self.representation_id,
            "representation_family": self.representation_family,
            "participant_count": self.participant_count,
            "case_count": self.case_count,
            "candidate_advantage": self.candidate.to_mapping(),
            "ablation_necessity": self.ablation.to_mapping(),
            "information_novel_fraction": self.information_novel_fraction,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
        }


@dataclass(frozen=True)
class CalibrationFrontierPoint:
    calibration_per_class: int
    lane_mean_candidate_advantage: Mapping[str, float]
    lane_mean_ablation_necessity: Mapping[str, float]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "calibration_per_class": self.calibration_per_class,
            "lane_mean_candidate_advantage": dict(self.lane_mean_candidate_advantage),
            "lane_mean_ablation_necessity": dict(self.lane_mean_ablation_necessity),
            "role": "secondary_descriptive_frontier",
        }


@dataclass(frozen=True)
class ConfirmatoryRepresentationResult:
    study_id: str
    mechanism_id: str
    policy: ConfirmatoryRepresentationPolicy
    lanes: tuple[ConfirmatoryLaneSummary, ...]
    observations: tuple[ConfirmatoryRepresentationObservation, ...]
    available_calibration_budgets: tuple[int, ...]
    calibration_frontier: tuple[CalibrationFrontierPoint, ...]
    participant_count: int
    representation_count: int
    representation_family_count: int
    reference_positive_fraction: float
    all_lane_positive_fraction: float
    direction_match_fraction: float
    ablation_direction_match_fraction: float
    information_novel_representation_fraction: float
    scientific_criteria_passed: bool
    criteria_reasons: tuple[str, ...]
    source_fingerprint: str

    @property
    def promotion_eligible(self) -> bool:
        return bool(self.policy.confirmatory_authority and self.scientific_criteria_passed)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "artifact_role": "confirmatory_representation_evidence",
            "benchmark": CONFIRMATORY_REPRESENTATION_BENCHMARK,
            "method_id": CONFIRMATORY_REPRESENTATION_METHOD_ID,
            "study_id": self.study_id,
            "mechanism_id": self.mechanism_id,
            "policy": self.policy.to_mapping(),
            "primary_calibration_per_class": self.policy.primary_calibration_per_class,
            "available_calibration_budgets": list(self.available_calibration_budgets),
            "participant_count": self.participant_count,
            "representation_count": self.representation_count,
            "representation_family_count": self.representation_family_count,
            "reference_positive_fraction": self.reference_positive_fraction,
            "all_lane_positive_fraction": self.all_lane_positive_fraction,
            "direction_match_fraction": self.direction_match_fraction,
            "ablation_direction_match_fraction": self.ablation_direction_match_fraction,
            "information_novel_representation_fraction": self.information_novel_representation_fraction,
            "scientific_criteria_passed": self.scientific_criteria_passed,
            "confirmatory_authority": self.policy.confirmatory_authority,
            "promotion_eligible": self.promotion_eligible,
            "criteria_reasons": list(self.criteria_reasons),
            "lanes": [lane.to_mapping() for lane in self.lanes],
            "calibration_frontier": [point.to_mapping() for point in self.calibration_frontier],
            "source_fingerprint": self.source_fingerprint,
            "physical_quantum_promotion_eligible": False,
            "interpretation": (
                "The primary estimand uses exactly one preregistered calibration budget and one "
                "preregistered classical control. Other calibration budgets are secondary. "
                "Cross-representation recurrence is not causal or physical-quantum evidence."
            ),
        }


def confirmatory_representation_signature() -> RecapitulationSignature:
    return RecapitulationSignature(
        id="BMRB_REPRESENTATION_CONFIRMATORY_V2",
        title="Confirmatory held-out neural mechanism conservation across representations",
        domain="representation_mechanism_conservation",
        target=(
            "participant-level candidate-vs-preregistered-control and ablation effects at one "
            "predeclared calibration budget across exact-paired representation lanes"
        ),
        inference_unit="participant",
        primary_metric="participant_balanced_candidate_advantage",
        favorable_direction="greater_than_preregistered_minimum_effect",
        required_controls=(
            "predeclared_primary_classical_control",
            "candidate_ablation",
            "exact_evidence_authority",
            "external_preregistration_binding",
        ),
        description=(
            "Separates confirmatory effect estimation from descriptive calibration frontiers and "
            "requires external preregistration evidence before any PASS can authorize promotion."
        ),
    )


def _participant_means(
    observations: Sequence[ConfirmatoryRepresentationObservation], field: str
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for item in observations:
        grouped.setdefault(item.participant_id, []).append(float(getattr(item, field)))
    return {key: float(np.mean(values)) for key, values in sorted(grouped.items())}


def _validate_exact_pairing(
    observations: Sequence[ConfirmatoryRepresentationObservation],
    reference_representation_id: str,
) -> None:
    by_lane: dict[str, dict[tuple[str, str, str, int], ConfirmatoryRepresentationObservation]] = {}
    for item in observations:
        lane = by_lane.setdefault(item.representation_id, {})
        if item.key in lane:
            raise ValueError(f"duplicate confirmatory observation in {item.representation_id}: {item.key}")
        lane[item.key] = item
    if reference_representation_id not in by_lane:
        raise ValueError("reference representation is absent")
    reference = by_lane[reference_representation_id]
    keys = set(reference)
    for lane_id, lane in sorted(by_lane.items()):
        if set(lane) != keys:
            missing = sorted(keys - set(lane))
            extra = sorted(set(lane) - keys)
            raise ValueError(
                f"confirmatory representation lanes must be exactly paired; lane={lane_id!r} "
                f"missing={missing} extra={extra}"
            )
        for key in sorted(keys):
            if lane[key].authority_fingerprint != reference[key].authority_fingerprint:
                raise ValueError(f"authority fingerprint mismatch for key={key}, lane={lane_id!r}")


def _lane_summary(
    observations: Sequence[ConfirmatoryRepresentationObservation],
    *,
    policy: ConfirmatoryRepresentationPolicy,
    lane_offset: int,
) -> ConfirmatoryLaneSummary:
    first = observations[0]
    effects = _participant_means(observations, "candidate_advantage")
    ablations = _participant_means(observations, "ablation_necessity")
    return ConfirmatoryLaneSummary(
        representation_id=first.representation_id,
        representation_family=first.representation_family,
        participant_count=len(effects),
        case_count=len(observations),
        candidate=participant_effect_inference(
            effects.values(),
            bootstrap_resamples=policy.bootstrap_resamples,
            seed=policy.inference_seed + lane_offset * 2,
        ),
        ablation=participant_effect_inference(
            ablations.values(),
            bootstrap_resamples=policy.bootstrap_resamples,
            seed=policy.inference_seed + lane_offset * 2 + 1,
        ),
        information_novel_fraction=float(np.mean([item.information_novel for item in observations])),
        model_id=first.model_id,
        model_revision=first.model_revision,
    )


def _frontier(
    observations: Sequence[ConfirmatoryRepresentationObservation],
) -> tuple[CalibrationFrontierPoint, ...]:
    budgets = sorted({item.calibration_per_class for item in observations})
    lanes = sorted({item.representation_id for item in observations})
    points: list[CalibrationFrontierPoint] = []
    for budget in budgets:
        effect: dict[str, float] = {}
        ablation: dict[str, float] = {}
        for lane in lanes:
            subset = [
                item
                for item in observations
                if item.calibration_per_class == budget and item.representation_id == lane
            ]
            effect[lane] = float(np.mean(list(_participant_means(subset, "candidate_advantage").values())))
            ablation[lane] = float(np.mean(list(_participant_means(subset, "ablation_necessity").values())))
        points.append(
            CalibrationFrontierPoint(
                calibration_per_class=budget,
                lane_mean_candidate_advantage=effect,
                lane_mean_ablation_necessity=ablation,
            )
        )
    return tuple(points)


def evaluate_confirmatory_representation(
    observations: Iterable[ConfirmatoryRepresentationObservation],
    *,
    study_id: str,
    mechanism_id: str,
    policy: ConfirmatoryRepresentationPolicy,
) -> ConfirmatoryRepresentationResult:
    materialized = tuple(observations)
    if not materialized:
        raise ValueError("confirmatory representation analysis requires observations")
    _validate_exact_pairing(materialized, policy.reference_representation_id)
    budgets = tuple(sorted({item.calibration_per_class for item in materialized}))
    if policy.primary_calibration_per_class not in budgets:
        raise ValueError(
            "primary_calibration_per_class is absent from the exact-paired calibration frontier"
        )
    primary = tuple(
        item
        for item in materialized
        if item.calibration_per_class == policy.primary_calibration_per_class
    )
    by_lane: dict[str, list[ConfirmatoryRepresentationObservation]] = {}
    for item in primary:
        by_lane.setdefault(item.representation_id, []).append(item)
    if policy.reference_representation_id not in by_lane:
        raise ValueError("reference representation has no primary-budget observations")

    lane_ids = sorted(by_lane)
    summaries = tuple(
        _lane_summary(by_lane[lane_id], policy=policy, lane_offset=index)
        for index, lane_id in enumerate(lane_ids)
    )
    participants = sorted({item.participant_id for item in primary})
    families = {item.representation_family for item in primary}
    lane_effects = {
        lane_id: _participant_means(by_lane[lane_id], "candidate_advantage")
        for lane_id in lane_ids
    }
    lane_ablations = {
        lane_id: _participant_means(by_lane[lane_id], "ablation_necessity")
        for lane_id in lane_ids
    }
    reference = lane_effects[policy.reference_representation_id]
    reference_ablation = lane_ablations[policy.reference_representation_id]
    reference_positive = float(
        np.mean([reference[p] >= policy.min_candidate_advantage for p in participants])
    )
    all_effects = [lane_effects[lane][p] for lane in lane_ids for p in participants]
    all_lane_positive = float(
        np.mean([value >= policy.min_candidate_advantage for value in all_effects])
    )
    direction_matches: list[bool] = []
    ablation_matches: list[bool] = []
    for lane_id in lane_ids:
        if lane_id == policy.reference_representation_id:
            continue
        for participant in participants:
            ref_effect = reference[participant] - policy.min_candidate_advantage
            lane_effect = lane_effects[lane_id][participant] - policy.min_candidate_advantage
            direction_matches.append(np.sign(ref_effect) == np.sign(lane_effect))
            ref_abl = reference_ablation[participant] - policy.min_ablation_necessity
            lane_abl = lane_ablations[lane_id][participant] - policy.min_ablation_necessity
            ablation_matches.append(np.sign(ref_abl) == np.sign(lane_abl))
    direction_match = float(np.mean(direction_matches)) if direction_matches else 1.0
    ablation_match = float(np.mean(ablation_matches)) if ablation_matches else 1.0
    novelty = float(np.mean([summary.information_novel_fraction >= 1.0 - 1e-12 for summary in summaries]))

    reasons: list[str] = []
    if len(participants) < policy.min_participants:
        reasons.append(f"participants {len(participants)} < preregistered {policy.min_participants}")
    if len(summaries) < policy.min_representations:
        reasons.append(f"representations {len(summaries)} < preregistered {policy.min_representations}")
    if len(families) < policy.min_representation_families:
        reasons.append(
            f"representation families {len(families)} < preregistered {policy.min_representation_families}"
        )
    if reference_positive < policy.min_reference_positive_fraction:
        reasons.append(
            f"reference positive fraction {reference_positive:.3f} < preregistered "
            f"{policy.min_reference_positive_fraction:.3f}"
        )
    if all_lane_positive < policy.min_all_lane_positive_fraction:
        reasons.append(
            f"all-lane positive fraction {all_lane_positive:.3f} < preregistered "
            f"{policy.min_all_lane_positive_fraction:.3f}"
        )
    if direction_match < policy.min_direction_match_fraction:
        reasons.append(
            f"direction match {direction_match:.3f} < preregistered {policy.min_direction_match_fraction:.3f}"
        )
    if ablation_match < policy.min_ablation_direction_match_fraction:
        reasons.append(
            f"ablation direction match {ablation_match:.3f} < preregistered "
            f"{policy.min_ablation_direction_match_fraction:.3f}"
        )
    for summary in summaries:
        if summary.candidate.observed_mean < policy.min_candidate_advantage:
            reasons.append(
                f"lane {summary.representation_id} mean candidate advantage "
                f"{summary.candidate.observed_mean:.6g} < preregistered {policy.min_candidate_advantage:.6g}"
            )
        if summary.ablation.observed_mean < policy.min_ablation_necessity:
            reasons.append(
                f"lane {summary.representation_id} mean ablation necessity "
                f"{summary.ablation.observed_mean:.6g} < preregistered {policy.min_ablation_necessity:.6g}"
            )
    if novelty < policy.min_information_novel_representation_fraction:
        reasons.append(
            f"information-novel representation fraction {novelty:.3f} < preregistered "
            f"{policy.min_information_novel_representation_fraction:.3f}"
        )

    identity = {
        "study_id": _required_text("study_id", study_id),
        "mechanism_id": _required_text("mechanism_id", mechanism_id),
        "policy": policy.to_mapping(),
        "observations": [
            {
                "participant_id": item.participant_id,
                "occasion_id": item.occasion_id,
                "case_id": item.case_id,
                "calibration_per_class": item.calibration_per_class,
                "representation_id": item.representation_id,
                "representation_family": item.representation_family,
                "authority_fingerprint": item.authority_fingerprint,
                "representation_sha256": item.representation_sha256,
                "source_fingerprint": item.source_fingerprint,
                "candidate_metric": item.candidate_metric,
                "primary_control_metric": item.primary_control_metric,
                "ablated_metric": item.ablated_metric,
                "information_novel": item.information_novel,
                "model_id": item.model_id,
                "model_revision": item.model_revision,
            }
            for item in sorted(
                materialized,
                key=lambda value: (
                    value.representation_id,
                    value.participant_id,
                    value.occasion_id,
                    value.case_id,
                    value.calibration_per_class,
                ),
            )
        ],
    }
    source_fingerprint = canonical_scientific_fingerprint(
        "quantumbci.confirmatory-representation-evidence.v2", identity
    )
    return ConfirmatoryRepresentationResult(
        study_id=identity["study_id"],
        mechanism_id=identity["mechanism_id"],
        policy=policy,
        lanes=summaries,
        observations=materialized,
        available_calibration_budgets=budgets,
        calibration_frontier=_frontier(materialized),
        participant_count=len(participants),
        representation_count=len(summaries),
        representation_family_count=len(families),
        reference_positive_fraction=reference_positive,
        all_lane_positive_fraction=all_lane_positive,
        direction_match_fraction=direction_match,
        ablation_direction_match_fraction=ablation_match,
        information_novel_representation_fraction=novelty,
        scientific_criteria_passed=not reasons,
        criteria_reasons=tuple(reasons),
        source_fingerprint=source_fingerprint,
    )


def build_confirmatory_representation_profile(
    result: ConfirmatoryRepresentationResult,
) -> MechanismNecessityProfile:
    """Map the v2 result onto BMRB without allowing retrospective PASS labels."""

    authority = result.policy.confirmatory_authority
    scientific = result.scientific_criteria_passed
    descriptive = EvidenceGate(
        id="paired_representation_authority",
        tier=EvidenceTier.DESCRIPTIVE,
        status=GateStatus.PASS,
        summary="Exact participant/occasion/case/budget and neurOS authority pairing passed.",
        evidence_ref=result.source_fingerprint,
        threshold="exact paired key and authority equality",
    )
    if not scientific:
        predictive_status = GateStatus.FAIL
    elif authority:
        predictive_status = GateStatus.PASS
    else:
        predictive_status = GateStatus.CHARACTERIZED
    predictive = EvidenceGate(
        id="confirmatory_primary_effect",
        tier=EvidenceTier.PREDICTIVE,
        status=predictive_status,
        summary=(
            "Primary-budget candidate and ablation effect criteria were evaluated against the "
            "predeclared classical control."
        ),
        evidence_ref=result.source_fingerprint,
        metric="participant_primary_budget_effect",
        value=float(min(summary.candidate.observed_mean for summary in result.lanes)),
        threshold=(
            f"candidate>={result.policy.min_candidate_advantage}; "
            f"ablation>={result.policy.min_ablation_necessity}"
            if authority and scientific
            else None
        ),
    )
    novelty_ok = (
        result.information_novel_representation_fraction
        >= result.policy.min_information_novel_representation_fraction
    )
    if not novelty_ok:
        adversary_status = GateStatus.FAIL
    elif predictive.status == GateStatus.PASS and authority:
        adversary_status = GateStatus.PASS
    else:
        adversary_status = GateStatus.CHARACTERIZED
    adversary = EvidenceGate(
        id="matched_representation_adversaries",
        tier=EvidenceTier.ADVERSARY_SURVIVING,
        status=adversary_status,
        summary="Information novelty is evaluated separately from cross-representation recurrence.",
        evidence_ref=result.source_fingerprint,
        metric="information_novel_representation_fraction",
        value=result.information_novel_representation_fraction,
        threshold=(
            f">={result.policy.min_information_novel_representation_fraction}"
            if adversary_status == GateStatus.PASS
            else None
        ),
    )
    stability_ok = (
        result.direction_match_fraction >= result.policy.min_direction_match_fraction
        and result.ablation_direction_match_fraction
        >= result.policy.min_ablation_direction_match_fraction
    )
    if not stability_ok:
        stability_status = GateStatus.FAIL
    elif adversary.status == GateStatus.PASS and authority:
        stability_status = GateStatus.PASS
    else:
        stability_status = GateStatus.CHARACTERIZED
    stability = EvidenceGate(
        id="cross_representation_stability",
        tier=EvidenceTier.SOURCE_STABILITY,
        status=stability_status,
        summary="Primary-budget candidate and ablation directions were compared across lanes.",
        evidence_ref=result.source_fingerprint,
        metric="minimum_direction_match_fraction",
        value=min(result.direction_match_fraction, result.ablation_direction_match_fraction),
        threshold=(
            f"candidate>={result.policy.min_direction_match_fraction}; "
            f"ablation>={result.policy.min_ablation_direction_match_fraction}"
            if stability_status == GateStatus.PASS
            else None
        ),
    )
    repeated_ok = (
        result.participant_count >= result.policy.min_participants
        and result.representation_count >= result.policy.min_representations
        and result.representation_family_count >= result.policy.min_representation_families
    )
    if not repeated_ok:
        repeated_status = GateStatus.FAIL
    elif stability.status == GateStatus.PASS and authority:
        repeated_status = GateStatus.PASS
    else:
        repeated_status = GateStatus.CHARACTERIZED
    repeated = EvidenceGate(
        id="cross_representation_replication",
        tier=EvidenceTier.REPEATED_CASE,
        status=repeated_status,
        summary="Participant and representation-family replication requirements were evaluated.",
        evidence_ref=result.source_fingerprint,
        metric="participant_and_representation_coverage",
        value=float(result.participant_count),
        threshold=(
            f"participants>={result.policy.min_participants}; "
            f"representations>={result.policy.min_representations}; "
            f"families>={result.policy.min_representation_families}"
            if repeated_status == GateStatus.PASS
            else None
        ),
    )
    causal = EvidenceGate(
        id="causal_intervention_and_ablation",
        tier=EvidenceTier.CAUSAL_MECHANISTIC,
        status=GateStatus.NOT_RUN,
        summary="Cross-representation recurrence is not causal necessity.",
    )
    physical = EvidenceGate(
        id="physical_quantum_witness",
        tier=EvidenceTier.PHYSICAL_QUANTUM,
        status=GateStatus.NOT_APPLICABLE,
        summary="Physical-quantum promotion requires an independent physical witness protocol.",
    )
    return MechanismNecessityProfile(
        mechanism_id=result.mechanism_id,
        claim_class=ClaimClass.QUANTUM_INSPIRED,
        signature=confirmatory_representation_signature(),
        gates=(descriptive, predictive, adversary, stability, repeated, causal, physical),
        metadata={
            "confirmatory_authority": authority,
            "policy_fingerprint": result.policy.decision_fingerprint,
            "primary_calibration_per_class": result.policy.primary_calibration_per_class,
            "primary_classical_control": result.policy.primary_classical_control,
        },
    )


def _resolve_dir(root: Path, value: Any, *, label: str) -> Path:
    path = Path(_required_text(label, value)).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def load_confirmatory_representation_manifest(
    manifest_path: str | Path,
) -> tuple[str, str, ConfirmatoryRepresentationPolicy, tuple[ConfirmatoryRepresentationObservation, ...], dict[str, Any]]:
    """Load verified v1 lane bundles into the stricter v2 confirmatory estimand."""

    path = Path(manifest_path).expanduser().resolve()
    manifest = _load_json(path, label="confirmatory representation manifest")
    if int(manifest.get("schema_version", 0)) != 2:
        raise ValueError("confirmatory representation manifest schema_version must be 2")
    study_id = _required_text("study_id", manifest.get("study_id"))
    mechanism_id = _required_text("mechanism_id", manifest.get("mechanism_id"))
    participant_key = _required_text("participant_key", manifest.get("participant_key", "subject"))
    policy_payload = manifest.get("policy")
    if not isinstance(policy_payload, Mapping):
        raise ValueError("confirmatory representation manifest is missing policy")
    policy = ConfirmatoryRepresentationPolicy.from_mapping(policy_payload)
    raw_lanes = manifest.get("lanes")
    if not isinstance(raw_lanes, list) or len(raw_lanes) < 2:
        raise ValueError("confirmatory representation manifest requires at least two lanes")

    observations: list[ConfirmatoryRepresentationObservation] = []
    lane_ids: set[str] = set()
    for lane_index, raw_lane in enumerate(raw_lanes):
        if not isinstance(raw_lane, Mapping):
            raise ValueError(f"lanes[{lane_index}] must be an object")
        lane_id = _required_text(f"lanes[{lane_index}].lane_id", raw_lane.get("lane_id"))
        if lane_id in lane_ids:
            raise ValueError(f"duplicate lane_id: {lane_id!r}")
        lane_ids.add(lane_id)
        artifact_dir = _resolve_dir(
            path.parent,
            raw_lane.get("artifact_dir"),
            label=f"lanes[{lane_index}].artifact_dir",
        )
        verification = verify_run_artifacts(artifact_dir)
        if not verification["valid"]:
            raise ValueError(f"lane {lane_id!r} failed closed-world verification: {verification}")
        lane_manifest = _load_json(artifact_dir / "study_manifest.json", label="lane manifest")
        lane_cases = _load_json(artifact_dir / "case_results.json", label="lane cases")
        if lane_manifest.get("artifact_role") != E001_REPRESENTATION_LANE_SCHEMA:
            raise ValueError(f"lane {lane_id!r} is not a supported E001 representation lane")
        representation_family = _required_text(
            "representation_family", lane_manifest.get("representation_family")
        )
        model_id = lane_manifest.get("model_id")
        model_revision = lane_manifest.get("model_revision")
        if representation_family == "foundation_model":
            model_id = _required_text("model_id", model_id)
            model_revision = _required_text("model_revision", model_revision)
        elif model_id is not None:
            model_id = _required_text("model_id", model_id)
            model_revision = _required_text("model_revision", model_revision)
        source_representation_id = _required_text(
            "representation_id", lane_manifest.get("representation_id")
        )
        expected_scientific = _required_text(
            "scientific_fingerprint", lane_manifest.get("scientific_fingerprint")
        )
        raw_cases = lane_cases.get("cases")
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ValueError(f"lane {lane_id!r} contains no case results")
        for case in raw_cases:
            if not isinstance(case, Mapping):
                raise ValueError("lane case must be an object")
            if case.get("representation_id") != source_representation_id:
                raise ValueError("case representation_id does not match lane manifest")
            representation_sha = _required_text(
                "representation_sha256", case.get("representation_sha256")
            )
            source_fingerprint = _required_text("study_fingerprint", case.get("study_fingerprint"))
            authority = case.get("authority")
            if not isinstance(authority, Mapping):
                raise ValueError("lane case lacks authority")
            case_id = _required_text("authority.case_id", authority.get("case_id"))
            authority_fingerprint = _required_text(
                "authority.authority_fingerprint", authority.get("authority_fingerprint")
            )
            case_metadata = authority.get("case_metadata")
            if not isinstance(case_metadata, Mapping):
                raise ValueError("lane case authority lacks case_metadata")
            participant = _required_text(
                f"case_metadata[{participant_key!r}]", case_metadata.get(participant_key)
            )
            held_out = authority.get("held_out_values")
            if not isinstance(held_out, list) or not held_out:
                raise ValueError("lane case authority lacks held_out_values")
            occasion = _required_text(
                "occasion",
                case_metadata.get("held_out_session", held_out[0]),
            )
            rows = case.get("rows")
            if not isinstance(rows, list) or not rows:
                raise ValueError("lane case contains no E001 rows")
            for row in rows:
                if not isinstance(row, Mapping):
                    raise ValueError("E001 row must be an object")
                benchmark = row.get("benchmark")
                if not isinstance(benchmark, Mapping):
                    raise ValueError("E001 row lacks benchmark")
                information_novel = _strict_bool(
                    "density_information_novel", benchmark.get("density_information_novel")
                )
                observations.append(
                    ConfirmatoryRepresentationObservation(
                        participant_id=participant,
                        occasion_id=occasion,
                        case_id=case_id,
                        calibration_per_class=int(row.get("calibration_per_class", -1)),
                        representation_id=lane_id,
                        representation_family=representation_family,
                        authority_fingerprint=authority_fingerprint,
                        representation_sha256=representation_sha,
                        source_fingerprint=source_fingerprint,
                        candidate_metric=_metric_balanced_accuracy(benchmark, "density"),
                        primary_control_metric=_metric_balanced_accuracy(
                            benchmark, policy.primary_classical_control
                        ),
                        ablated_metric=_metric_balanced_accuracy(
                            benchmark, "offdiagonal_ablation"
                        ),
                        information_novel=information_novel,
                        model_id=model_id,
                        model_revision=model_revision,
                    )
                )
        expected = raw_lane.get("scientific_fingerprint")
        if expected is not None and str(expected) != expected_scientific:
            raise ValueError(f"lane {lane_id!r} scientific_fingerprint mismatch")
    if policy.reference_representation_id not in lane_ids:
        raise ValueError("reference_representation_id is not a declared lane")
    return study_id, mechanism_id, policy, tuple(observations), dict(manifest)


def build_confirmatory_representation_bundle(manifest_path: str | Path) -> dict[str, Any]:
    study_id, mechanism_id, policy, observations, manifest = load_confirmatory_representation_manifest(
        manifest_path
    )
    result = evaluate_confirmatory_representation(
        observations,
        study_id=study_id,
        mechanism_id=mechanism_id,
        policy=policy,
    )
    profile = build_confirmatory_representation_profile(result)
    payload = {
        "schema_version": 2,
        "artifact_role": "bmrb_confirmatory_representation_bundle",
        "benchmark": CONFIRMATORY_REPRESENTATION_BENCHMARK,
        "study_id": study_id,
        "mechanism_id": mechanism_id,
        "representation_evidence": result.to_mapping(),
        "mechanism_profile": profile.to_mapping(),
        "manifest_metadata": manifest.get("metadata", {}),
        "claim_boundary": [
            "primary calibration budget is fixed before evaluation",
            "primary classical control is fixed before evaluation",
            "external preregistration evidence is required for confirmatory promotion",
            "participant is the inference unit",
            "other calibration budgets are secondary/descriptive",
            "cross-representation recurrence is not causal necessity",
            "physical-quantum claims require independent witness evidence",
        ],
    }
    payload["source_fingerprint"] = canonical_scientific_fingerprint(
        "quantumbci.bmrb-confirmatory-representation.v2", payload
    )
    return payload


def write_confirmatory_representation_bundle(
    manifest_path: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    payload = build_confirmatory_representation_bundle(manifest_path)
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"confirmatory output already contains files: {output}")
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "bmrb_confirmatory_representation.json"
    report_path = output / "report.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence = payload["representation_evidence"]
    profile = payload["mechanism_profile"]
    lines = [
        "# BMRB Confirmatory Representation",
        "",
        f"- Study: `{payload['study_id']}`",
        f"- Mechanism: `{payload['mechanism_id']}`",
        f"- Primary calibration/class: `{evidence['primary_calibration_per_class']}`",
        f"- Confirmatory authority: **{evidence['confirmatory_authority']}**",
        f"- Scientific criteria passed: **{evidence['scientific_criteria_passed']}**",
        f"- Promotion eligible: **{evidence['promotion_eligible']}**",
        f"- Promotion ceiling: `{profile['promotion_ceiling']}`",
        f"- First failing gate: `{profile['first_failing_gate']}`",
        "",
        "## Lane estimates",
        "",
        "| Lane | Candidate mean [bootstrap CI] | Sign-flip p | Ablation mean [bootstrap CI] |",
        "| --- | --- | ---: | --- |",
    ]
    for lane in evidence["lanes"]:
        candidate = lane["candidate_advantage"]
        ablation = lane["ablation_necessity"]
        lines.append(
            f"| {lane['representation_id']} | {candidate['observed_mean']:.4f} "
            f"[{candidate['bootstrap_ci_lower']:.4f}, {candidate['bootstrap_ci_upper']:.4f}] | "
            f"{candidate['sign_flip_pvalue_two_sided']:.4g} | "
            f"{ablation['observed_mean']:.4f} [{ablation['bootstrap_ci_lower']:.4f}, "
            f"{ablation['bootstrap_ci_upper']:.4f}] |"
        )
    lines.extend(
        [
            "",
            "Calibration-frontier values are secondary descriptive evidence and are not averaged "
            "into the primary estimand.",
            "",
            "Sign-flip p-values and bootstrap intervals are reported for uncertainty; promotion "
            "still follows the externally registered decision rule, not a hidden p-value cutoff.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, report_path
