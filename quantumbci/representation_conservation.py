"""Cross-representation mechanism conservation for BMRB-Representation.

The representation benchmark asks whether the same participant-level mechanism effect
recurs when the frozen representation changes. Conservation and information novelty
are intentionally separate. A mathematically equivalent classical representation may
show perfect conservation while still failing the adversary-survival gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .claims import ClaimClass
from .recapitulation import (
    EvidenceGate,
    EvidenceTier,
    GateStatus,
    MechanismNecessityProfile,
    bmrb_representation_signature,
)

REPRESENTATION_CONSERVATION_METHOD_ID = "participant_balanced_cross_representation_v1"


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


def _fingerprint(domain: bytes, value: Any) -> str:
    return sha256(domain + _canonical_json(value).encode("utf-8")).hexdigest()


def _sign(value: float, *, atol: float = 1e-12) -> int:
    if value > atol:
        return 1
    if value < -atol:
        return -1
    return 0


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if len(a) != len(b):
        raise ValueError("correlation vectors must align")
    if len(a) < 3:
        return None
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return None
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else None


@dataclass(frozen=True)
class RepresentationEffectCase:
    """One authority-bound E001-like mechanism observation in one representation lane."""

    participant_id: str
    occasion_id: str
    case_id: str
    calibration_per_class: int
    representation_id: str
    representation_family: str
    source_representation_id: str
    mechanism_id: str
    authority_fingerprint: str
    representation_sha256: str
    source_fingerprint: str
    candidate_metric: float
    strongest_control_metric: float
    ablated_metric: float
    higher_is_better: bool
    information_novel: bool
    model_id: str | None = None
    model_revision: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "participant_id",
            "occasion_id",
            "case_id",
            "representation_id",
            "representation_family",
            "source_representation_id",
            "mechanism_id",
            "authority_fingerprint",
            "representation_sha256",
            "source_fingerprint",
        ):
            _required_text(name, getattr(self, name))
        if int(self.calibration_per_class) < 0:
            raise ValueError("calibration_per_class must be non-negative")
        object.__setattr__(self, "calibration_per_class", int(self.calibration_per_class))
        object.__setattr__(self, "candidate_metric", _finite("candidate_metric", self.candidate_metric))
        object.__setattr__(
            self,
            "strongest_control_metric",
            _finite("strongest_control_metric", self.strongest_control_metric),
        )
        object.__setattr__(self, "ablated_metric", _finite("ablated_metric", self.ablated_metric))
        object.__setattr__(self, "higher_is_better", _strict_bool("higher_is_better", self.higher_is_better))
        object.__setattr__(self, "information_novel", _strict_bool("information_novel", self.information_novel))
        if self.model_id is not None:
            object.__setattr__(self, "model_id", _required_text("model_id", self.model_id))
        if self.model_revision is not None:
            object.__setattr__(self, "model_revision", _required_text("model_revision", self.model_revision))

    @property
    def key(self) -> tuple[str, str, str, int]:
        return (
            self.participant_id,
            self.occasion_id,
            self.case_id,
            self.calibration_per_class,
        )

    @property
    def orientation(self) -> float:
        return 1.0 if self.higher_is_better else -1.0

    @property
    def candidate_advantage(self) -> float:
        return self.orientation * (self.candidate_metric - self.strongest_control_metric)

    @property
    def ablation_necessity(self) -> float:
        return self.orientation * (self.candidate_metric - self.ablated_metric)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "occasion_id": self.occasion_id,
            "case_id": self.case_id,
            "calibration_per_class": self.calibration_per_class,
            "representation_id": self.representation_id,
            "representation_family": self.representation_family,
            "source_representation_id": self.source_representation_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "mechanism_id": self.mechanism_id,
            "authority_fingerprint": self.authority_fingerprint,
            "representation_sha256": self.representation_sha256,
            "source_fingerprint": self.source_fingerprint,
            "candidate_metric": self.candidate_metric,
            "strongest_control_metric": self.strongest_control_metric,
            "ablated_metric": self.ablated_metric,
            "higher_is_better": self.higher_is_better,
            "information_novel": self.information_novel,
            "candidate_advantage": self.candidate_advantage,
            "ablation_necessity": self.ablation_necessity,
        }


@dataclass(frozen=True)
class RepresentationConservationPolicy:
    """Preregisterable promotion policy for one cross-representation mechanism study."""

    policy_id: str
    preregistered: bool
    reference_representation_id: str
    min_participants: int = 3
    min_representations: int = 2
    min_representation_families: int = 2
    min_reference_positive_fraction: float = 0.80
    min_all_lane_positive_fraction: float = 0.80
    min_all_lane_ablation_positive_fraction: float = 0.80
    min_direction_match_fraction: float = 0.80
    min_ablation_direction_match_fraction: float = 0.80
    min_information_novel_representation_fraction: float = 1.0

    def __post_init__(self) -> None:
        _required_text("policy_id", self.policy_id)
        _required_text("reference_representation_id", self.reference_representation_id)
        object.__setattr__(self, "preregistered", _strict_bool("preregistered", self.preregistered))
        if int(self.min_participants) < 2:
            raise ValueError("min_participants must be at least 2")
        if int(self.min_representations) < 2:
            raise ValueError("min_representations must be at least 2")
        if int(self.min_representation_families) < 1:
            raise ValueError("min_representation_families must be positive")
        for name in (
            "min_reference_positive_fraction",
            "min_all_lane_positive_fraction",
            "min_all_lane_ablation_positive_fraction",
            "min_direction_match_fraction",
            "min_ablation_direction_match_fraction",
            "min_information_novel_representation_fraction",
        ):
            object.__setattr__(self, name, _fraction(name, getattr(self, name)))

    @property
    def decision_rule(self) -> str:
        return (
            f"participants>={self.min_participants}; "
            f"representations>={self.min_representations}; "
            f"families>={self.min_representation_families}; "
            f"reference_positive>={self.min_reference_positive_fraction:.3f}; "
            f"all_lane_positive>={self.min_all_lane_positive_fraction:.3f}; "
            f"all_lane_ablation_positive>={self.min_all_lane_ablation_positive_fraction:.3f}; "
            f"direction_match>={self.min_direction_match_fraction:.3f}; "
            f"ablation_direction_match>={self.min_ablation_direction_match_fraction:.3f}; "
            f"information_novel_representation_fraction>="
            f"{self.min_information_novel_representation_fraction:.3f}"
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "preregistered": self.preregistered,
            "reference_representation_id": self.reference_representation_id,
            "min_participants": int(self.min_participants),
            "min_representations": int(self.min_representations),
            "min_representation_families": int(self.min_representation_families),
            "min_reference_positive_fraction": self.min_reference_positive_fraction,
            "min_all_lane_positive_fraction": self.min_all_lane_positive_fraction,
            "min_all_lane_ablation_positive_fraction": self.min_all_lane_ablation_positive_fraction,
            "min_direction_match_fraction": self.min_direction_match_fraction,
            "min_ablation_direction_match_fraction": self.min_ablation_direction_match_fraction,
            "min_information_novel_representation_fraction": self.min_information_novel_representation_fraction,
            "decision_rule": self.decision_rule,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RepresentationConservationPolicy":
        if "preregistered" not in payload:
            raise ValueError("policy.preregistered is required")
        return cls(
            policy_id=_required_text("policy.policy_id", payload.get("policy_id")),
            preregistered=_strict_bool("policy.preregistered", payload.get("preregistered")),
            reference_representation_id=_required_text(
                "policy.reference_representation_id", payload.get("reference_representation_id")
            ),
            min_participants=int(payload.get("min_participants", 3)),
            min_representations=int(payload.get("min_representations", 2)),
            min_representation_families=int(payload.get("min_representation_families", 2)),
            min_reference_positive_fraction=float(payload.get("min_reference_positive_fraction", 0.80)),
            min_all_lane_positive_fraction=float(payload.get("min_all_lane_positive_fraction", 0.80)),
            min_all_lane_ablation_positive_fraction=float(
                payload.get("min_all_lane_ablation_positive_fraction", 0.80)
            ),
            min_direction_match_fraction=float(payload.get("min_direction_match_fraction", 0.80)),
            min_ablation_direction_match_fraction=float(
                payload.get("min_ablation_direction_match_fraction", 0.80)
            ),
            min_information_novel_representation_fraction=float(
                payload.get("min_information_novel_representation_fraction", 1.0)
            ),
        )


@dataclass(frozen=True)
class RepresentationLaneSummary:
    representation_id: str
    representation_family: str
    source_representation_id: str
    model_id: str | None
    model_revision: str | None
    participant_count: int
    case_count: int
    mean_candidate_advantage: float
    mean_ablation_necessity: float
    participant_positive_fraction: float
    participant_ablation_positive_fraction: float
    information_novel_fraction: float

    @property
    def information_novel(self) -> bool:
        return self.information_novel_fraction >= 1.0 - 1e-12

    def to_mapping(self) -> dict[str, Any]:
        return {
            "representation_id": self.representation_id,
            "representation_family": self.representation_family,
            "source_representation_id": self.source_representation_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "participant_count": self.participant_count,
            "case_count": self.case_count,
            "mean_candidate_advantage": self.mean_candidate_advantage,
            "mean_ablation_necessity": self.mean_ablation_necessity,
            "participant_positive_fraction": self.participant_positive_fraction,
            "participant_ablation_positive_fraction": self.participant_ablation_positive_fraction,
            "information_novel_fraction": self.information_novel_fraction,
            "information_novel": self.information_novel,
        }


@dataclass(frozen=True)
class RepresentationConservationResult:
    mechanism_id: str
    policy: RepresentationConservationPolicy
    lanes: tuple[RepresentationLaneSummary, ...]
    cases: tuple[RepresentationEffectCase, ...]
    participant_count: int
    representation_count: int
    representation_family_count: int
    reference_positive_fraction: float
    all_lane_positive_fraction: float
    all_lane_ablation_positive_fraction: float
    direction_match_fraction: float
    ablation_direction_match_fraction: float
    information_novel_representation_fraction: float
    pairwise_reference_correlations: Mapping[str, float | None]
    conservation_criteria_passed: bool
    adversary_survival_passed: bool
    conservation_reasons: tuple[str, ...]
    adversary_reasons: tuple[str, ...]
    source_fingerprint: str

    @property
    def promotion_eligible(self) -> bool:
        return bool(
            self.policy.preregistered
            and self.conservation_criteria_passed
            and self.adversary_survival_passed
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_role": "representation_conservation_evidence",
            "method_id": REPRESENTATION_CONSERVATION_METHOD_ID,
            "mechanism_id": self.mechanism_id,
            "policy": self.policy.to_mapping(),
            "participant_count": self.participant_count,
            "representation_count": self.representation_count,
            "representation_family_count": self.representation_family_count,
            "reference_positive_fraction": self.reference_positive_fraction,
            "all_lane_positive_fraction": self.all_lane_positive_fraction,
            "all_lane_ablation_positive_fraction": self.all_lane_ablation_positive_fraction,
            "direction_match_fraction": self.direction_match_fraction,
            "ablation_direction_match_fraction": self.ablation_direction_match_fraction,
            "information_novel_representation_fraction": self.information_novel_representation_fraction,
            "pairwise_reference_correlations": dict(self.pairwise_reference_correlations),
            "conservation_criteria_passed": self.conservation_criteria_passed,
            "adversary_survival_passed": self.adversary_survival_passed,
            "promotion_eligible": self.promotion_eligible,
            "conservation_reasons": list(self.conservation_reasons),
            "adversary_reasons": list(self.adversary_reasons),
            "lanes": [lane.to_mapping() for lane in self.lanes],
            "cases": [case.to_mapping() for case in self.cases],
            "source_fingerprint": self.source_fingerprint,
            "physical_quantum_promotion_eligible": False,
            "interpretation": (
                "Cross-representation conservation shows whether a mechanism-like contrast recurs "
                "under frozen representation changes. It does not establish information novelty. "
                "Adversary survival is evaluated separately and physical-quantum claims remain "
                "outside this benchmark."
            ),
        }


def _participant_means(
    cases: Sequence[RepresentationEffectCase],
    *,
    field: str,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for case in cases:
        grouped.setdefault(case.participant_id, []).append(float(getattr(case, field)))
    return {
        participant: float(np.mean(values))
        for participant, values in sorted(grouped.items())
    }


def _lane_summary(cases: Sequence[RepresentationEffectCase]) -> RepresentationLaneSummary:
    if not cases:
        raise ValueError("representation lane must contain cases")
    first = cases[0]
    for case in cases[1:]:
        for name in (
            "representation_id",
            "representation_family",
            "source_representation_id",
            "model_id",
            "model_revision",
            "mechanism_id",
        ):
            if getattr(case, name) != getattr(first, name):
                raise ValueError(
                    f"representation lane {first.representation_id!r} mixes {name} values"
                )
    effect = _participant_means(cases, field="candidate_advantage")
    ablation = _participant_means(cases, field="ablation_necessity")
    by_participant_novel: dict[str, list[float]] = {}
    for case in cases:
        by_participant_novel.setdefault(case.participant_id, []).append(float(case.information_novel))
    novelty = [float(np.mean(values)) for values in by_participant_novel.values()]
    return RepresentationLaneSummary(
        representation_id=first.representation_id,
        representation_family=first.representation_family,
        source_representation_id=first.source_representation_id,
        model_id=first.model_id,
        model_revision=first.model_revision,
        participant_count=len(effect),
        case_count=len(cases),
        mean_candidate_advantage=float(np.mean(list(effect.values()))),
        mean_ablation_necessity=float(np.mean(list(ablation.values()))),
        participant_positive_fraction=float(np.mean([value > 1e-12 for value in effect.values()])),
        participant_ablation_positive_fraction=float(
            np.mean([value > 1e-12 for value in ablation.values()])
        ),
        information_novel_fraction=float(np.mean(novelty)),
    )


def evaluate_representation_conservation(
    cases: Iterable[RepresentationEffectCase],
    *,
    policy: RepresentationConservationPolicy,
) -> RepresentationConservationResult:
    """Evaluate exact-paired participant-level conservation across representation lanes.

    BMRB-Representation v1 is deliberately strict: every lane must contain the exact
    same participant/occasion/case/calibration keys, and the neurOS authority
    fingerprint for each key must be identical across lanes. Missing-pair imputation or
    intersection-only analysis would change the estimand and therefore fails closed.
    """

    materialized = tuple(cases)
    if not materialized:
        raise ValueError("representation conservation requires case evidence")
    mechanisms = {case.mechanism_id for case in materialized}
    if len(mechanisms) != 1:
        raise ValueError("representation conservation cases must share one mechanism_id")

    by_lane: dict[str, list[RepresentationEffectCase]] = {}
    for case in materialized:
        by_lane.setdefault(case.representation_id, []).append(case)
    if policy.reference_representation_id not in by_lane:
        raise ValueError("reference_representation_id is absent from representation cases")

    lane_maps: dict[str, dict[tuple[str, str, str, int], RepresentationEffectCase]] = {}
    for lane_id, lane_cases in sorted(by_lane.items()):
        mapped: dict[tuple[str, str, str, int], RepresentationEffectCase] = {}
        for case in lane_cases:
            if case.key in mapped:
                raise ValueError(
                    f"duplicate representation observation in lane {lane_id!r}: {case.key}"
                )
            mapped[case.key] = case
        lane_maps[lane_id] = mapped

    reference_map = lane_maps[policy.reference_representation_id]
    reference_keys = set(reference_map)
    for lane_id, mapping in lane_maps.items():
        keys = set(mapping)
        if keys != reference_keys:
            missing = sorted(reference_keys - keys)
            extra = sorted(keys - reference_keys)
            raise ValueError(
                "representation lanes must be exactly paired; "
                f"lane={lane_id!r} missing={missing} extra={extra}"
            )
        for key in sorted(reference_keys):
            if mapping[key].authority_fingerprint != reference_map[key].authority_fingerprint:
                raise ValueError(
                    "authority fingerprint mismatch across representation lanes for "
                    f"key={key}, lane={lane_id!r}"
                )

    lanes = tuple(_lane_summary(by_lane[lane_id]) for lane_id in sorted(by_lane))
    families = {lane.representation_family for lane in lanes}
    participants = sorted({case.participant_id for case in materialized})

    lane_effects = {
        lane_id: _participant_means(lane_cases, field="candidate_advantage")
        for lane_id, lane_cases in by_lane.items()
    }
    lane_ablations = {
        lane_id: _participant_means(lane_cases, field="ablation_necessity")
        for lane_id, lane_cases in by_lane.items()
    }
    ref_effect = lane_effects[policy.reference_representation_id]
    ref_ablation = lane_ablations[policy.reference_representation_id]

    reference_positive = float(np.mean([ref_effect[p] > 1e-12 for p in participants]))
    all_effect_values = [lane_effects[lane][p] for lane in sorted(lane_effects) for p in participants]
    all_ablation_values = [
        lane_ablations[lane][p] for lane in sorted(lane_ablations) for p in participants
    ]
    all_lane_positive = float(np.mean([value > 1e-12 for value in all_effect_values]))
    all_lane_ablation_positive = float(
        np.mean([value > 1e-12 for value in all_ablation_values])
    )

    direction_matches: list[bool] = []
    ablation_matches: list[bool] = []
    correlations: dict[str, float | None] = {}
    for lane_id in sorted(lane_effects):
        if lane_id == policy.reference_representation_id:
            continue
        direction_matches.extend(
            _sign(lane_effects[lane_id][participant]) == _sign(ref_effect[participant])
            for participant in participants
        )
        ablation_matches.extend(
            _sign(lane_ablations[lane_id][participant]) == _sign(ref_ablation[participant])
            for participant in participants
        )
        correlations[lane_id] = _pearson(
            [ref_effect[p] for p in participants],
            [lane_effects[lane_id][p] for p in participants],
        )
    direction_match = float(np.mean(direction_matches)) if direction_matches else 1.0
    ablation_match = float(np.mean(ablation_matches)) if ablation_matches else 1.0
    novel_representation_fraction = float(
        np.mean([lane.information_novel for lane in lanes])
    )

    conservation_reasons: list[str] = []
    if len(participants) < policy.min_participants:
        conservation_reasons.append(
            f"independent participants {len(participants)} < {policy.min_participants}"
        )
    if len(lanes) < policy.min_representations:
        conservation_reasons.append(
            f"representations {len(lanes)} < {policy.min_representations}"
        )
    if len(families) < policy.min_representation_families:
        conservation_reasons.append(
            f"representation families {len(families)} < {policy.min_representation_families}"
        )
    if reference_positive < policy.min_reference_positive_fraction:
        conservation_reasons.append(
            f"reference positive fraction {reference_positive:.3f} < "
            f"{policy.min_reference_positive_fraction:.3f}"
        )
    if all_lane_positive < policy.min_all_lane_positive_fraction:
        conservation_reasons.append(
            f"all-lane positive fraction {all_lane_positive:.3f} < "
            f"{policy.min_all_lane_positive_fraction:.3f}"
        )
    if all_lane_ablation_positive < policy.min_all_lane_ablation_positive_fraction:
        conservation_reasons.append(
            f"all-lane ablation positive fraction {all_lane_ablation_positive:.3f} < "
            f"{policy.min_all_lane_ablation_positive_fraction:.3f}"
        )
    if direction_match < policy.min_direction_match_fraction:
        conservation_reasons.append(
            f"direction match {direction_match:.3f} < {policy.min_direction_match_fraction:.3f}"
        )
    if ablation_match < policy.min_ablation_direction_match_fraction:
        conservation_reasons.append(
            f"ablation direction match {ablation_match:.3f} < "
            f"{policy.min_ablation_direction_match_fraction:.3f}"
        )

    adversary_reasons: list[str] = []
    if novel_representation_fraction < policy.min_information_novel_representation_fraction:
        adversary_reasons.append(
            "information-novel representation fraction "
            f"{novel_representation_fraction:.3f} < "
            f"{policy.min_information_novel_representation_fraction:.3f}"
        )

    source_payload = {
        "policy": policy.to_mapping(),
        "cases": [
            case.to_mapping()
            for case in sorted(
                materialized,
                key=lambda item: (
                    item.representation_id,
                    item.participant_id,
                    item.occasion_id,
                    item.case_id,
                    item.calibration_per_class,
                ),
            )
        ],
    }
    return RepresentationConservationResult(
        mechanism_id=next(iter(mechanisms)),
        policy=policy,
        lanes=lanes,
        cases=materialized,
        participant_count=len(participants),
        representation_count=len(lanes),
        representation_family_count=len(families),
        reference_positive_fraction=reference_positive,
        all_lane_positive_fraction=all_lane_positive,
        all_lane_ablation_positive_fraction=all_lane_ablation_positive,
        direction_match_fraction=direction_match,
        ablation_direction_match_fraction=ablation_match,
        information_novel_representation_fraction=novel_representation_fraction,
        pairwise_reference_correlations=correlations,
        conservation_criteria_passed=not conservation_reasons,
        adversary_survival_passed=not adversary_reasons,
        conservation_reasons=tuple(conservation_reasons),
        adversary_reasons=tuple(adversary_reasons),
        source_fingerprint=_fingerprint(
            b"quantumbci.representation-conservation.v1\0", source_payload
        ),
    )


def build_representation_necessity_profile(
    result: RepresentationConservationResult,
) -> MechanismNecessityProfile:
    """Map cross-representation evidence onto the monotonic BMRB evidence ladder."""

    policy = result.policy
    lane_map = {lane.representation_id: lane for lane in result.lanes}
    reference = lane_map[policy.reference_representation_id]
    predictive_ok = (
        result.reference_positive_fraction >= policy.min_reference_positive_fraction
        and result.all_lane_positive_fraction >= policy.min_all_lane_positive_fraction
    )
    stability_ok = (
        result.direction_match_fraction >= policy.min_direction_match_fraction
        and result.ablation_direction_match_fraction
        >= policy.min_ablation_direction_match_fraction
    )
    repeated_ok = (
        result.participant_count >= policy.min_participants
        and result.representation_count >= policy.min_representations
        and result.representation_family_count >= policy.min_representation_families
        and result.all_lane_ablation_positive_fraction
        >= policy.min_all_lane_ablation_positive_fraction
    )

    descriptive = EvidenceGate(
        id="paired_representation_authority",
        tier=EvidenceTier.DESCRIPTIVE,
        status=GateStatus.PASS,
        summary=(
            "Every representation lane contains the exact same participant/occasion/case/"
            "calibration keys and matching authority fingerprints."
        ),
        evidence_ref=result.source_fingerprint,
        threshold="exact key-set equality and authority-fingerprint equality across lanes",
    )

    if not predictive_ok:
        predictive_status = GateStatus.FAIL
        predictive_summary = (
            "The candidate effect is not consistently favorable in the reference and paired "
            "representation lanes under the declared thresholds."
        )
        predictive_threshold = None
    elif policy.preregistered:
        predictive_status = GateStatus.PASS
        predictive_summary = "Preregistered held-out candidate-vs-control effect thresholds passed."
        predictive_threshold = (
            f"reference_positive>={policy.min_reference_positive_fraction:.3f}; "
            f"all_lane_positive>={policy.min_all_lane_positive_fraction:.3f}"
        )
    else:
        predictive_status = GateStatus.CHARACTERIZED
        predictive_summary = (
            "Held-out candidate effects were favorable, but the cross-representation policy was "
            "not preregistered."
        )
        predictive_threshold = None
    predictive = EvidenceGate(
        id="held_out_representation_effect",
        tier=EvidenceTier.PREDICTIVE,
        status=predictive_status,
        summary=predictive_summary,
        evidence_ref=result.source_fingerprint,
        metric="participant_balanced_candidate_advantage",
        value=float(reference.mean_candidate_advantage),
        threshold=predictive_threshold,
    )

    upstream_pass = predictive.status == GateStatus.PASS
    if not result.adversary_survival_passed:
        adversary_status = GateStatus.FAIL
        adversary_summary = (
            "One or more representation lanes remain information-equivalent to their strongest "
            "matched classical control. Conservation therefore cannot establish novel "
            "representation information."
        )
        adversary_threshold = None
    elif upstream_pass and policy.preregistered:
        adversary_status = GateStatus.PASS
        adversary_summary = "Preregistered information-novel representation-lane threshold passed."
        adversary_threshold = (
            "information_novel_representation_fraction>="
            f"{policy.min_information_novel_representation_fraction:.3f}"
        )
    else:
        adversary_status = GateStatus.CHARACTERIZED
        adversary_summary = (
            "Information novelty was measured and satisfied, but an earlier BMRB tier or "
            "preregistration requirement blocks promotion."
        )
        adversary_threshold = None
    adversary = EvidenceGate(
        id="matched_representation_adversaries",
        tier=EvidenceTier.ADVERSARY_SURVIVING,
        status=adversary_status,
        summary=adversary_summary,
        evidence_ref=result.source_fingerprint,
        metric="information_novel_representation_fraction",
        value=float(result.information_novel_representation_fraction),
        threshold=adversary_threshold,
    )

    upstream_pass = upstream_pass and adversary.status == GateStatus.PASS
    if not stability_ok:
        stability_status = GateStatus.FAIL
        stability_summary = (
            "The candidate/ablation effect direction is not sufficiently conserved across "
            "representation changes."
        )
        stability_threshold = None
    elif upstream_pass and policy.preregistered:
        stability_status = GateStatus.PASS
        stability_summary = "Preregistered cross-representation direction-conservation gates passed."
        stability_threshold = (
            f"direction_match>={policy.min_direction_match_fraction:.3f}; "
            f"ablation_direction_match>={policy.min_ablation_direction_match_fraction:.3f}"
        )
    else:
        stability_status = GateStatus.CHARACTERIZED
        stability_summary = (
            "Cross-representation direction conservation was measured, but promotion remains "
            "blocked by an earlier tier or retrospective policy."
        )
        stability_threshold = None
    stability = EvidenceGate(
        id="cross_representation_stability",
        tier=EvidenceTier.SOURCE_STABILITY,
        status=stability_status,
        summary=stability_summary,
        evidence_ref=result.source_fingerprint,
        metric="direction_and_ablation_match_fraction",
        value=float(min(result.direction_match_fraction, result.ablation_direction_match_fraction)),
        threshold=stability_threshold,
    )

    upstream_pass = upstream_pass and stability.status == GateStatus.PASS
    if not repeated_ok:
        repeated_status = GateStatus.FAIL
        repeated_summary = (
            "Independent-participant, representation-count, family-count, or ablation-necessity "
            "replication requirements were not met."
        )
        repeated_threshold = None
    elif upstream_pass and policy.preregistered:
        repeated_status = GateStatus.PASS
        repeated_summary = (
            "The candidate and its ablation consequence replicated across independent participants "
            "and the declared representation families."
        )
        repeated_threshold = (
            f"participants>={policy.min_participants}; representations>={policy.min_representations}; "
            f"families>={policy.min_representation_families}; all_lane_ablation_positive>="
            f"{policy.min_all_lane_ablation_positive_fraction:.3f}"
        )
    else:
        repeated_status = GateStatus.CHARACTERIZED
        repeated_summary = (
            "Participant-level cross-representation replication was measured, but an earlier "
            "BMRB tier or preregistration requirement blocks promotion."
        )
        repeated_threshold = None
    repeated = EvidenceGate(
        id="cross_representation_replication",
        tier=EvidenceTier.REPEATED_CASE,
        status=repeated_status,
        summary=repeated_summary,
        evidence_ref=result.source_fingerprint,
        metric="all_lane_ablation_positive_fraction",
        value=float(result.all_lane_ablation_positive_fraction),
        threshold=repeated_threshold,
    )

    causal = EvidenceGate(
        id="causal_intervention_and_ablation",
        tier=EvidenceTier.CAUSAL_MECHANISTIC,
        status=GateStatus.NOT_RUN,
        summary=(
            "Cross-representation recurrence is not causal necessity. Intervention, dose-response, "
            "faithfulness and matched recovery evidence remain a separate promotion layer."
        ),
    )
    physical = EvidenceGate(
        id="physical_quantum_witness",
        tier=EvidenceTier.PHYSICAL_QUANTUM,
        status=GateStatus.NOT_APPLICABLE,
        summary=(
            "BMRB-Representation evaluates quantum-inspired computational structure only. "
            "Physical-quantum claims require an independent witness/substrate protocol."
        ),
    )
    return MechanismNecessityProfile(
        mechanism_id=result.mechanism_id,
        claim_class=ClaimClass.QUANTUM_INSPIRED,
        signature=bmrb_representation_signature(),
        gates=(descriptive, predictive, adversary, stability, repeated, causal, physical),
        metadata={
            "representation_conservation_source_fingerprint": result.source_fingerprint,
            "reference_representation_id": policy.reference_representation_id,
            "representation_count": result.representation_count,
            "representation_family_count": result.representation_family_count,
            "participant_count": result.participant_count,
        },
    )
