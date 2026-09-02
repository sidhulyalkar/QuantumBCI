"""Outcome-blind decision authority for confirmatory Kumar2024 E001 studies.

This module freezes how a future confirmatory E001 result may be interpreted.  It
does not execute E001, load result bundles, or choose scientific thresholds.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from .kumar2024_authority_artifacts import load_kumar2024_authority_capsule
from .preregistration import PreregistrationEvidence

KUMAR2024_E001_DECISION_METHOD = "kumar2024_e001_confirmatory_decision_plan_v1"
KUMAR2024_E001_SEAL_ROLE = "kumar2024_e001_confirmatory_preregistration_seal_v1"
KUMAR2024_E001_SCHEMA_VERSION = 1
KUMAR2024_E001_DOMAIN = b"quantumbci.kumar2024-e001-decision.v1\0"
KUMAR2024_E001_SEAL_DOMAIN = b"quantumbci.kumar2024-e001-preregistration-seal.v1\0"

KUMAR2024_AUTHORITY_CAPSULE_FINGERPRINT = (
    "1013358b419436a3a9592c8a48eec2372701b1977e7ced06f4c25cfd4ebae29d"
)
KUMAR2024_COHORT_AUTHORITY_FINGERPRINT = (
    "36cdfdf42e5ac375999d4defa02554cf4d2d04472ed6c06a08c389b5ad02b81c"
)
KUMAR2024_RAW_DATASET_FINGERPRINT = (
    "c91c6dca34be880e688359e210686c1823461ad93923f71e947bb3d0725d6c8b"
)
KUMAR2024_SCIENCE_SOURCE_SHA = "681ea12c436fce121ba74de6f877a8267e94dd3f"
KUMAR2024_NEUROS_SOURCE_SHA = "ffa28ed552dc75158b673fdcd70729b1c9c69b47"
KUMAR2024_SUBJECTS = tuple(range(1, 19))
KUMAR2024_PROTOCOL_GROUPS = {
    "GR": tuple(range(1, 10)),
    "PAR": tuple(range(10, 19)),
}
KUMAR2024_HELD_OUT_SESSION = "5"
KUMAR2024_DATASET_ID = "moabb-kumar2024"
KUMAR2024_EVALUATION_FRACTION = 0.5
KUMAR2024_MIN_MAX_BUDGET_PER_CLASS = 14

E001_CANDIDATE = "density"
E001_EXACT_EQUIVALENCE_CONTROL = "normalized_covariance"
E001_CLASSICAL_CONTROLS = (
    "normalized_covariance",
    "covariance",
    "log_covariance",
    "bilinear_second_moment",
    "pooled_mean_std",
    "pca_flattened",
    "diagonal_density",
)
E001_ABLATION = "offdiagonal_ablation"
E001_CLOSED_FAMILY = (E001_CANDIDATE, *E001_CLASSICAL_CONTROLS, E001_ABLATION)
E001_PRIMARY_ESTIMAND = "participant_mean_density_minus_offdiagonal_ablation"
E001_PRIMARY_STATISTIC = "participant_bootstrap_ci_lower"
E001_BOOTSTRAP_METHOD = "paired_participant_bootstrap_v1"
E001_CONFIDENCE_INTERVAL = "percentile_95_v1"
E001_INFORMATION_NOVELTY_GATE = "density_representation_information_novelty"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fingerprint(domain: bytes, payload: Mapping[str, Any]) -> str:
    return sha256(domain + _canonical_json(payload).encode("utf-8")).hexdigest()


def _required_text(name: str, value: Any) -> str:
    if value is None:
        raise ValueError(f"{name} must not be null")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _required_bool(name: str, value: Any, expected: bool | None = None) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a JSON boolean")
    if expected is not None and value is not expected:
        raise ValueError(f"{name} must be {str(expected).lower()}")
    return value


def _finite(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive_int(name: str, value: Any, *, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if number < minimum or number != value:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return number


def _mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _exact_text_sequence(name: str, value: Any, expected: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be an array")
    observed = tuple(str(item) for item in value)
    if observed != tuple(expected):
        raise ValueError(f"{name} drifted from the frozen v1 authority")
    return observed


@dataclass(frozen=True)
class Kumar2024E001PrimaryCriterion:
    """One explicit primary criterion. No scientific threshold has a default."""

    calibration_per_class: int
    minimum_effect: float
    rationale: str
    estimand: str = E001_PRIMARY_ESTIMAND
    control: str = E001_ABLATION
    statistic: str = E001_PRIMARY_STATISTIC
    comparison: str = "greater_than_or_equal"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "calibration_per_class",
            _positive_int("calibration_per_class", self.calibration_per_class, minimum=0),
        )
        object.__setattr__(
            self, "minimum_effect", _finite("minimum_effect", self.minimum_effect)
        )
        if self.minimum_effect < 0.0:
            raise ValueError("minimum_effect must be non-negative")
        object.__setattr__(self, "rationale", _required_text("rationale", self.rationale))
        if self.estimand != E001_PRIMARY_ESTIMAND:
            raise ValueError("v1 primary estimand must be participant-level density minus ablation")
        if self.control != E001_ABLATION:
            raise ValueError("v1 primary control must be offdiagonal_ablation")
        if self.statistic != E001_PRIMARY_STATISTIC:
            raise ValueError("v1 primary statistic must be the participant bootstrap CI lower bound")
        if self.comparison != "greater_than_or_equal":
            raise ValueError("v1 primary criterion must use a lower-bound comparison")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "estimand": self.estimand,
            "control": self.control,
            "calibration_per_class": self.calibration_per_class,
            "statistic": self.statistic,
            "comparison": self.comparison,
            "minimum_effect": self.minimum_effect,
            "rationale": self.rationale,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Kumar2024E001PrimaryCriterion":
        return cls(
            calibration_per_class=payload.get("calibration_per_class"),
            minimum_effect=payload.get("minimum_effect"),
            rationale=payload.get("rationale"),
            estimand=_required_text("criterion.estimand", payload.get("estimand")),
            control=_required_text("criterion.control", payload.get("control")),
            statistic=_required_text("criterion.statistic", payload.get("statistic")),
            comparison=_required_text("criterion.comparison", payload.get("comparison")),
        )


@dataclass(frozen=True)
class Kumar2024E001BootstrapAuthority:
    n_resamples: int
    seed: int
    method: str = E001_BOOTSTRAP_METHOD
    inference_unit: str = "subject"
    confidence_interval: str = E001_CONFIDENCE_INTERVAL

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "n_resamples", _positive_int("n_resamples", self.n_resamples, minimum=100)
        )
        object.__setattr__(self, "seed", _positive_int("seed", self.seed, minimum=0))
        if self.method != E001_BOOTSTRAP_METHOD:
            raise ValueError("unknown Kumar2024 E001 bootstrap method")
        if self.inference_unit != "subject":
            raise ValueError("Kumar2024 E001 v1 fixes participant/subject as the inference unit")
        if self.confidence_interval != E001_CONFIDENCE_INTERVAL:
            raise ValueError("v1 uses the production percentile 95% bootstrap interval")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "inference_unit": self.inference_unit,
            "n_resamples": self.n_resamples,
            "seed": self.seed,
            "confidence_interval": self.confidence_interval,
            "bootstrap_probability_positive_promotion_authoritative": False,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Kumar2024E001BootstrapAuthority":
        _required_bool(
            "bootstrap_probability_positive_promotion_authoritative",
            payload.get("bootstrap_probability_positive_promotion_authoritative"),
            False,
        )
        return cls(
            n_resamples=payload.get("n_resamples"),
            seed=payload.get("seed"),
            method=_required_text("bootstrap.method", payload.get("method")),
            inference_unit=_required_text(
                "bootstrap.inference_unit", payload.get("inference_unit")
            ),
            confidence_interval=_required_text(
                "bootstrap.confidence_interval", payload.get("confidence_interval")
            ),
        )


@dataclass(frozen=True)
class Kumar2024E001ControlAuthority:
    candidate: str = E001_CANDIDATE
    exact_equivalence_control: str = E001_EXACT_EQUIVALENCE_CONTROL
    classical_controls: tuple[str, ...] = E001_CLASSICAL_CONTROLS
    ablation: str = E001_ABLATION

    def __post_init__(self) -> None:
        if self.candidate != E001_CANDIDATE:
            raise ValueError("v1 candidate must be density")
        if self.exact_equivalence_control != E001_EXACT_EQUIVALENCE_CONTROL:
            raise ValueError("normalized_covariance must remain the exact equivalence control")
        if tuple(self.classical_controls) != E001_CLASSICAL_CONTROLS:
            raise ValueError("classical control family drifted from production E001")
        if self.ablation != E001_ABLATION:
            raise ValueError("v1 ablation must be offdiagonal_ablation")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "exact_equivalence_control": self.exact_equivalence_control,
            "classical_controls": list(self.classical_controls),
            "ablation": self.ablation,
            "closed_family": list(E001_CLOSED_FAMILY),
            "strongest_classical_control_selection": "maximum_balanced_accuracy_within_closed_family",
            "strongest_classical_control_promotion_authoritative": False,
            "multiplicity_method": "one_family_one_primary_v1",
            "confirmatory_primary_hypothesis": "density_vs_offdiagonal_ablation",
            "additional_controls_promotion_authoritative": False,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Kumar2024E001ControlAuthority":
        _exact_text_sequence(
            "control.classical_controls",
            payload.get("classical_controls"),
            E001_CLASSICAL_CONTROLS,
        )
        _exact_text_sequence(
            "control.closed_family", payload.get("closed_family"), E001_CLOSED_FAMILY
        )
        if payload.get("strongest_classical_control_selection") != (
            "maximum_balanced_accuracy_within_closed_family"
        ):
            raise ValueError("strongest classical-control selection drifted")
        _required_bool(
            "strongest_classical_control_promotion_authoritative",
            payload.get("strongest_classical_control_promotion_authoritative"),
            False,
        )
        if payload.get("multiplicity_method") != "one_family_one_primary_v1":
            raise ValueError("Kumar2024 E001 multiplicity authority drifted")
        if payload.get("confirmatory_primary_hypothesis") != "density_vs_offdiagonal_ablation":
            raise ValueError("Kumar2024 E001 primary hypothesis drifted")
        _required_bool(
            "additional_controls_promotion_authoritative",
            payload.get("additional_controls_promotion_authoritative"),
            False,
        )
        return cls(
            candidate=_required_text("control.candidate", payload.get("candidate")),
            exact_equivalence_control=_required_text(
                "control.exact_equivalence_control",
                payload.get("exact_equivalence_control"),
            ),
            classical_controls=tuple(E001_CLASSICAL_CONTROLS),
            ablation=_required_text("control.ablation", payload.get("ablation")),
        )


@dataclass(frozen=True)
class Kumar2024E001EvidenceHandlingAuthority:
    complete_cohort_required: bool = True
    no_silent_intersection: bool = True
    invalid_evidence_is_scientific_null: bool = False
    technical_failure_action: str = "stop_and_adjudicate_before_unblinding"

    def __post_init__(self) -> None:
        if not self.complete_cohort_required or not self.no_silent_intersection:
            raise ValueError("v1 requires complete-cohort evidence with no silent intersection")
        if self.invalid_evidence_is_scientific_null:
            raise ValueError("invalid/missing evidence must not be converted to a scientific null")
        if self.technical_failure_action != "stop_and_adjudicate_before_unblinding":
            raise ValueError("unknown v1 technical-failure action")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "complete_cohort_required": True,
            "no_silent_intersection": True,
            "invalid_evidence_is_scientific_null": False,
            "technical_failure_action": self.technical_failure_action,
        }

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> "Kumar2024E001EvidenceHandlingAuthority":
        return cls(
            complete_cohort_required=_required_bool(
                "complete_cohort_required", payload.get("complete_cohort_required"), True
            ),
            no_silent_intersection=_required_bool(
                "no_silent_intersection", payload.get("no_silent_intersection"), True
            ),
            invalid_evidence_is_scientific_null=_required_bool(
                "invalid_evidence_is_scientific_null",
                payload.get("invalid_evidence_is_scientific_null"),
                False,
            ),
            technical_failure_action=_required_text(
                "technical_failure_action", payload.get("technical_failure_action")
            ),
        )


@dataclass(frozen=True)
class Kumar2024E001SubgroupAuthority:
    groups: Mapping[str, tuple[int, ...]] | None = None
    promotion_authoritative: bool = False

    def __post_init__(self) -> None:
        groups = (
            {key: tuple(values) for key, values in KUMAR2024_PROTOCOL_GROUPS.items()}
            if self.groups is None
            else {str(key): tuple(int(v) for v in values) for key, values in self.groups.items()}
        )
        if groups != KUMAR2024_PROTOCOL_GROUPS:
            raise ValueError("GR/PAR subgroup membership drifted")
        if self.promotion_authoritative:
            raise ValueError("GR/PAR subgroup results are diagnostic-only in v1")
        object.__setattr__(self, "groups", groups)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "groups": {key: list(values) for key, values in self.groups.items()},
            "role": "diagnostic_only",
            "promotion_authoritative": False,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Kumar2024E001SubgroupAuthority":
        if payload.get("role") != "diagnostic_only":
            raise ValueError("GR/PAR subgroup role must remain diagnostic_only")
        _required_bool(
            "subgroup.promotion_authoritative",
            payload.get("promotion_authoritative"),
            False,
        )
        groups = _mapping("subgroup.groups", payload.get("groups"))
        parsed = {key: tuple(int(v) for v in value) for key, value in groups.items()}
        return cls(groups=parsed, promotion_authoritative=False)


def _authority_identity(capsule: Mapping[str, Any]) -> dict[str, Any]:
    if capsule.get("capsule_fingerprint") != KUMAR2024_AUTHORITY_CAPSULE_FINGERPRINT:
        raise ValueError("wrong Kumar2024 authority capsule fingerprint")
    freeze = _mapping("authority_freeze", capsule.get("authority_freeze"))
    raw = _mapping("raw_source_fingerprint", capsule.get("raw_source_fingerprint"))
    if freeze.get("cohort_authority_fingerprint") != KUMAR2024_COHORT_AUTHORITY_FINGERPRINT:
        raise ValueError("wrong Kumar2024 cohort authority fingerprint")
    if raw.get("fingerprint") != KUMAR2024_RAW_DATASET_FINGERPRINT:
        raise ValueError("wrong Kumar2024 raw dataset fingerprint")
    cases = freeze.get("cases")
    if not isinstance(cases, list) or len(cases) != 18:
        raise ValueError("Kumar2024 decision authority requires exactly 18 case authorities")
    authorities = []
    for expected_subject, case in zip(KUMAR2024_SUBJECTS, cases, strict=True):
        mapping = _mapping("authority case", case)
        if int(mapping.get("subject", -1)) != expected_subject:
            raise ValueError("Kumar2024 authority subject order drifted")
        if mapping.get("held_out_session") != KUMAR2024_HELD_OUT_SESSION:
            raise ValueError("Kumar2024 held-out session drifted")
        authorities.append(
            {
                "subject": expected_subject,
                "authority_fingerprint": _required_text(
                    "case.authority_fingerprint", mapping.get("authority_fingerprint")
                ),
                "partition_fingerprint": _required_text(
                    "case.partition_fingerprint", mapping.get("partition_fingerprint")
                ),
                "calibration_split_fingerprint": _required_text(
                    "case.calibration_split_fingerprint",
                    mapping.get("calibration_split_fingerprint"),
                ),
                "processed_data_sha256": _required_text(
                    "case.processed_data_sha256", mapping.get("processed_data_sha256")
                ),
            }
        )
    return {
        "capsule_fingerprint": KUMAR2024_AUTHORITY_CAPSULE_FINGERPRINT,
        "cohort_authority_fingerprint": KUMAR2024_COHORT_AUTHORITY_FINGERPRINT,
        "raw_dataset_fingerprint": KUMAR2024_RAW_DATASET_FINGERPRINT,
        "dataset_id": KUMAR2024_DATASET_ID,
        "evaluation_fraction": KUMAR2024_EVALUATION_FRACTION,
        "science_source_sha": KUMAR2024_SCIENCE_SOURCE_SHA,
        "neuros_source_sha": KUMAR2024_NEUROS_SOURCE_SHA,
        "subjects": list(KUMAR2024_SUBJECTS),
        "held_out_session": KUMAR2024_HELD_OUT_SESSION,
        "case_authorities": authorities,
    }


@dataclass(frozen=True)
class Kumar2024E001DecisionPlan:
    authority: Mapping[str, Any]
    primary_criterion: Kumar2024E001PrimaryCriterion
    bootstrap: Kumar2024E001BootstrapAuthority
    control_authority: Kumar2024E001ControlAuthority
    evidence_handling: Kumar2024E001EvidenceHandlingAuthority
    subgroup_authority: Kumar2024E001SubgroupAuthority
    rationale: str
    method: str = KUMAR2024_E001_DECISION_METHOD

    def __post_init__(self) -> None:
        if self.method != KUMAR2024_E001_DECISION_METHOD:
            raise ValueError("unknown Kumar2024 E001 decision method")
        object.__setattr__(self, "rationale", _required_text("rationale", self.rationale))
        authority = dict(self.authority)
        if authority.get("capsule_fingerprint") != KUMAR2024_AUTHORITY_CAPSULE_FINGERPRINT:
            raise ValueError("decision plan must bind the exact merged Kumar2024 capsule")
        if authority.get("cohort_authority_fingerprint") != KUMAR2024_COHORT_AUTHORITY_FINGERPRINT:
            raise ValueError("decision plan must bind the exact cohort authority")
        if authority.get("raw_dataset_fingerprint") != KUMAR2024_RAW_DATASET_FINGERPRINT:
            raise ValueError("decision plan must bind the exact raw dataset")
        if authority.get("science_source_sha") != KUMAR2024_SCIENCE_SOURCE_SHA:
            raise ValueError("decision plan science source drifted")
        if authority.get("neuros_source_sha") != KUMAR2024_NEUROS_SOURCE_SHA:
            raise ValueError("decision plan neurOS source drifted")
        if authority.get("dataset_id") != KUMAR2024_DATASET_ID:
            raise ValueError("decision plan dataset identity drifted")
        if float(authority.get("evaluation_fraction", -1.0)) != KUMAR2024_EVALUATION_FRACTION:
            raise ValueError("decision plan evaluation fraction drifted")
        if authority.get("held_out_session") != KUMAR2024_HELD_OUT_SESSION:
            raise ValueError("decision plan held-out session drifted")
        if authority.get("subjects") != list(KUMAR2024_SUBJECTS):
            raise ValueError("decision plan must bind subjects 1..18 exactly")
        cases = authority.get("case_authorities")
        if not isinstance(cases, list) or len(cases) != 18:
            raise ValueError("decision plan must bind all 18 case authorities")
        normalized_cases = []
        for expected_subject, case in zip(KUMAR2024_SUBJECTS, cases, strict=True):
            mapping = _mapping("decision case authority", case)
            if int(mapping.get("subject", -1)) != expected_subject:
                raise ValueError("decision case-authority subject order drifted")
            normalized_cases.append(
                {
                    "subject": expected_subject,
                    "authority_fingerprint": _required_text(
                        "authority_fingerprint", mapping.get("authority_fingerprint")
                    ),
                    "partition_fingerprint": _required_text(
                        "partition_fingerprint", mapping.get("partition_fingerprint")
                    ),
                    "calibration_split_fingerprint": _required_text(
                        "calibration_split_fingerprint",
                        mapping.get("calibration_split_fingerprint"),
                    ),
                    "processed_data_sha256": _required_text(
                        "processed_data_sha256", mapping.get("processed_data_sha256")
                    ),
                }
            )
        cohort_identity = {
            "dataset_id": KUMAR2024_DATASET_ID,
            "subjects": list(KUMAR2024_SUBJECTS),
            "held_out_session": KUMAR2024_HELD_OUT_SESSION,
            "evaluation_fraction": KUMAR2024_EVALUATION_FRACTION,
            "raw_dataset_fingerprint": KUMAR2024_RAW_DATASET_FINGERPRINT,
            "case_authorities": normalized_cases,
        }
        observed_cohort_fingerprint = sha256(
            _canonical_json(cohort_identity).encode("utf-8")
        ).hexdigest()
        if observed_cohort_fingerprint != KUMAR2024_COHORT_AUTHORITY_FINGERPRINT:
            raise ValueError("decision case authorities do not reproduce the frozen cohort fingerprint")
        if (
            self.primary_criterion.calibration_per_class
            > KUMAR2024_MIN_MAX_BUDGET_PER_CLASS
        ):
            raise ValueError(
                "primary calibration budget exceeds the frozen cohort-wide supported minimum"
            )

    @classmethod
    def from_verified_authority_capsule(
        cls,
        capsule_path: str | Path,
        *,
        primary_criterion: Kumar2024E001PrimaryCriterion,
        bootstrap: Kumar2024E001BootstrapAuthority,
        rationale: str,
    ) -> "Kumar2024E001DecisionPlan":
        capsule = load_kumar2024_authority_capsule(capsule_path)
        return cls(
            authority=_authority_identity(capsule),
            primary_criterion=primary_criterion,
            bootstrap=bootstrap,
            control_authority=Kumar2024E001ControlAuthority(),
            evidence_handling=Kumar2024E001EvidenceHandlingAuthority(),
            subgroup_authority=Kumar2024E001SubgroupAuthority(),
            rationale=rationale,
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(KUMAR2024_E001_DOMAIN, self._mapping_without_fingerprint())

    def _mapping_without_fingerprint(self) -> dict[str, Any]:
        return {
            "schema_version": KUMAR2024_E001_SCHEMA_VERSION,
            "method": self.method,
            "authority": dict(self.authority),
            "control_authority": self.control_authority.to_mapping(),
            "primary_criterion": self.primary_criterion.to_mapping(),
            "bootstrap": self.bootstrap.to_mapping(),
            "evidence_handling": self.evidence_handling.to_mapping(),
            "subgroup_authority": self.subgroup_authority.to_mapping(),
            "decision_semantics": {
                "primary_question": "cross_covariance_dependence",
                "primary_pass_rule": (
                    "participant bootstrap CI lower bound for density minus offdiagonal "
                    "ablation is >= the explicitly preregistered minimum effect"
                ),
                "closed_family_predictive_adversary_role": "descriptive_adversary",
                "information_novelty_gate": E001_INFORMATION_NOVELTY_GATE,
                "information_novelty_promotion_eligible": False,
                "current_density_representation_information_novel": False,
                "biological_mechanism_established": False,
                "physical_quantum_promotion_eligible": False,
                "confirmatory_outcomes_observed": False,
                "evaluation_executed": False,
                "qualified_executor_bound": False,
                "execution_authorized": False,
            },
            "rationale": self.rationale,
        }

    def to_mapping(self) -> dict[str, Any]:
        payload = self._mapping_without_fingerprint()
        payload["plan_fingerprint"] = self.fingerprint
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Kumar2024E001DecisionPlan":
        if int(payload.get("schema_version", 0)) != KUMAR2024_E001_SCHEMA_VERSION:
            raise ValueError("unexpected Kumar2024 E001 decision schema")
        if payload.get("method") != KUMAR2024_E001_DECISION_METHOD:
            raise ValueError("unexpected Kumar2024 E001 decision method")
        semantics = _mapping("decision_semantics", payload.get("decision_semantics"))
        expected_semantics = {
            "primary_question": "cross_covariance_dependence",
            "primary_pass_rule": (
                "participant bootstrap CI lower bound for density minus offdiagonal "
                "ablation is >= the explicitly preregistered minimum effect"
            ),
            "closed_family_predictive_adversary_role": "descriptive_adversary",
            "information_novelty_gate": E001_INFORMATION_NOVELTY_GATE,
            "information_novelty_promotion_eligible": False,
            "current_density_representation_information_novel": False,
            "biological_mechanism_established": False,
            "physical_quantum_promotion_eligible": False,
            "confirmatory_outcomes_observed": False,
            "evaluation_executed": False,
            "qualified_executor_bound": False,
            "execution_authorized": False,
        }
        if dict(semantics) != expected_semantics:
            raise ValueError("Kumar2024 E001 decision semantics drifted")
        plan = cls(
            authority=dict(_mapping("authority", payload.get("authority"))),
            primary_criterion=Kumar2024E001PrimaryCriterion.from_mapping(
                _mapping("primary_criterion", payload.get("primary_criterion"))
            ),
            bootstrap=Kumar2024E001BootstrapAuthority.from_mapping(
                _mapping("bootstrap", payload.get("bootstrap"))
            ),
            control_authority=Kumar2024E001ControlAuthority.from_mapping(
                _mapping("control_authority", payload.get("control_authority"))
            ),
            evidence_handling=Kumar2024E001EvidenceHandlingAuthority.from_mapping(
                _mapping("evidence_handling", payload.get("evidence_handling"))
            ),
            subgroup_authority=Kumar2024E001SubgroupAuthority.from_mapping(
                _mapping("subgroup_authority", payload.get("subgroup_authority"))
            ),
            rationale=_required_text("rationale", payload.get("rationale")),
            method=_required_text("method", payload.get("method")),
        )
        observed = _required_text("plan_fingerprint", payload.get("plan_fingerprint"))
        if observed != plan.fingerprint:
            raise ValueError("Kumar2024 E001 decision plan fingerprint is stale or noncanonical")
        return plan


@dataclass(frozen=True)
class Kumar2024E001PreregistrationSeal:
    plan: Kumar2024E001DecisionPlan
    preregistration: PreregistrationEvidence

    def __post_init__(self) -> None:
        if not self.preregistration.matches_policy(self.plan.fingerprint):
            raise ValueError("external preregistration does not bind the exact decision plan")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(KUMAR2024_E001_SEAL_DOMAIN, self._mapping_without_fingerprint())

    def _mapping_without_fingerprint(self) -> dict[str, Any]:
        return {
            "schema_version": KUMAR2024_E001_SCHEMA_VERSION,
            "artifact_role": KUMAR2024_E001_SEAL_ROLE,
            "plan": self.plan.to_mapping(),
            "preregistration": self.preregistration.to_mapping(),
            "evaluation_executed": False,
            "confirmatory_outcomes_observed": False,
            "execution_authorized": False,
            "information_novelty_promotion_eligible": False,
            "biological_mechanism_established": False,
            "physical_quantum_promotion_eligible": False,
            "claim_boundary": (
                "This seal binds an externally registered decision policy. It does not execute "
                "Kumar2024 E001, establish a biological mechanism, establish representation "
                "information novelty for the covariance-equivalent density constructor, or "
                "authorize a physical-quantum claim."
            ),
        }

    def to_mapping(self) -> dict[str, Any]:
        payload = self._mapping_without_fingerprint()
        payload["artifact_fingerprint"] = self.fingerprint
        return payload

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> "Kumar2024E001PreregistrationSeal":
        if int(payload.get("schema_version", 0)) != KUMAR2024_E001_SCHEMA_VERSION:
            raise ValueError("unexpected preregistration seal schema")
        if payload.get("artifact_role") != KUMAR2024_E001_SEAL_ROLE:
            raise ValueError("unexpected preregistration seal role")
        for key in (
            "evaluation_executed",
            "confirmatory_outcomes_observed",
            "execution_authorized",
            "information_novelty_promotion_eligible",
            "biological_mechanism_established",
            "physical_quantum_promotion_eligible",
        ):
            _required_bool(key, payload.get(key), False)
        expected_boundary = (
            "This seal binds an externally registered decision policy. It does not execute "
            "Kumar2024 E001, establish a biological mechanism, establish representation "
            "information novelty for the covariance-equivalent density constructor, or "
            "authorize a physical-quantum claim."
        )
        if payload.get("claim_boundary") != expected_boundary:
            raise ValueError("preregistration seal claim boundary drifted")
        preregistration_payload = _mapping(
            "preregistration", payload.get("preregistration")
        )
        _required_text(
            "preregistration.registration_uri",
            preregistration_payload.get("registration_uri"),
        )
        _required_text(
            "preregistration.registered_at",
            preregistration_payload.get("registered_at"),
        )
        seal = cls(
            plan=Kumar2024E001DecisionPlan.from_mapping(
                _mapping("plan", payload.get("plan"))
            ),
            preregistration=PreregistrationEvidence.from_mapping(
                preregistration_payload
            ),
        )
        observed = _required_text(
            "artifact_fingerprint", payload.get("artifact_fingerprint")
        )
        if observed != seal.fingerprint:
            raise ValueError("preregistration seal fingerprint is stale or noncanonical")
        return seal
