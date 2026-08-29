"""Closed-world multiplicity authority for one candidate neural-mechanism family.

BMRB's scientific promotion rules are effect- and gate-based. Participant sign-flip p-values
are evidence summaries, not hidden promotion switches. This module therefore does not silently
inject a multiple-testing p-value correction into an existing decision rule.

Instead it closes a different researcher-degree-of-freedom loophole first: trying many candidate
mechanisms, layers, tasks, or metrics and promoting whichever survivor looks best. A multiplicity
plan freezes one complete candidate family, each candidate's order and role, and the single
promotion-authoritative primary before final evidence is inspected.

The v1 promotion rule is deliberately narrow: exactly one closed family and exactly one
predeclared primary candidate can promote. Secondary and exploratory candidates remain reportable
evidence but cannot inherit promotion authority post hoc. Multi-family or corrected multi-primary
procedures must use a new explicit method rather than laundering extra search opportunities into
v1 family labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .preregistration import canonical_scientific_fingerprint

BMRB_MULTIPLICITY_METHOD = "predeclared_candidate_family_primary_only_v1"
BMRB_MULTIPLICITY_RESULT_ROLE = "bmrb_multiplicity_decision_v1"
CANDIDATE_ROLES = frozenset({"primary", "secondary", "exploratory"})


def _required_text(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _strict_bool(name: str, value: Any) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a JSON/Python boolean")
    return value


def _nonnegative_int(name: str, value: Any) -> int:
    if type(value) is bool:
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if integer < 0 or integer != value:
        raise ValueError(f"{name} must be a non-negative integer")
    return integer


@dataclass(frozen=True)
class BMRBMultiplicityCandidate:
    """One candidate hypothesis frozen inside the single closed v1 family."""

    candidate_id: str
    family_id: str
    role: str
    order: int
    rationale: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _required_text("candidate_id", self.candidate_id))
        object.__setattr__(self, "family_id", _required_text("family_id", self.family_id))
        role = _required_text("role", self.role)
        if role not in CANDIDATE_ROLES:
            raise ValueError(f"role must be one of {sorted(CANDIDATE_ROLES)}")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "order", _nonnegative_int("order", self.order))
        object.__setattr__(self, "rationale", _required_text("rationale", self.rationale))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "family_id": self.family_id,
            "role": self.role,
            "order": self.order,
            "rationale": self.rationale,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BMRBMultiplicityCandidate":
        return cls(
            candidate_id=_required_text("candidate_id", payload.get("candidate_id")),
            family_id=_required_text("family_id", payload.get("family_id")),
            role=_required_text("role", payload.get("role")),
            order=_nonnegative_int("order", payload.get("order", -1)),
            rationale=_required_text("rationale", payload.get("rationale")),
        )


@dataclass(frozen=True)
class BMRBMultiplicityPlan:
    """Predeclared closed-world authority for exactly one candidate family.

    v1 does not add a p-value gate and does not support hierarchical multi-family promotion.
    Exactly one candidate in the one closed family is primary and occupies order zero. Secondary
    and exploratory candidates are reported but never gain promotion authority from their observed
    result.
    """

    plan_id: str
    family_order: tuple[str, ...]
    candidates: tuple[BMRBMultiplicityCandidate, ...]
    scientific_rationale: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _required_text("plan_id", self.plan_id))
        object.__setattr__(
            self,
            "scientific_rationale",
            _required_text("scientific_rationale", self.scientific_rationale),
        )
        family_order = tuple(_required_text("family_id", value) for value in self.family_order)
        if len(family_order) != 1:
            raise ValueError(
                "v1 multiplicity authority requires exactly one closed candidate family; "
                "multi-family multiplicity requires a new method"
            )
        object.__setattr__(self, "family_order", family_order)

        candidates = tuple(self.candidates)
        if not candidates:
            raise ValueError("multiplicity plan requires at least one candidate")
        if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
            raise ValueError("candidate_id values must be unique across the multiplicity plan")
        family_id = family_order[0]
        candidate_families = {candidate.family_id for candidate in candidates}
        if candidate_families != {family_id}:
            raise ValueError(
                "every v1 multiplicity candidate must belong to the single frozen family "
                f"{family_id!r}; observed={sorted(candidate_families)}"
            )

        ordered = tuple(sorted(candidates, key=lambda candidate: (candidate.order, candidate.candidate_id)))
        object.__setattr__(self, "candidates", ordered)
        observed_orders = [candidate.order for candidate in ordered]
        expected_orders = list(range(len(ordered)))
        if observed_orders != expected_orders:
            raise ValueError(
                f"family {family_id!r} candidate order must be contiguous from zero"
            )
        primaries = [candidate for candidate in ordered if candidate.role == "primary"]
        if len(primaries) != 1:
            raise ValueError(
                f"family {family_id!r} requires exactly one predeclared primary candidate"
            )
        if primaries[0].order != 0:
            raise ValueError(
                f"family {family_id!r} primary candidate must occupy order zero"
            )

    @property
    def family_id(self) -> str:
        return self.family_order[0]

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(candidate.candidate_id for candidate in self.candidates)

    @property
    def primary_candidate_id(self) -> str:
        return next(candidate.candidate_id for candidate in self.candidates if candidate.role == "primary")

    def candidate(self, candidate_id: str) -> BMRBMultiplicityCandidate:
        requested = _required_text("candidate_id", candidate_id)
        for candidate in self.candidates:
            if candidate.candidate_id == requested:
                return candidate
        raise KeyError(f"candidate {requested!r} is outside the frozen multiplicity family")

    def decision_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "method": BMRB_MULTIPLICITY_METHOD,
            "plan_id": self.plan_id,
            "family_order": list(self.family_order),
            "candidates": [candidate.to_mapping() for candidate in self.candidates],
            "promotion_rule": "single_family_predeclared_primary_only",
            "scientific_rationale": self.scientific_rationale,
        }

    @property
    def plan_fingerprint(self) -> str:
        return canonical_scientific_fingerprint(
            "quantumbci.bmrb-multiplicity-plan.v1",
            self.decision_payload(),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {**self.decision_payload(), "plan_fingerprint": self.plan_fingerprint}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BMRBMultiplicityPlan":
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("BMRB multiplicity plan schema_version must be 1")
        if payload.get("method") != BMRB_MULTIPLICITY_METHOD:
            raise ValueError("BMRB multiplicity plan method mismatch")
        if payload.get("promotion_rule") != "single_family_predeclared_primary_only":
            raise ValueError("BMRB multiplicity promotion_rule mismatch")
        raw_candidates = payload.get("candidates")
        if not isinstance(raw_candidates, list):
            raise ValueError("BMRB multiplicity candidates must be a list")
        raw_families = payload.get("family_order")
        if not isinstance(raw_families, list):
            raise ValueError("BMRB multiplicity family_order must be a list")
        plan = cls(
            plan_id=_required_text("plan_id", payload.get("plan_id")),
            family_order=tuple(str(value) for value in raw_families),
            candidates=tuple(
                BMRBMultiplicityCandidate.from_mapping(item)
                for item in raw_candidates
                if isinstance(item, Mapping)
            ),
            scientific_rationale=_required_text(
                "scientific_rationale", payload.get("scientific_rationale")
            ),
        )
        if len(plan.candidates) != len(raw_candidates):
            raise ValueError("every multiplicity candidate must be an object")
        supplied = _required_text("plan_fingerprint", payload.get("plan_fingerprint"))
        if supplied != plan.plan_fingerprint:
            raise ValueError("BMRB multiplicity plan fingerprint mismatch")
        if plan.to_mapping() != dict(payload):
            raise ValueError("BMRB multiplicity plan is noncanonical")
        return plan


@dataclass(frozen=True)
class BMRBMultiplicityCandidateDecision:
    candidate_id: str
    family_id: str
    role: str
    order: int
    scientific_criteria_passed: bool
    promotion_authority: bool
    promotion_eligible: bool

    def to_mapping(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "family_id": self.family_id,
            "role": self.role,
            "order": self.order,
            "scientific_criteria_passed": self.scientific_criteria_passed,
            "promotion_authority": self.promotion_authority,
            "promotion_eligible": self.promotion_eligible,
        }


@dataclass(frozen=True)
class BMRBMultiplicityDecision:
    plan: BMRBMultiplicityPlan
    candidates: tuple[BMRBMultiplicityCandidateDecision, ...]

    @property
    def naive_any_survivor(self) -> bool:
        return any(candidate.scientific_criteria_passed for candidate in self.candidates)

    @property
    def authorized_any_promotion(self) -> bool:
        return any(candidate.promotion_eligible for candidate in self.candidates)

    @property
    def suppressed_nonprimary_survivors(self) -> tuple[str, ...]:
        return tuple(
            candidate.candidate_id
            for candidate in self.candidates
            if candidate.scientific_criteria_passed and not candidate.promotion_authority
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_role": BMRB_MULTIPLICITY_RESULT_ROLE,
            "plan_fingerprint": self.plan.plan_fingerprint,
            "family_id": self.plan.family_id,
            "primary_candidate_id": self.plan.primary_candidate_id,
            "candidate_count": len(self.candidates),
            "naive_any_survivor": self.naive_any_survivor,
            "authorized_any_promotion": self.authorized_any_promotion,
            "suppressed_nonprimary_survivors": list(self.suppressed_nonprimary_survivors),
            "candidates": [candidate.to_mapping() for candidate in self.candidates],
            "interpretation": (
                "A scientific PASS does not acquire promotion authority merely because it was the "
                "best-looking member of a searched candidate family. v1 admits exactly one closed "
                "family and restricts promotion authority to its predeclared primary candidate."
            ),
            "physical_quantum_promotion_eligible": False,
        }


def apply_multiplicity_plan(
    plan: BMRBMultiplicityPlan,
    scientific_results: Mapping[str, bool],
) -> BMRBMultiplicityDecision:
    """Apply one frozen multiplicity authority to exact closed-world scientific decisions.

    ``scientific_results`` must contain every declared candidate exactly once and no undeclared
    candidates. This prevents silent dropping of failed candidates and silent addition of a new
    winner after evidence has been inspected.
    """

    if any(not isinstance(key, str) for key in scientific_results):
        raise ValueError("multiplicity result keys must be candidate-id strings")
    observed = dict(scientific_results)
    expected = set(plan.candidate_ids)
    actual = set(observed)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "multiplicity results must exactly match the frozen candidate family; "
            f"missing={missing} extra={extra}"
        )

    decisions: list[BMRBMultiplicityCandidateDecision] = []
    for candidate in plan.candidates:
        scientific_pass = _strict_bool(
            f"scientific_results[{candidate.candidate_id!r}]",
            observed[candidate.candidate_id],
        )
        authority = candidate.candidate_id == plan.primary_candidate_id
        decisions.append(
            BMRBMultiplicityCandidateDecision(
                candidate_id=candidate.candidate_id,
                family_id=candidate.family_id,
                role=candidate.role,
                order=candidate.order,
                scientific_criteria_passed=scientific_pass,
                promotion_authority=authority,
                promotion_eligible=bool(authority and scientific_pass),
            )
        )
    return BMRBMultiplicityDecision(plan=plan, candidates=tuple(decisions))


def winner_picking_demo_plan(*, exploratory_candidates: int = 19) -> BMRBMultiplicityPlan:
    """Return a deterministic software fixture illustrating post-hoc winner selection.

    The fixture is not a biological multiplicity recommendation. It freezes one primary candidate
    and a configurable exploratory search family so tests can prove that exploratory survivors do
    not become promotion-authoritative after inspection.
    """

    if type(exploratory_candidates) is bool or int(exploratory_candidates) < 1:
        raise ValueError("exploratory_candidates must be positive")
    exploratory_candidates = int(exploratory_candidates)
    candidates = [
        BMRBMultiplicityCandidate(
            candidate_id="mechanism-primary",
            family_id="mechanism-search",
            role="primary",
            order=0,
            rationale="Predeclared primary mechanism for the software multiplicity fixture.",
        )
    ]
    candidates.extend(
        BMRBMultiplicityCandidate(
            candidate_id=f"mechanism-exploratory-{index:02d}",
            family_id="mechanism-search",
            role="exploratory",
            order=index,
            rationale="Exploratory candidate retained for complete reporting, not promotion.",
        )
        for index in range(1, exploratory_candidates + 1)
    )
    return BMRBMultiplicityPlan(
        plan_id="winner-picking-software-fixture-v1",
        family_order=("mechanism-search",),
        candidates=tuple(candidates),
        scientific_rationale=(
            "Software fixture proving that searching more candidates cannot transfer promotion "
            "authority away from the predeclared primary mechanism."
        ),
    )
