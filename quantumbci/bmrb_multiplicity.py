"""Closed-world multiplicity authority for candidate neural-mechanism families.

BMRB's scientific promotion rules are effect- and gate-based. Participant sign-flip p-values
are evidence summaries, not hidden promotion switches. This module therefore does not silently
inject a multiple-testing p-value correction into an existing decision rule.

Instead it closes a different researcher-degree-of-freedom loophole first: trying many candidate
mechanisms, layers, tasks, or metrics and promoting whichever survivor looks best. A multiplicity
plan freezes the complete candidate family, the family order, each candidate's role, and which
candidates have promotion authority before final evidence is inspected.

The v1 promotion rule is deliberately narrow: exactly one predeclared primary candidate per family
can promote. Secondary and exploratory candidates remain reportable evidence but cannot inherit
promotion authority post hoc. Future corrected multi-primary procedures should use a new explicit
method rather than weakening this contract.
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


@dataclass(frozen=True)
class BMRBMultiplicityCandidate:
    """One candidate hypothesis frozen inside a closed multiplicity family."""

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
        order = int(self.order)
        if order < 0 or order != self.order:
            raise ValueError("order must be a non-negative integer")
        object.__setattr__(self, "order", order)
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
            order=int(payload.get("order", -1)),
            rationale=_required_text("rationale", payload.get("rationale")),
        )


@dataclass(frozen=True)
class BMRBMultiplicityPlan:
    """Predeclared closed-world candidate-family authority.

    v1 uses family grouping and ordering rather than adding a new p-value gate. Every family has
    exactly one primary candidate at order zero. Secondary and exploratory candidates are always
    reported but never gain promotion authority from their observed result.
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
        if not family_order:
            raise ValueError("multiplicity plan requires at least one family")
        if len(set(family_order)) != len(family_order):
            raise ValueError("family_order values must be unique")
        object.__setattr__(self, "family_order", family_order)

        candidates = tuple(self.candidates)
        if not candidates:
            raise ValueError("multiplicity plan requires at least one candidate")
        if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
            raise ValueError("candidate_id values must be unique across the multiplicity plan")
        candidate_families = {candidate.family_id for candidate in candidates}
        if candidate_families != set(family_order):
            missing = sorted(set(family_order) - candidate_families)
            extra = sorted(candidate_families - set(family_order))
            raise ValueError(
                "family_order must exactly cover candidate families; "
                f"missing={missing} extra={extra}"
            )

        family_position = {family_id: index for index, family_id in enumerate(family_order)}
        ordered = tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    family_position[candidate.family_id],
                    candidate.order,
                    candidate.candidate_id,
                ),
            )
        )
        object.__setattr__(self, "candidates", ordered)

        for family_id in family_order:
            family = [candidate for candidate in ordered if candidate.family_id == family_id]
            observed_orders = [candidate.order for candidate in family]
            expected_orders = list(range(len(family)))
            if observed_orders != expected_orders:
                raise ValueError(
                    f"family {family_id!r} candidate order must be contiguous from zero"
                )
            primaries = [candidate for candidate in family if candidate.role == "primary"]
            if len(primaries) != 1:
                raise ValueError(
                    f"family {family_id!r} requires exactly one predeclared primary candidate"
                )
            if primaries[0].order != 0:
                raise ValueError(
                    f"family {family_id!r} primary candidate must occupy order zero"
                )

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(candidate.candidate_id for candidate in self.candidates)

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
            "promotion_rule": "predeclared_primary_only",
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
        if payload.get("promotion_rule") != "predeclared_primary_only":
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
            "candidate_count": len(self.candidates),
            "naive_any_survivor": self.naive_any_survivor,
            "authorized_any_promotion": self.authorized_any_promotion,
            "suppressed_nonprimary_survivors": list(self.suppressed_nonprimary_survivors),
            "candidates": [candidate.to_mapping() for candidate in self.candidates],
            "interpretation": (
                "A scientific PASS does not acquire promotion authority merely because it was the "
                "best-looking member of a searched candidate family. v1 promotion authority is "
                "restricted to the predeclared primary candidate in each closed family."
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

    observed = {str(key): value for key, value in scientific_results.items()}
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
        authority = candidate.role == "primary"
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

    if int(exploratory_candidates) < 1:
        raise ValueError("exploratory_candidates must be positive")
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
        for index in range(1, int(exploratory_candidates) + 1)
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
