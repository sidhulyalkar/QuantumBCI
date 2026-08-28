"""Gate-level operating diagnostics for known-truth BMRB validation.

The operating-curves artifact answers whether BMRB makes the intended overall decision.
This sibling artifact explains *how* that decision emerges by reporting gate confusion,
first-failing-gate behavior, false-promotion escape paths, and known-positive loss paths.

It deliberately reuses the frozen BMRB operating policy and production validation replicate
runner. It does not define universal qualification thresholds and does not turn software-invalid
evidence into scientific failures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from .bmrb_validation import (
    BMRBValidationReplicate,
    BMRBValidationScenario,
    default_validation_scenarios,
    run_validation_replicate,
)
from .bmrb_validation_operating import (
    BMRBOperatingStudyPolicy,
    OperatingCurveGrid,
    SimulationSeedPartition,
    _scaled_scenario,
    _scientific_fingerprint,
    _wilson_interval,
)

BMRB_GATE_DIAGNOSTICS_BENCHMARK = "BMRB_KNOWN_TRUTH_GATE_DIAGNOSTICS_V1"
BMRB_GATE_DIAGNOSTICS_METHOD = "frozen_gate_confusion_monte_carlo_v1"
GateName = Literal["effect", "adversary", "conservation", "coverage"]
GATE_ORDER: tuple[GateName, ...] = (
    "effect",
    "adversary",
    "conservation",
    "coverage",
)


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _required_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _required_list(name: str, value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


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
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if normalized < 0 or normalized != value:
        raise ValueError(f"{name} must be a non-negative integer")
    return normalized


def _positive_int(name: str, value: Any) -> int:
    normalized = _nonnegative_int(name, value)
    if normalized == 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _fraction(name: str, value: Any) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return normalized


def _close(name: str, observed: Any, expected: float, *, atol: float = 1e-12) -> None:
    value = float(observed)
    if not np.isfinite(value) or not np.isclose(value, expected, rtol=0.0, atol=atol):
        raise ValueError(f"{name} is inconsistent with reconstructed diagnostics")


def expected_gate_passes(scenario: BMRBValidationScenario) -> dict[GateName, bool]:
    """Return declared gate truth for one known-truth scenario.

    Current v1 DGMs isolate one intended failing gate at a time. Known-positive scenarios
    expect all gates to pass. This function makes that assumption explicit and auditable.
    """

    expected: dict[GateName, bool] = {gate: True for gate in GATE_ORDER}
    failure = scenario.expected_failure_component
    if failure is not None:
        if failure not in GATE_ORDER:
            raise ValueError(f"scenario declares unsupported gate failure: {failure!r}")
        expected[failure] = False
    return expected


def observed_gate_passes(row: BMRBValidationReplicate) -> dict[GateName, bool]:
    return {
        "effect": bool(row.effect_criteria_passed),
        "adversary": bool(row.adversary_survival_passed),
        "conservation": bool(row.conservation_criteria_passed),
        "coverage": bool(row.coverage_criteria_passed),
    }


def first_failing_gate(row: BMRBValidationReplicate) -> GateName | None:
    observed = observed_gate_passes(row)
    return next((gate for gate in GATE_ORDER if not observed[gate]), None)


@dataclass(frozen=True)
class GateConfusion:
    gate: GateName
    expected_pass: bool
    trials: int
    observed_passes: int
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    observed_pass_rate: float
    pass_rate_ci_lower: float
    pass_rate_ci_upper: float

    @classmethod
    def from_rows(
        cls,
        gate: GateName,
        *,
        expected_pass: bool,
        rows: tuple[BMRBValidationReplicate, ...],
    ) -> "GateConfusion":
        if not rows:
            raise ValueError("gate confusion requires at least one replicate")
        observed = [observed_gate_passes(row)[gate] for row in rows]
        passes = int(sum(observed))
        trials = len(rows)
        if expected_pass:
            tp, fn, tn, fp = passes, trials - passes, 0, 0
        else:
            tp, fn, tn, fp = 0, 0, trials - passes, passes
        lower, upper = _wilson_interval(passes, trials)
        return cls(
            gate=gate,
            expected_pass=bool(expected_pass),
            trials=trials,
            observed_passes=passes,
            true_positive=tp,
            true_negative=tn,
            false_positive=fp,
            false_negative=fn,
            observed_pass_rate=float(passes / trials),
            pass_rate_ci_lower=lower,
            pass_rate_ci_upper=upper,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "expected_pass": self.expected_pass,
            "trials": self.trials,
            "observed_passes": self.observed_passes,
            "true_positive": self.true_positive,
            "true_negative": self.true_negative,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "observed_pass_rate": self.observed_pass_rate,
            "pass_rate_ci_lower": self.pass_rate_ci_lower,
            "pass_rate_ci_upper": self.pass_rate_ci_upper,
        }


@dataclass(frozen=True)
class GateDiagnosticCell:
    cell_index: int
    scenario_id: str
    truth_class: str
    expected_scientific_pass: bool
    expected_failure_component: str | None
    participant_count: int
    effect_scale: float
    heterogeneity_scale: float
    measurement_noise_scale: float
    base_seed: int
    replicates: int
    observed_scientific_passes: int
    gate_confusion: tuple[GateConfusion, ...]
    first_failing_gate_counts: Mapping[str, int]
    first_failure_localization_rate: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "cell_index": self.cell_index,
            "scenario_id": self.scenario_id,
            "truth_class": self.truth_class,
            "expected_scientific_pass": self.expected_scientific_pass,
            "expected_failure_component": self.expected_failure_component,
            "participant_count": self.participant_count,
            "effect_scale": self.effect_scale,
            "heterogeneity_scale": self.heterogeneity_scale,
            "measurement_noise_scale": self.measurement_noise_scale,
            "base_seed": self.base_seed,
            "replicates": self.replicates,
            "observed_scientific_passes": self.observed_scientific_passes,
            "observed_scientific_pass_rate": float(
                self.observed_scientific_passes / self.replicates
            ),
            "gate_confusion": [item.to_mapping() for item in self.gate_confusion],
            "first_failing_gate_counts": dict(self.first_failing_gate_counts),
            "first_failure_localization_rate": self.first_failure_localization_rate,
        }


def _summarize_cell(
    *,
    cell_index: int,
    scenario: BMRBValidationScenario,
    participant_count: int,
    effect_scale: float,
    heterogeneity_scale: float,
    measurement_noise_scale: float,
    base_seed: int,
    rows: tuple[BMRBValidationReplicate, ...],
) -> GateDiagnosticCell:
    if not rows:
        raise ValueError("gate-diagnostic cell requires at least one replicate")
    expected = expected_gate_passes(scenario)
    confusion = tuple(
        GateConfusion.from_rows(gate, expected_pass=expected[gate], rows=rows)
        for gate in GATE_ORDER
    )
    counts = {gate: 0 for gate in GATE_ORDER}
    counts["none"] = 0
    for row in rows:
        failure = first_failing_gate(row)
        counts["none" if failure is None else failure] += 1
    expected_first = scenario.expected_failure_component
    expected_key = "none" if expected_first is None else expected_first
    localized = counts[expected_key] / len(rows)
    scientific_passes = sum(row.scientific_criteria_passed for row in rows)
    if scientific_passes != counts["none"]:
        raise ValueError("scientific PASS must equal absence of any failing gate")
    return GateDiagnosticCell(
        cell_index=int(cell_index),
        scenario_id=scenario.scenario_id,
        truth_class=scenario.truth_class,
        expected_scientific_pass=scenario.expected_scientific_pass,
        expected_failure_component=scenario.expected_failure_component,
        participant_count=int(participant_count),
        effect_scale=float(effect_scale),
        heterogeneity_scale=float(heterogeneity_scale),
        measurement_noise_scale=float(measurement_noise_scale),
        base_seed=int(base_seed),
        replicates=len(rows),
        observed_scientific_passes=int(scientific_passes),
        gate_confusion=confusion,
        first_failing_gate_counts=counts,
        first_failure_localization_rate=float(localized),
    )


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _aggregate_gate_confusion(cells: tuple[GateDiagnosticCell, ...]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for gate in GATE_ORDER:
        entries = [
            next(item for item in cell.gate_confusion if item.gate == gate) for cell in cells
        ]
        tp = sum(item.true_positive for item in entries)
        tn = sum(item.true_negative for item in entries)
        fp = sum(item.false_positive for item in entries)
        fn = sum(item.false_negative for item in entries)
        pass_support = tp + fn
        failure_support = tn + fp
        output[gate] = {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "expected_pass_support": pass_support,
            "expected_failure_support": failure_support,
            "pass_sensitivity": _safe_rate(tp, pass_support),
            "failure_specificity": _safe_rate(tn, failure_support),
            "false_pass_rate": _safe_rate(fp, failure_support),
            "false_fail_rate": _safe_rate(fn, pass_support),
        }
    return output


def _aggregate_mapping(cells: tuple[GateDiagnosticCell, ...]) -> dict[str, Any]:
    false_promotion_escape = {gate: 0 for gate in GATE_ORDER}
    positive_loss_first_gate = {gate: 0 for gate in GATE_ORDER}
    first_failure_counts = {gate: 0 for gate in GATE_ORDER}
    first_failure_counts["none"] = 0

    for cell in cells:
        for key, value in cell.first_failing_gate_counts.items():
            first_failure_counts[key] += int(value)
        if not cell.expected_scientific_pass:
            expected_failure = cell.expected_failure_component
            if expected_failure is not None:
                false_promotion_escape[expected_failure] += cell.observed_scientific_passes
        else:
            for gate in GATE_ORDER:
                positive_loss_first_gate[gate] += int(cell.first_failing_gate_counts[gate])

    return {
        "gate_confusion": _aggregate_gate_confusion(cells),
        "first_failing_gate_counts": first_failure_counts,
        "mean_first_failure_localization_rate": float(
            np.mean([cell.first_failure_localization_rate for cell in cells])
        ),
        "false_promotion_escape_counts_by_expected_gate": false_promotion_escape,
        "known_positive_loss_first_gate_counts": positive_loss_first_gate,
        "software_invalid_trials_in_gate_confusion": 0,
        "software_invalid_handling": (
            "Software-invalid evidence is excluded from scientific gate confusion and is "
            "validated separately by fail-closed exact-pairing contracts."
        ),
    }


@dataclass(frozen=True)
class BMRBGateDiagnosticsResult:
    policy: BMRBOperatingStudyPolicy
    cells: tuple[GateDiagnosticCell, ...]

    def __post_init__(self) -> None:
        if len(self.cells) != self.policy.grid.cell_count:
            raise ValueError("gate diagnostics do not cover the complete frozen grid")
        if tuple(cell.cell_index for cell in self.cells) != tuple(range(len(self.cells))):
            raise ValueError("gate-diagnostic cell indices must be complete and ordered")

    def scientific_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "benchmark": BMRB_GATE_DIAGNOSTICS_BENCHMARK,
            "method": BMRB_GATE_DIAGNOSTICS_METHOD,
            "operating_policy": self.policy.to_mapping(),
            "operating_policy_fingerprint": self.policy.policy_fingerprint,
            "aggregate": _aggregate_mapping(self.cells),
            "cells": [cell.to_mapping() for cell in self.cells],
            "qualification_defined": False,
            "interpretation": (
                "Gate diagnostics quantify decision-path behavior under declared synthetic "
                "DGMs and a frozen operating policy. Missing truth support is reported as null "
                "rather than invented. This does not validate biological truth or authorize a "
                "physical-quantum interpretation."
            ),
        }

    @property
    def artifact_fingerprint(self) -> str:
        return _scientific_fingerprint(
            "quantumbci.bmrb-gate-diagnostics-result.v1",
            self.scientific_payload(),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            **self.scientific_payload(),
            "artifact_fingerprint": self.artifact_fingerprint,
        }


def run_bmrb_gate_diagnostics(
    policy: BMRBOperatingStudyPolicy,
) -> BMRBGateDiagnosticsResult:
    """Run the frozen grid and retain gate-level decision-path diagnostics."""

    scenarios = {scenario.scenario_id: scenario for scenario in default_validation_scenarios()}
    unknown = sorted(set(policy.grid.scenario_ids) - set(scenarios))
    if unknown:
        raise ValueError(f"unknown gate-diagnostic scenario ids: {unknown}")

    cells: list[GateDiagnosticCell] = []
    combinations = product(
        policy.grid.scenario_ids,
        policy.grid.participant_counts,
        policy.grid.effect_scales,
        policy.grid.heterogeneity_scales,
        policy.grid.measurement_noise_scales,
    )
    for cell_index, (
        scenario_id,
        participant_count,
        effect_scale,
        heterogeneity_scale,
        measurement_noise_scale,
    ) in enumerate(combinations):
        scenario = _scaled_scenario(
            scenarios[scenario_id],
            effect_scale=effect_scale,
            heterogeneity_scale=heterogeneity_scale,
            measurement_noise_scale=measurement_noise_scale,
        )
        base_seed = policy.seed_partition.base_seed(policy.partition, cell_index=cell_index)
        rows = tuple(
            run_validation_replicate(
                scenario,
                replicate=replicate,
                seed=base_seed,
                participants=participant_count,
                primary_calibration_per_class=policy.primary_calibration_per_class,
                bootstrap_resamples=policy.bootstrap_resamples,
            )
            for replicate in range(policy.replicates_per_cell)
        )
        cells.append(
            _summarize_cell(
                cell_index=cell_index,
                scenario=scenario,
                participant_count=participant_count,
                effect_scale=effect_scale,
                heterogeneity_scale=heterogeneity_scale,
                measurement_noise_scale=measurement_noise_scale,
                base_seed=base_seed,
                rows=rows,
            )
        )
    return BMRBGateDiagnosticsResult(policy=policy, cells=tuple(cells))


def write_bmrb_gate_diagnostics(
    result: BMRBGateDiagnosticsResult,
    output: str | Path,
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(result.to_mapping()) + "\n", encoding="utf-8")
    return path


def _policy_from_mapping(payload: Mapping[str, Any]) -> BMRBOperatingStudyPolicy:
    grid_payload = _required_mapping("operating_policy.grid", payload.get("grid"))
    grid = OperatingCurveGrid(
        scenario_ids=tuple(_required_list("grid.scenario_ids", grid_payload.get("scenario_ids"))),
        participant_counts=tuple(
            _required_list("grid.participant_counts", grid_payload.get("participant_counts"))
        ),
        effect_scales=tuple(_required_list("grid.effect_scales", grid_payload.get("effect_scales"))),
        heterogeneity_scales=tuple(
            _required_list("grid.heterogeneity_scales", grid_payload.get("heterogeneity_scales"))
        ),
        measurement_noise_scales=tuple(
            _required_list(
                "grid.measurement_noise_scales",
                grid_payload.get("measurement_noise_scales"),
            )
        ),
    )
    if _required_text("grid_fingerprint", payload.get("grid_fingerprint")) != grid.fingerprint:
        raise ValueError("gate-diagnostic operating grid fingerprint mismatch")

    seed_payload = _required_mapping("operating_policy.seed_partition", payload.get("seed_partition"))
    seed_partition = SimulationSeedPartition(
        development_offset=_positive_int(
            "seed_partition.development_offset", seed_payload.get("development_offset")
        ),
        evaluation_offset=_positive_int(
            "seed_partition.evaluation_offset", seed_payload.get("evaluation_offset")
        ),
        cell_stride=_positive_int("seed_partition.cell_stride", seed_payload.get("cell_stride")),
        replicate_stride=_positive_int(
            "seed_partition.replicate_stride", seed_payload.get("replicate_stride")
        ),
        max_replicates_per_cell=_positive_int(
            "seed_partition.max_replicates_per_cell",
            seed_payload.get("max_replicates_per_cell"),
        ),
    )
    if (
        _required_text(
            "seed_partition_fingerprint", payload.get("seed_partition_fingerprint")
        )
        != seed_partition.fingerprint
    ):
        raise ValueError("gate-diagnostic seed-partition fingerprint mismatch")

    policy = BMRBOperatingStudyPolicy(
        study_id=_required_text("operating_policy.study_id", payload.get("study_id")),
        source_sha=_required_text("operating_policy.source_sha", payload.get("source_sha")),
        partition=_required_text("operating_policy.partition", payload.get("partition")),
        grid=grid,
        replicates_per_cell=_positive_int(
            "operating_policy.replicates_per_cell", payload.get("replicates_per_cell")
        ),
        bootstrap_resamples=_positive_int(
            "operating_policy.bootstrap_resamples", payload.get("bootstrap_resamples")
        ),
        primary_calibration_per_class=_nonnegative_int(
            "operating_policy.primary_calibration_per_class",
            payload.get("primary_calibration_per_class"),
        ),
        seed_partition=seed_partition,
    )
    claimed = _required_text("operating_policy.policy_fingerprint", payload.get("policy_fingerprint"))
    if claimed != policy.policy_fingerprint:
        raise ValueError("gate-diagnostic operating-policy fingerprint mismatch")
    return policy


def _expected_confusion(expected_pass: bool, trials: int, passes: int) -> dict[str, int]:
    if expected_pass:
        return {
            "true_positive": passes,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": trials - passes,
        }
    return {
        "true_positive": 0,
        "true_negative": trials - passes,
        "false_positive": passes,
        "false_negative": 0,
    }


def verify_bmrb_gate_diagnostics_mapping(payload: Mapping[str, Any]) -> None:
    """Verify fingerprints and reconstructible scientific invariants before reuse."""

    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("gate-diagnostic schema_version must be 1")
    if payload.get("benchmark") != BMRB_GATE_DIAGNOSTICS_BENCHMARK:
        raise ValueError("artifact is not BMRB gate diagnostics")
    if payload.get("method") != BMRB_GATE_DIAGNOSTICS_METHOD:
        raise ValueError("gate-diagnostic method mismatch")
    if payload.get("qualification_defined") is not False:
        raise ValueError("gate diagnostics must not invent universal qualification thresholds")

    claimed_artifact = _required_text("artifact_fingerprint", payload.get("artifact_fingerprint"))
    core = {key: value for key, value in payload.items() if key != "artifact_fingerprint"}
    expected_artifact = _scientific_fingerprint(
        "quantumbci.bmrb-gate-diagnostics-result.v1", core
    )
    if claimed_artifact != expected_artifact:
        raise ValueError("gate-diagnostic artifact fingerprint mismatch")

    policy_payload = _required_mapping("operating_policy", payload.get("operating_policy"))
    policy = _policy_from_mapping(policy_payload)
    if _required_text(
        "operating_policy_fingerprint", payload.get("operating_policy_fingerprint")
    ) != policy.policy_fingerprint:
        raise ValueError("root operating-policy fingerprint mismatch")

    scenarios = {scenario.scenario_id: scenario for scenario in default_validation_scenarios()}
    cells = _required_list("cells", payload.get("cells"))
    if len(cells) != policy.grid.cell_count:
        raise ValueError("gate diagnostics do not cover the complete frozen grid")
    combinations = tuple(
        product(
            policy.grid.scenario_ids,
            policy.grid.participant_counts,
            policy.grid.effect_scales,
            policy.grid.heterogeneity_scales,
            policy.grid.measurement_noise_scales,
        )
    )

    reconstructed_cells: list[dict[str, Any]] = []
    for index, (raw_cell, combination) in enumerate(zip(cells, combinations, strict=True)):
        cell = _required_mapping(f"cells[{index}]", raw_cell)
        scenario_id, participants, effect_scale, heterogeneity_scale, noise_scale = combination
        scenario = scenarios[scenario_id]
        if _nonnegative_int("cell_index", cell.get("cell_index")) != index:
            raise ValueError("gate-diagnostic cell indices are incomplete or out of order")
        if cell.get("scenario_id") != scenario_id:
            raise ValueError("gate-diagnostic scenario order differs from frozen grid")
        if cell.get("truth_class") != scenario.truth_class:
            raise ValueError("gate-diagnostic truth_class disagrees with declared DGM")
        if _strict_bool(
            "cell.expected_scientific_pass", cell.get("expected_scientific_pass")
        ) is not scenario.expected_scientific_pass:
            raise ValueError("cell expected scientific truth disagrees with DGM")
        if cell.get("expected_failure_component") != scenario.expected_failure_component:
            raise ValueError("cell expected failure component disagrees with DGM")
        if _positive_int("cell.participant_count", cell.get("participant_count")) != participants:
            raise ValueError("cell participant count differs from frozen grid")
        _close("cell.effect_scale", cell.get("effect_scale"), effect_scale)
        _close("cell.heterogeneity_scale", cell.get("heterogeneity_scale"), heterogeneity_scale)
        _close("cell.measurement_noise_scale", cell.get("measurement_noise_scale"), noise_scale)
        expected_seed = policy.seed_partition.base_seed(policy.partition, cell_index=index)
        if _positive_int("cell.base_seed", cell.get("base_seed")) != expected_seed:
            raise ValueError("cell base seed differs from frozen seed authority")
        trials = _positive_int("cell.replicates", cell.get("replicates"))
        if trials != policy.replicates_per_cell:
            raise ValueError("cell replicate count differs from operating policy")

        expected_gates = expected_gate_passes(scenario)
        gate_rows = _required_list("cell.gate_confusion", cell.get("gate_confusion"))
        if len(gate_rows) != len(GATE_ORDER):
            raise ValueError("cell must contain exactly one diagnostic for every gate")
        gate_map: dict[str, dict[str, Any]] = {}
        for gate_index, gate in enumerate(GATE_ORDER):
            gate_row = _required_mapping(f"cell.gate_confusion[{gate_index}]", gate_rows[gate_index])
            if gate_row.get("gate") != gate:
                raise ValueError("gate diagnostics must follow canonical gate order")
            if _strict_bool("gate.expected_pass", gate_row.get("expected_pass")) is not expected_gates[gate]:
                raise ValueError("gate expected truth disagrees with declared DGM")
            if _positive_int("gate.trials", gate_row.get("trials")) != trials:
                raise ValueError("gate trials differ from cell replicate count")
            passes = _nonnegative_int("gate.observed_passes", gate_row.get("observed_passes"))
            if passes > trials:
                raise ValueError("gate observed_passes exceeds trials")
            _close("gate.observed_pass_rate", gate_row.get("observed_pass_rate"), passes / trials)
            lower, upper = _wilson_interval(passes, trials)
            _close("gate.pass_rate_ci_lower", gate_row.get("pass_rate_ci_lower"), lower)
            _close("gate.pass_rate_ci_upper", gate_row.get("pass_rate_ci_upper"), upper)
            expected_counts = _expected_confusion(expected_gates[gate], trials, passes)
            for key, expected_count in expected_counts.items():
                if _nonnegative_int(f"gate.{key}", gate_row.get(key)) != expected_count:
                    raise ValueError(f"gate {key} disagrees with expected confusion accounting")
            gate_map[gate] = dict(gate_row)

        failure_counts = _required_mapping(
            "cell.first_failing_gate_counts", cell.get("first_failing_gate_counts")
        )
        allowed_failure_keys = set(GATE_ORDER) | {"none"}
        if set(failure_counts) != allowed_failure_keys:
            raise ValueError("first-failing-gate counts must use the canonical gate set plus none")
        normalized_counts = {
            key: _nonnegative_int(f"first_failing_gate_counts.{key}", failure_counts[key])
            for key in allowed_failure_keys
        }
        if sum(normalized_counts.values()) != trials:
            raise ValueError("first-failing-gate counts must partition all replicates")
        scientific_passes = _nonnegative_int(
            "cell.observed_scientific_passes", cell.get("observed_scientific_passes")
        )
        if scientific_passes != normalized_counts["none"]:
            raise ValueError("scientific pass count must equal first-failure none count")
        _close(
            "cell.observed_scientific_pass_rate",
            cell.get("observed_scientific_pass_rate"),
            scientific_passes / trials,
        )
        expected_first = "none" if scenario.expected_failure_component is None else scenario.expected_failure_component
        localization = normalized_counts[expected_first] / trials
        _close(
            "cell.first_failure_localization_rate",
            cell.get("first_failure_localization_rate"),
            localization,
        )
        reconstructed_cells.append(
            {
                "scenario": scenario,
                "trials": trials,
                "scientific_passes": scientific_passes,
                "gate_map": gate_map,
                "failure_counts": normalized_counts,
                "localization": localization,
            }
        )

    aggregate = _required_mapping("aggregate", payload.get("aggregate"))
    if _nonnegative_int(
        "aggregate.software_invalid_trials_in_gate_confusion",
        aggregate.get("software_invalid_trials_in_gate_confusion"),
    ) != 0:
        raise ValueError("software-invalid trials must be excluded from scientific gate confusion")
    _required_text("aggregate.software_invalid_handling", aggregate.get("software_invalid_handling"))

    aggregate_gates = _required_mapping("aggregate.gate_confusion", aggregate.get("gate_confusion"))
    for gate in GATE_ORDER:
        gate_payload = _required_mapping(f"aggregate.gate_confusion.{gate}", aggregate_gates.get(gate))
        tp = sum(item["gate_map"][gate]["true_positive"] for item in reconstructed_cells)
        tn = sum(item["gate_map"][gate]["true_negative"] for item in reconstructed_cells)
        fp = sum(item["gate_map"][gate]["false_positive"] for item in reconstructed_cells)
        fn = sum(item["gate_map"][gate]["false_negative"] for item in reconstructed_cells)
        expected_values = {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "expected_pass_support": tp + fn,
            "expected_failure_support": tn + fp,
        }
        for key, expected_value in expected_values.items():
            if _nonnegative_int(f"aggregate.{gate}.{key}", gate_payload.get(key)) != expected_value:
                raise ValueError("aggregate gate confusion does not match cell evidence")
        rates = {
            "pass_sensitivity": _safe_rate(tp, tp + fn),
            "failure_specificity": _safe_rate(tn, tn + fp),
            "false_pass_rate": _safe_rate(fp, tn + fp),
            "false_fail_rate": _safe_rate(fn, tp + fn),
        }
        for key, expected_value in rates.items():
            observed_value = gate_payload.get(key)
            if expected_value is None:
                if observed_value is not None:
                    raise ValueError(f"aggregate {gate}.{key} must be null without truth support")
            else:
                _close(f"aggregate.{gate}.{key}", observed_value, expected_value)

    total_first = {gate: 0 for gate in GATE_ORDER}
    total_first["none"] = 0
    false_escape = {gate: 0 for gate in GATE_ORDER}
    positive_loss = {gate: 0 for gate in GATE_ORDER}
    for item in reconstructed_cells:
        scenario = item["scenario"]
        for key, count in item["failure_counts"].items():
            total_first[key] += count
        if scenario.expected_scientific_pass:
            for gate in GATE_ORDER:
                positive_loss[gate] += item["failure_counts"][gate]
        elif scenario.expected_failure_component is not None:
            false_escape[scenario.expected_failure_component] += item["scientific_passes"]

    for field_name, expected_mapping in (
        ("first_failing_gate_counts", total_first),
        ("false_promotion_escape_counts_by_expected_gate", false_escape),
        ("known_positive_loss_first_gate_counts", positive_loss),
    ):
        observed_mapping = _required_mapping(f"aggregate.{field_name}", aggregate.get(field_name))
        if set(observed_mapping) != set(expected_mapping):
            raise ValueError(f"aggregate {field_name} has unexpected keys")
        for key, value in expected_mapping.items():
            if _nonnegative_int(f"aggregate.{field_name}.{key}", observed_mapping[key]) != value:
                raise ValueError(f"aggregate {field_name} does not match cell evidence")
    _close(
        "aggregate.mean_first_failure_localization_rate",
        aggregate.get("mean_first_failure_localization_rate"),
        float(np.mean([item["localization"] for item in reconstructed_cells])),
    )
    _required_text("interpretation", payload.get("interpretation"))


def load_bmrb_gate_diagnostics(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("gate-diagnostic artifact must contain a JSON object")
    verify_bmrb_gate_diagnostics_mapping(payload)
    return payload
