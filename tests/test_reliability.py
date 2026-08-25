from __future__ import annotations

import numpy as np
import pytest

from quantumbci.reliability import (
    ICC_METHOD_ID,
    RepeatedCaseEstimate,
    audit_repeated_case_estimate,
    audit_repeated_case_reliability,
    estimates_from_stability_artifact,
)


def _row(participant: str, occasion: str, value: float, *, estimate: str = "omega_x") -> RepeatedCaseEstimate:
    return RepeatedCaseEstimate(
        participant_id=participant,
        occasion_id=occasion,
        case_id=f"{participant}-{occasion}",
        estimate_name=estimate,
        value=value,
        authority_fingerprint=f"authority-{participant}-{occasion}",
        data_sha256=f"data-{participant}-{occasion}",
        artifact_sha256=f"artifact-{participant}-{occasion}",
    )


def test_balanced_repeated_panel_computes_icc_and_population_recurrence() -> None:
    rows = []
    participant_values = {
        "p1": (1.00, 1.04),
        "p2": (1.70, 1.66),
        "p3": (2.50, 2.55),
        "p4": (3.30, 3.26),
    }
    for participant, values in participant_values.items():
        rows.append(_row(participant, "session-1", values[0]))
        rows.append(_row(participant, "session-2", values[1]))

    result = audit_repeated_case_estimate(rows, n_resamples=400, seed=7)

    assert result.balanced_panel is True
    assert result.icc is not None
    assert result.icc.method_id == ICC_METHOD_ID
    assert result.icc.value > 0.95
    assert result.population_sign_consistency == 1.0
    assert result.participant_positive_fraction == 1.0
    assert result.bootstrap_ci_low > 0.0
    assert result.reliability_gate_defined if hasattr(result, "reliability_gate_defined") else True
    mapping = result.to_mapping()
    assert mapping["reliability_gate_defined"] is False
    assert mapping["reliability_gate_pass"] is None


def test_unbalanced_panel_keeps_population_inference_but_refuses_icc() -> None:
    rows = [
        _row("p1", "session-1", 1.0),
        _row("p1", "session-2", 1.1),
        _row("p2", "session-1", 1.2),
        _row("p3", "session-1", 0.9),
        _row("p3", "session-2", 1.0),
    ]
    result = audit_repeated_case_estimate(rows, n_resamples=300, seed=11)

    assert result.balanced_panel is False
    assert result.icc is None
    assert result.icc_unavailable_reason is not None
    assert "balanced" in result.icc_unavailable_reason
    assert result.bootstrap_ci_low > 0.0
    assert result.population_sign_consistency == 1.0


def test_icc_and_population_recurrence_are_distinct_surfaces() -> None:
    rows = [
        _row("p1", "session-1", 1.00),
        _row("p1", "session-2", 1.04),
        _row("p2", "session-1", 1.01),
        _row("p2", "session-2", 0.97),
        _row("p3", "session-1", 1.02),
        _row("p3", "session-2", 1.05),
        _row("p4", "session-1", 0.99),
        _row("p4", "session-2", 1.02),
    ]
    result = audit_repeated_case_estimate(rows, n_resamples=300, seed=19)

    assert result.population_sign_consistency == 1.0
    assert result.bootstrap_ci_low > 0.0
    assert result.icc is not None
    # Individual-difference reliability can be modest or negative when true
    # between-participant variance is tiny even though the mechanism recurs.
    assert result.icc.value < 0.8


def test_stability_artifact_extraction_preserves_authority_and_refuses_fake_icc() -> None:
    artifact = {
        "experiment": "E002",
        "artifact_role": "bootstrap_stability_evidence",
        "status": "pass",
        "evaluation_resampled": False,
        "single_case_bootstrap_is_icc": False,
        "authority_fingerprint": "authority-a",
        "data_sha256": "data-a",
        "point_estimates": {"omega_x": 0.7, "gamma_relaxation": 0.2},
    }
    rows = estimates_from_stability_artifact(
        artifact,
        participant_id="p1",
        occasion_id="s1",
        case_id="case-1",
        artifact_sha256="artifact-a",
        estimate_names=("omega_x", "gamma_relaxation"),
    )
    assert [row.estimate_name for row in rows] == ["omega_x", "gamma_relaxation"]
    assert all(row.authority_fingerprint == "authority-a" for row in rows)

    artifact["single_case_bootstrap_is_icc"] = True
    with pytest.raises(ValueError, match="single-case bootstrap as ICC"):
        estimates_from_stability_artifact(
            artifact,
            participant_id="p1",
            occasion_id="s1",
            case_id="case-1",
        )


def test_bundle_groups_estimates_without_mixing_scales() -> None:
    rows = []
    for participant, base in (("p1", 1.0), ("p2", 1.5), ("p3", 2.0)):
        for occasion, shift in (("s1", 0.0), ("s2", 0.05)):
            rows.append(_row(participant, occasion, base + shift, estimate="omega_x"))
            rows.append(_row(participant, occasion, 0.2 + 0.01 * base + shift / 10, estimate="gamma_relaxation"))

    bundle = audit_repeated_case_reliability(
        rows,
        study_id="study-a",
        n_resamples=200,
        seed=23,
    )
    assert bundle.estimate_names == ("gamma_relaxation", "omega_x")
    assert bundle.case_count == 6
    assert bundle.participant_count == 3
    assert len(bundle.results) == 2
    assert bundle.results[0].source_fingerprint != bundle.results[1].source_fingerprint


def test_duplicate_participant_occasion_estimate_fails_closed() -> None:
    row = _row("p1", "s1", 1.0)
    with pytest.raises(ValueError, match="duplicate participant/occasion"):
        audit_repeated_case_estimate([row, row], n_resamples=100)
