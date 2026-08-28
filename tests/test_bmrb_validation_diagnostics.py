from __future__ import annotations

from copy import deepcopy

import pytest

from quantumbci.bmrb_validation_diagnostics import (
    BMRB_GATE_DIAGNOSTICS_BENCHMARK,
    GATE_ORDER,
    _scientific_fingerprint,
    load_bmrb_gate_diagnostics,
    run_bmrb_gate_diagnostics,
    verify_bmrb_gate_diagnostics_mapping,
    write_bmrb_gate_diagnostics,
)
from quantumbci.bmrb_validation_operating import (
    BMRBOperatingStudyPolicy,
    OperatingCurveGrid,
)


def diagnostic_policy() -> BMRBOperatingStudyPolicy:
    return BMRBOperatingStudyPolicy(
        study_id="gate-diagnostics-ci",
        source_sha="test-source-sha",
        partition="development",
        grid=OperatingCurveGrid(
            scenario_ids=(
                "effect-null",
                "equivalence-null",
                "predictive-shortcut",
                "coverage-insufficient-family-support",
                "shared-mechanism-positive",
            ),
            participant_counts=(8,),
            effect_scales=(1.0,),
            heterogeneity_scales=(1.0,),
            measurement_noise_scales=(1.0,),
        ),
        replicates_per_cell=4,
        bootstrap_resamples=100,
    )


def _refingerprint(payload: dict) -> None:
    core = {key: value for key, value in payload.items() if key != "artifact_fingerprint"}
    payload["artifact_fingerprint"] = _scientific_fingerprint(
        "quantumbci.bmrb-gate-diagnostics-result.v1",
        core,
    )


def test_gate_diagnostics_localize_declared_failure_modes() -> None:
    result = run_bmrb_gate_diagnostics(diagnostic_policy())
    payload = result.to_mapping()

    assert payload["benchmark"] == BMRB_GATE_DIAGNOSTICS_BENCHMARK
    assert payload["qualification_defined"] is False
    cells = {cell["scenario_id"]: cell for cell in payload["cells"]}

    assert cells["effect-null"]["first_failing_gate_counts"]["effect"] == 4
    assert cells["equivalence-null"]["first_failing_gate_counts"]["adversary"] == 4
    assert cells["predictive-shortcut"]["first_failing_gate_counts"]["conservation"] == 4
    assert cells["coverage-insufficient-family-support"]["first_failing_gate_counts"][
        "coverage"
    ] == 4
    assert cells["shared-mechanism-positive"]["first_failing_gate_counts"]["none"] == 4
    assert all(cell["first_failure_localization_rate"] == 1.0 for cell in cells.values())


def test_gate_confusion_measures_all_four_gate_truth_classes() -> None:
    result = run_bmrb_gate_diagnostics(diagnostic_policy())
    gates = result.to_mapping()["aggregate"]["gate_confusion"]

    for gate in GATE_ORDER:
        assert gates[gate]["pass_sensitivity"] == pytest.approx(1.0)
        assert gates[gate]["failure_specificity"] == pytest.approx(1.0)
        assert gates[gate]["false_pass_rate"] == pytest.approx(0.0)
        assert gates[gate]["false_fail_rate"] == pytest.approx(0.0)
        assert gates[gate]["expected_pass_support"] > 0
        assert gates[gate]["expected_failure_support"] > 0

    coverage = gates["coverage"]
    assert coverage["expected_failure_support"] == 4
    assert coverage["true_negative"] == 4


def test_gate_diagnostics_decompose_false_promotion_and_positive_loss_paths() -> None:
    aggregate = run_bmrb_gate_diagnostics(diagnostic_policy()).to_mapping()["aggregate"]
    assert aggregate["false_promotion_escape_counts_by_expected_gate"] == {
        gate: 0 for gate in GATE_ORDER
    }
    assert aggregate["known_positive_loss_first_gate_counts"] == {
        gate: 0 for gate in GATE_ORDER
    }
    assert aggregate["software_invalid_trials_in_gate_confusion"] == 0
    assert "excluded" in aggregate["software_invalid_handling"]


def test_gate_diagnostic_artifact_round_trip_is_verified(tmp_path) -> None:
    result = run_bmrb_gate_diagnostics(diagnostic_policy())
    output = write_bmrb_gate_diagnostics(result, tmp_path / "diagnostics.json")
    loaded = load_bmrb_gate_diagnostics(output)
    assert loaded["artifact_fingerprint"] == result.artifact_fingerprint
    assert loaded["operating_policy_fingerprint"] == result.policy.policy_fingerprint


def test_gate_diagnostic_verifier_rejects_semantic_tampering_after_outer_refingerprint() -> None:
    payload = run_bmrb_gate_diagnostics(diagnostic_policy()).to_mapping()
    tampered = deepcopy(payload)
    cell = tampered["cells"][0]
    cell["first_failing_gate_counts"]["effect"] = 3
    cell["first_failing_gate_counts"]["none"] = 1
    cell["observed_scientific_passes"] = 1
    cell["observed_scientific_pass_rate"] = 0.25
    _refingerprint(tampered)
    with pytest.raises(ValueError, match="first_failure_localization_rate"):
        verify_bmrb_gate_diagnostics_mapping(tampered)


def test_gate_diagnostic_verifier_rejects_policy_switch_even_when_outer_hash_is_fresh() -> None:
    payload = run_bmrb_gate_diagnostics(diagnostic_policy()).to_mapping()
    tampered = deepcopy(payload)
    tampered["operating_policy"]["partition"] = "evaluation"
    _refingerprint(tampered)
    with pytest.raises(ValueError, match="operating-policy fingerprint"):
        verify_bmrb_gate_diagnostics_mapping(tampered)
