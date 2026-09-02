from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean

import pytest

from quantumbci.bmrb_study_operating_artifacts import verify_bmrb_study_operating_mapping

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "bmrb-study-development-v1"
EXPECTED_ARTIFACT_FINGERPRINT = (
    "53e18166c3bbf071e929d13d79e1eef09d9046d9a99d49536d728a4c0ff36879"
)
EXPECTED_SCIENCE_SOURCE = "681ea12c436fce121ba74de6f877a8267e94dd3f"


def _load(name: str) -> dict:
    payload = json.loads((EVIDENCE / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _mean_rate(cells: list[dict], scenario_ids: set[str]) -> float:
    rows = [cell for cell in cells if cell["scenario_id"] in scenario_ids]
    assert rows
    return float(mean(float(cell["observed_replication_pass_rate"]) for cell in rows))


def test_complete_development_result_is_exact_verified_development_evidence() -> None:
    result = _load("development-result.json")
    verify_bmrb_study_operating_mapping(result)

    assert result["artifact_fingerprint"] == EXPECTED_ARTIFACT_FINGERPRINT
    assert result["policy"]["source_sha"] == EXPECTED_SCIENCE_SOURCE
    assert result["policy"]["partition"] == "development"
    assert result["policy"]["replicates_per_cell"] == 8
    assert result["policy"]["bootstrap_resamples"] == 100
    assert len(result["cells"]) == 648
    assert result["qualification_defined"] is False
    assert result["evaluation_partition_executed"] is False
    assert result["physical_quantum_promotion_eligible"] is False


def test_persisted_manifest_binds_every_capsule_file() -> None:
    manifest = _load("sha256-manifest.json")
    expected = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(EVIDENCE.iterdir())
        if path.is_file() and path.name != "sha256-manifest.json"
    }
    assert manifest == expected


def test_applicability_aware_estimands_are_recomputed_from_cells() -> None:
    result = _load("development-result.json")
    analysis = _load("development-analysis.json")
    cells = result["cells"]
    estimands = analysis["estimand_classes"]

    assert analysis["artifact_fingerprint"] == result["artifact_fingerprint"]
    assert analysis["policy_fingerprint"] == result["policy"]["policy_fingerprint"]
    assert analysis["cell_count"] == 648

    assert estimands["pure_null_broad_promotion_rate"] == pytest.approx(
        _mean_rate(cells, {"homogeneous-null-3", "homogeneous-null-4"})
    )
    assert estimands["homogeneous_positive_broad_recovery_rate"] == pytest.approx(
        _mean_rate(cells, {"homogeneous-positive-3", "homogeneous-positive-4"})
    )
    assert estimands["contextual_or_failed_primary_broad_promotion_rate"] == pytest.approx(
        _mean_rate(
            cells,
            {"primary-only-positive-4", "primary-fail-replications-positive-4"},
        )
    )
    assert estimands["conflicted_positive_broad_recovery_rate"] == pytest.approx(
        _mean_rate(cells, {"fragile-one-conflict-4", "redundant-one-conflict-5"})
    )

    primary_fail = [
        cell
        for cell in cells
        if cell["scenario_id"] == "primary-fail-replications-positive-4"
    ]
    fragile = [cell for cell in cells if cell["scenario_id"] == "fragile-one-conflict-4"]
    assert estimands["failed_primary_role_protection_rate"] == pytest.approx(
        mean(float(cell["primary_role_protection_rate"]) for cell in primary_fail)
    )
    assert estimands["fragile_conflict_detection_rate"] == pytest.approx(
        mean(float(cell["fragile_claim_detection_rate"]) for cell in fragile)
    )


def test_analysis_does_not_average_non_applicable_diagnostics_as_perfect_scores() -> None:
    analysis = _load("development-analysis.json")
    by_scenario = analysis["by_scenario"]

    for summary in by_scenario.values():
        assert "primary_role_protection_rate" not in summary
        assert "fragile_claim_detection_rate" not in summary

    assert "by_cross_study_effect_scale" not in analysis
    assert "by_scenario_and_cross_study_effect_scale" in analysis
    applicability = analysis["metric_applicability"]
    assert "non-applicable" in applicability["primary_role_protection_rate"]
    assert "non-applicable" in applicability["fragile_claim_detection_rate"]
    assert "all-null" in applicability["cross_study_effect_scale"]


def test_null_warning_semantics_are_preserved_as_a_v1_problem_not_hidden() -> None:
    analysis = _load("development-analysis.json")
    by_scenario = analysis["by_scenario"]

    assert by_scenario["homogeneous-null-3"]["observed_replication_pass_rate"] == 0.0
    assert by_scenario["homogeneous-null-4"]["observed_replication_pass_rate"] == 0.0
    assert by_scenario["homogeneous-null-3"]["sensitivity_warning_match_rate"] == pytest.approx(
        0.24845679012345678
    )
    assert by_scenario["homogeneous-null-4"]["sensitivity_warning_match_rate"] == pytest.approx(
        0.5154320987654321
    )

    warning_note = analysis["metric_applicability"]["sensitivity_warning_under_null"]
    assert "observed primary sign" in warning_note
    assert "not well posed" in warning_note


def test_coarse_rate_resolution_is_explicit_and_not_precision_calibration() -> None:
    analysis = _load("development-analysis.json")
    coarse = analysis["coarse_rate_resolution"]

    assert coarse["step"] == 0.125
    assert coarse["wilson_95_for_0_of_8"][1] > 0.3
    assert coarse["wilson_95_for_8_of_8"][0] < 0.7
    assert "do not provide precision" in coarse["interpretation"]


def test_legacy_aggregate_is_retained_but_explicitly_not_type_i_error_or_power() -> None:
    result = _load("development-result.json")
    analysis = _load("development-analysis.json")
    legacy = analysis["legacy_aggregate_mapping"]

    assert legacy["values"] == result["aggregate"]
    assert "not classical Type-I error" in legacy["interpretation"]
    assert "power estimands" in legacy["interpretation"]


def test_evidence_readme_keeps_claim_ceiling_below_biology_and_final_evaluation() -> None:
    text = (EVIDENCE / "README.md").read_text(encoding="utf-8")
    assert "final evaluation executed: **false**" in text
    assert "qualification defined: **false**" in text
    assert "physical-quantum promotion eligible: **false**" in text
    assert "does not define final" in text
    assert "establish biological truth" in text
