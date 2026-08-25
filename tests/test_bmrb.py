from __future__ import annotations

import json
from pathlib import Path

from quantumbci.bmrb import (
    DEFAULT_E002_RELIABILITY_ESTIMATES,
    build_bmrb_dynamics_bundle,
    write_bmrb_dynamics_bundle,
)
from quantumbci.recapitulation import EvidenceTier


def _artifact(participant_index: int, occasion_index: int) -> dict:
    base = 0.2 * participant_index
    point_estimates = {
        "omega_x": 0.7 + base + 0.01 * occasion_index,
        "omega_z": -0.5 - base - 0.01 * occasion_index,
        "gamma_dephasing": 0.18 + 0.02 * participant_index + 0.005 * occasion_index,
        "gamma_relaxation": 0.25 + 0.03 * participant_index + 0.004 * occasion_index,
        "canonical_structure_residual": 0.08 + 0.01 * participant_index,
        "canonical_minus_affine_one_step_rmse": 0.03 + 0.002 * occasion_index,
        "canonical_minus_affine_rollout_rmse": 0.05 + 0.003 * occasion_index,
        "direct_minus_nonlinear_mean_nll": 0.02 + 0.004 * participant_index,
        "direct_minus_nonlinear_one_step_rmse": 0.01 + 0.002 * participant_index,
    }
    return {
        "schema_version": 2,
        "experiment": "E002",
        "claim_class": "quantum_inspired",
        "artifact_role": "bootstrap_stability_evidence",
        "status": "pass",
        "evaluation_resampled": False,
        "single_case_bootstrap_is_icc": False,
        "participant_icc_computed": False,
        "stability_gate_defined": False,
        "stability_gate_pass": None,
        "predictive_adversary_ladder_complete": True,
        "dynamical_information_novel": False,
        "authority_fingerprint": f"authority-{participant_index}-{occasion_index}",
        "data_sha256": f"data-{participant_index}-{occasion_index}",
        "point_estimates": point_estimates,
    }


def _manifest(tmp_path: Path) -> Path:
    cases = []
    for participant in range(1, 4):
        for occasion in range(1, 3):
            filename = f"p{participant}-s{occasion}.json"
            (tmp_path / filename).write_text(
                json.dumps(_artifact(participant, occasion)), encoding="utf-8"
            )
            cases.append(
                {
                    "participant_id": f"p{participant}",
                    "occasion_id": f"session-{occasion}",
                    "case_id": f"p{participant}-session-{occasion}",
                    "artifact": filename,
                }
            )
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "study_id": "synthetic-bmrb-dynamics",
                "metadata": {"fixture": "v015"},
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_bmrb_dynamics_builds_reliability_and_falsification_profile(tmp_path: Path) -> None:
    bundle = build_bmrb_dynamics_bundle(
        _manifest(tmp_path),
        n_resamples=200,
        seed=31,
    )

    assert bundle.study_id == "synthetic-bmrb-dynamics"
    assert len(bundle.case_specs) == 6
    assert bundle.reliability.participant_count == 3
    assert bundle.reliability.estimate_names == tuple(sorted(DEFAULT_E002_RELIABILITY_ESTIMATES))
    assert all(result.icc is not None for result in bundle.reliability.results)
    assert bundle.profile.evidence_coverage_tier == EvidenceTier.REPEATED_CASE
    assert bundle.profile.promotion_ceiling == EvidenceTier.DESCRIPTIVE
    assert bundle.profile.first_failing_gate == "matched_classical_adversaries"
    mapping = bundle.to_mapping()
    assert mapping["mechanism_profile"]["necessity_claim_permitted"] is False
    assert mapping["claim_ceiling"] == "quantum_inspired"


def test_bmrb_writes_standalone_machine_and_html_reports(tmp_path: Path) -> None:
    bundle = build_bmrb_dynamics_bundle(
        _manifest(tmp_path),
        estimate_names=("omega_x", "gamma_relaxation"),
        n_resamples=150,
        seed=37,
    )
    json_path, html_path = write_bmrb_dynamics_bundle(bundle, tmp_path / "out")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    html = html_path.read_text(encoding="utf-8")
    assert payload["artifact_role"] == "bmrb_dynamics_bundle"
    assert payload["reliability"]["artifact_role"] == "repeated_case_reliability_evidence"
    assert "Mechanism necessity ladder" in html
    assert "Repeated-case mechanism quantities" in html
    assert "matched_classical_adversaries" in html
    assert bundle.source_fingerprint in html


def test_bmrb_refuses_duplicate_participant_occasion(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(_artifact(1, 1)), encoding="utf-8")
    manifest_path = tmp_path / "bad.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "study_id": "bad",
                "cases": [
                    {
                        "participant_id": "p1",
                        "occasion_id": "s1",
                        "case_id": "case-a",
                        "artifact": "artifact.json",
                    },
                    {
                        "participant_id": "p1",
                        "occasion_id": "s1",
                        "case_id": "case-b",
                        "artifact": "artifact.json",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        build_bmrb_dynamics_bundle(manifest_path, n_resamples=100)
    except ValueError as exc:
        assert "one case artifact per participant/occasion" in str(exc)
    else:
        raise AssertionError("duplicate participant/occasion should fail closed")
