from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from quantumbci.exporting import verify_run_artifacts
from quantumbci.studies import kumar2024 as study


fm = pytest.importorskip("neuros.foundation_models")
moabb_longitudinal = pytest.importorskip("neuros.foundation_models.moabb_longitudinal")


def _fixture(subject: int):
    GroupedEvaluationData = fm.GroupedEvaluationData
    rng = np.random.default_rng(1000 + int(subject))
    t = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    X = []
    y = []
    subjects = []
    sessions = []
    runs = []
    for session_index, session_id in enumerate(("0", "1", "2", "3", "4", "5")):
        for label_index, label in enumerate(("left_hand", "right_hand")):
            for trial in range(6):
                phase = 0.06 * trial + 0.025 * session_index
                latent = np.sin(t + phase)
                sign = 1.0 if label_index else -1.0
                epoch = np.stack(
                    [
                        latent,
                        sign * latent,
                        np.cos(2.0 * t + 0.02 * session_index),
                        np.sin(3.0 * t + phase),
                    ],
                    axis=0,
                )
                epoch += rng.normal(0.0, 0.025, size=epoch.shape)
                X.append(epoch.astype(np.float32))
                y.append(label)
                subjects.append(str(subject))
                sessions.append(session_id)
                runs.append(f"r-{trial // 3}")
    return GroupedEvaluationData(
        dataset_id="moabb-kumar2024",
        X=np.asarray(X),
        y=np.asarray(y),
        groups={
            "subject": np.asarray(subjects),
            "session": np.asarray(sessions),
            "run": np.asarray(runs),
        },
    )


def _expected_seed(subject: int, target: str) -> int:
    raw = "|".join(["2026", "moabb-kumar2024", str(subject), str(target)])
    return int.from_bytes(hashlib.sha256(raw.encode("utf-8")).digest()[:4], "big")


def test_kumar_subject_matches_merged_neuros_authority_semantics() -> None:
    spec = moabb_longitudinal.get_moabb_longitudinal_spec("kumar2024")
    config = study.Kumar2024StudyConfig(
        subjects=(1, 10),
        held_out_sessions=("5",),
        budgets_per_class=(0, 1, 2),
    )

    authorities, cases = study.run_kumar2024_subject(
        _fixture(1),
        spec,
        subject=1,
        config=config,
        upstream_dataset_fingerprint="f" * 64,
        quantumbci_source_sha="qbc-study-test",
        neuros_source_sha="neuros-study-test",
    )
    assert len(authorities) == 1
    assert len(cases) == 1
    authority = authorities[0]
    case = cases[0]
    expected_seed = _expected_seed(1, "5")

    assert authority.case_id == (
        f"moabb-kumar2024/subject-1/session-5/split-{expected_seed}"
    )
    assert authority.source_group_values == ("0", "1", "2", "3", "4")
    assert authority.held_out_values == ("5",)
    assert authority.case_metadata["subject"] == 1
    assert authority.case_metadata["original_protocol"] == "GR"
    assert authority.case_metadata["split_seed"] == expected_seed
    assert case.provenance["upstream_dataset_fingerprint"] == "f" * 64
    assert len(case.representation_sha256) == 64
    assert [row.calibration_samples for row in case.rows] == [0, 2, 4]
    for row in case.rows:
        assert row.result.equivalence_audit["equivalent_within_tolerance"] is True
        assert np.array_equal(
            row.result.predictions["density"],
            row.result.predictions["normalized_covariance"],
        )


def test_two_participant_kumar_bundle_is_closed_world_and_export_ready(tmp_path: Path) -> None:
    spec = moabb_longitudinal.get_moabb_longitudinal_spec("kumar2024")
    config = study.Kumar2024StudyConfig(
        subjects=(1, 10),
        held_out_sessions=("5",),
        budgets_per_class=(0, 1, 2),
    )
    authorities = []
    cases = []
    for subject in config.subjects:
        subject_authorities, subject_cases = study.run_kumar2024_subject(
            _fixture(subject),
            spec,
            subject=subject,
            config=config,
            upstream_dataset_fingerprint="a" * 64,
            quantumbci_source_sha="qbc-study-test",
            neuros_source_sha="neuros-study-test",
        )
        authorities.extend(subject_authorities)
        cases.extend(subject_cases)

    raw_fingerprint = {
        "schema_version": 1,
        "kind": "raw_source_content_fingerprint",
        "dataset_key": "kumar2024",
        "dataset_id": "moabb-kumar2024",
        "subjects": [1, 10],
        "by_subject": {},
        "fingerprint": "a" * 64,
    }
    output = tmp_path / "study"
    result = study._write_study_bundle(
        output,
        config=config,
        dataset_spec=spec,
        raw_fingerprint=raw_fingerprint,
        authorities=authorities,
        cases=cases,
        quantumbci_source_sha="qbc-study-test",
        neuros_source_sha="neuros-study-test",
        overwrite=False,
    )

    assert result["authority_cases"] == 2
    assert result["equivalence_promotion_eligible"] is False
    assert verify_run_artifacts(output)["valid"] is True
    assert (output / "run.json").is_file()
    assert (output / "dataset_fingerprint.json").is_file()
    assert (output / "neuros_authority.json").is_file()
    assert (output / "representation_index.json").is_file()
    assert (output / "predictions.jsonl").is_file()
    assert (output / "bootstrap_metrics.json").is_file()
    assert (output / "evidence_ledger.json").is_file()
    assert (output / "report.md").is_file()

    ledger = json.loads((output / "evidence_ledger.json").read_text())
    assert ledger["information_novelty_promotion_eligible"] is False
    bootstrap = json.loads((output / "bootstrap_metrics.json").read_text())
    normalized = bootstrap["controls"]["normalized_covariance"]["summaries"]
    assert all(abs(item["observed_mean_delta"]) < 1e-12 for item in normalized)
    assert all(abs(item["ci_lower"]) < 1e-12 for item in normalized)
    assert all(abs(item["ci_upper"]) < 1e-12 for item in normalized)

    with (output / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["subject"] for row in rows} == {"1", "10"}
    assert {row["original_protocol"] for row in rows} == {"GR", "PAR"}
    assert "normalized_covariance" in {row["method"] for row in rows}
