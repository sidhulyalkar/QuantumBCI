from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from quantumbci.audit_cli import main


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    rng = np.random.default_rng(2026)
    embeddings = []
    labels = []
    t = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    for index in range(60):
        label = index % 2
        a = np.sin(t + 0.05 * index)
        sign = 1.0 if label else -1.0
        values = np.stack(
            [a, sign * a, np.cos(2 * t), np.sin(3 * t)],
            axis=1,
        )
        values += rng.normal(0.0, 0.02, size=values.shape)
        embeddings.append(values)
        labels.append(label)
    paths = (
        tmp_path / "embeddings.npy",
        tmp_path / "labels.npy",
        tmp_path / "train.npy",
        tmp_path / "test.npy",
    )
    np.save(paths[0], np.stack(embeddings))
    np.save(paths[1], np.asarray(labels))
    np.save(paths[2], np.arange(44))
    np.save(paths[3], np.arange(44, 60))
    return paths


def test_density_audit_cli_reports_exact_classical_equivalence(tmp_path: Path, capsys) -> None:
    embeddings, _, _, _ = _fixture(tmp_path)
    assert main(["density", str(embeddings), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["equivalent_within_tolerance"] is True
    assert payload["novel_information"] is False
    assert payload["promotion_eligible_as_new_information"] is False
    assert payload["equivalence_class"] == "trace_normalized_hermitian_second_moment"


def test_e001_audit_cli_runs_control_gauntlet_and_writes_output(tmp_path: Path, capsys) -> None:
    embeddings, labels, train, test = _fixture(tmp_path)
    output = tmp_path / "e001.json"
    assert main(
        [
            "e001",
            str(embeddings),
            str(labels),
            "--train-indices",
            str(train),
            "--test-indices",
            str(test),
            "--output",
            str(output),
            "--json",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert output.is_file()
    assert payload["density_information_novel"] is False
    assert payload["equivalence_audit"]["equivalent_within_tolerance"] is True
    assert "normalized_covariance" in payload["metrics"]
    assert "log_covariance" in payload["metrics"]
    assert "pca_flattened" in payload["metrics"]
    assert payload["strongest_classical_control"] in payload["metrics"]
