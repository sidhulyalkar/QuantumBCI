from __future__ import annotations

import json
from pathlib import Path

from quantumbci.experiments.tasks import main


def test_e001_equivalence_stage_materializes_a_real_gate(tmp_path: Path, capsys) -> None:
    output = tmp_path / "equivalence_audit.json"
    assert main(
        [
            "equivalence-audit",
            "E001",
            "density-covariance",
            "--output",
            str(output),
        ]
    ) == 0
    stdout = json.loads(capsys.readouterr().out)
    artifact = json.loads(output.read_text())
    assert stdout["status"] == "pass"
    assert artifact["equivalent_within_tolerance"] is True
    assert artifact["representation_information_novel"] is False
    assert artifact["equivalence_class"] == "trace_normalized_hermitian_second_moment"
    assert {probe["probe"] for probe in artifact["probes"]} == {"real", "complex"}
    assert {probe["center"] for probe in artifact["probes"]} == {True, False}


def test_unimplemented_manifest_task_still_fails_closed(capsys) -> None:
    assert main(["extract-embeddings", "E001"]) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "not_implemented"
