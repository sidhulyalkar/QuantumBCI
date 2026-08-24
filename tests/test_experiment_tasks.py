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


def test_e002_synthetic_and_identifiability_stages_are_executable(
    tmp_path: Path,
    capsys,
) -> None:
    recovery = tmp_path / "synthetic_recovery.json"
    gate = tmp_path / "identifiability_gate.json"

    assert main(
        [
            "synthetic-recovery",
            "E002",
            "--seed",
            "2027",
            "--noise-std",
            "0.003",
            "--output",
            str(recovery),
        ]
    ) == 0
    recovery_stdout = json.loads(capsys.readouterr().out)
    recovery_payload = json.loads(recovery.read_text())
    assert recovery_stdout["status"] == "pass"
    assert recovery_payload["synthetic_identifiability_gate_pass"] is True
    assert recovery_payload["affine_equivalence_pass"] is True
    assert recovery_payload["gauge_nonidentifiability_witness_pass"] is True
    assert recovery_payload["canonical_structure_pass"] is True
    assert recovery_payload["max_canonical_structure_residual"] <= 0.05
    assert recovery_payload["classical_adversary"]["rejected_as_canonical_family"] is True
    assert recovery_payload["classical_adversary"]["canonical_structure_residual"] >= 0.10
    assert recovery_payload["dynamical_information_novel"] is False

    assert main(
        [
            "gate",
            "E002",
            "identifiability",
            "--input",
            str(recovery),
            "--output",
            str(gate),
        ]
    ) == 0
    gate_stdout = json.loads(capsys.readouterr().out)
    gate_payload = json.loads(gate.read_text())
    assert gate_stdout["status"] == "pass"
    assert gate_payload["trajectory_contract_stage_eligible"] is True
    assert gate_payload["dynamical_information_novel"] is False
    assert gate_payload["physical_quantum_promotion_eligible"] is False
    assert gate_payload["observed"]["median_normalized_recovery_error"] <= 0.20
    assert gate_payload["observed"]["max_canonical_structure_residual"] <= 0.05
    assert gate_payload["observed"]["classical_adversary_structure_residual"] >= 0.10
    assert gate_payload["observed"]["classical_adversary_rejected"] is True


def test_e002_gate_independently_rejects_missing_family_specificity(
    tmp_path: Path,
    capsys,
) -> None:
    recovery = tmp_path / "synthetic_recovery.json"
    gate = tmp_path / "identifiability_gate.json"
    assert main(
        [
            "synthetic-recovery",
            "E002",
            "--output",
            str(recovery),
        ]
    ) == 0
    capsys.readouterr()

    payload = json.loads(recovery.read_text())
    # Simulate an artifact that still claims parameter recovery but no longer rejects
    # a stable noncanonical affine look-alike. The downstream gate must not trust only
    # the source artifact's summary boolean.
    payload["classical_adversary"]["canonical_structure_residual"] = 0.01
    payload["classical_adversary"]["rejected_as_canonical_family"] = False
    recovery.write_text(json.dumps(payload), encoding="utf-8")

    assert main(
        [
            "gate",
            "E002",
            "identifiability",
            "--input",
            str(recovery),
            "--output",
            str(gate),
        ]
    ) == 2
    stdout = json.loads(capsys.readouterr().out)
    artifact = json.loads(gate.read_text())
    assert stdout["status"] == "fail"
    assert artifact["trajectory_contract_stage_eligible"] is False
    assert artifact["observed"]["classical_adversary_rejected"] is False


def test_e002_identifiability_gate_fails_closed_without_recovery_artifact(
    tmp_path: Path,
    capsys,
) -> None:
    assert main(
        [
            "gate",
            "E002",
            "identifiability",
            "--input",
            str(tmp_path / "missing.json"),
        ]
    ) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert "not found" in payload["message"]


def test_unimplemented_manifest_task_still_fails_closed(capsys) -> None:
    assert main(["extract-embeddings", "E001"]) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "not_implemented"

    assert main(["trajectory-contract", "E002"]) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "not_implemented"
