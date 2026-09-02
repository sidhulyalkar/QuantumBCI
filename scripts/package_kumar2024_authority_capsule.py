#!/usr/bin/env python3
"""Package a successful Kumar2024 authority-freeze artifact for durable review."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from quantumbci.kumar2024_authority_artifacts import (
    KUMAR2024_AUTHORITY_CAPSULE_DOMAIN,
    KUMAR2024_AUTHORITY_CAPSULE_ROLE,
    KUMAR2024_SUBJECTS,
    verify_kumar2024_authority_capsule_mapping,
)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def pretty_sha256(value: object) -> str:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return sha256(raw).hexdigest()


def canonical_sha256(value: object) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workflow-run-id", type=int, required=True)
    parser.add_argument("--workflow-head-sha", required=True)
    parser.add_argument("--artifact-id", type=int, required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--artifact-zip-digest", required=True)
    args = parser.parse_args()

    source = args.input
    freeze = load_json(source / "authority-freeze.json")
    raw = load_json(source / "raw-source-fingerprint.json")
    original_manifest = load_json(source / "sha256-manifest.json")
    authorities = [
        load_json(source / f"authority-subject-{subject}-session-5.json")
        for subject in KUMAR2024_SUBJECTS
    ]

    semantic_files: dict[str, object] = {
        "authority-freeze.json": freeze,
        "raw-source-fingerprint.json": raw,
        **{
            f"authority-subject-{subject}-session-5.json": authority
            for subject, authority in zip(KUMAR2024_SUBJECTS, authorities, strict=True)
        },
    }
    reproduced_manifest = {
        name: pretty_sha256(document)
        for name, document in sorted(semantic_files.items())
    }
    if reproduced_manifest != original_manifest:
        raise RuntimeError(
            "downloaded source artifact does not reproduce its own SHA-256 manifest"
        )

    base: dict[str, object] = {
        "schema_version": 1,
        "artifact_role": KUMAR2024_AUTHORITY_CAPSULE_ROLE,
        "purpose": "persisted_outcome_blind_design_and_preregistration_authority",
        "source_artifact": {
            "repository": "sidhulyalkar/QuantumBCI",
            "workflow_run_id": args.workflow_run_id,
            "workflow_head_sha": args.workflow_head_sha,
            "artifact_id": args.artifact_id,
            "artifact_name": args.artifact_name,
            "artifact_zip_digest": args.artifact_zip_digest,
            "producing_workflow_conclusion": "success",
            "producing_workflow_outcome_blind_guard_passed": True,
            "original_file_sha256_manifest": original_manifest,
        },
        "authority_freeze": freeze,
        "raw_source_fingerprint": raw,
        "subject_authorities": authorities,
        "canonical_component_sha256": {
            name: canonical_sha256(document)
            for name, document in sorted(semantic_files.items())
        },
        "claim_boundary": {
            "dataset_structure_inspected": True,
            "class_labels_used_for_stratified_split_construction": True,
            "evidence_assignment_frozen": True,
            "e001_executed": False,
            "predictions_computed": False,
            "mechanism_effects_computed": False,
            "control_comparisons_computed": False,
            "confirmatory_outcomes_observed": False,
            "biological_mechanism_established": False,
            "physical_quantum_promotion_eligible": False,
        },
    }
    capsule = dict(base)
    capsule["capsule_fingerprint"] = sha256(
        KUMAR2024_AUTHORITY_CAPSULE_DOMAIN
        + canonical_json(base).encode("utf-8")
    ).hexdigest()
    verify_kumar2024_authority_capsule_mapping(capsule)

    args.output.mkdir(parents=True, exist_ok=True)
    capsule_path = args.output / "authority-capsule.json"
    capsule_path.write_text(
        json.dumps(capsule, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    repository_manifest = {
        "authority-capsule.json": sha256(capsule_path.read_bytes()).hexdigest()
    }
    (args.output / "sha256-manifest.json").write_text(
        json.dumps(repository_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "capsule_fingerprint": capsule["capsule_fingerprint"],
                "cohort_authority_fingerprint": freeze[
                    "cohort_authority_fingerprint"
                ],
                "raw_dataset_fingerprint": raw["fingerprint"],
                "subjects": len(authorities),
                "raw_files": len(raw["files"]),
                "e001_executed": False,
                "confirmatory_outcomes_observed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
