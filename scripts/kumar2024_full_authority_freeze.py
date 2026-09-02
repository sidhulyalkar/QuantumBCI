#!/usr/bin/env python3
"""Freeze the complete Kumar2024 cohort authority without computing mechanism outcomes.

This program intentionally performs design-stage inspection only. It fingerprints the exact
public source files, validates dataset geometry/session chronology, and freezes the session-5
longitudinal calibration/evaluation authority for all 18 participants. It does not call E001,
fit a predictive model, compute candidate/control metrics, or evaluate a mechanism effect.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from neuros.foundation_models.moabb_longitudinal import build_moabb_longitudinal_dataset
from neuros.foundation_models.real_world import collect_moabb

from quantumbci.studies.kumar2024 import (
    _neuros_authority_api,
    _stable_seed,
    fingerprint_raw_dataset,
)

SUBJECTS = tuple(range(1, 19))
TARGET_SESSION = "5"
BASE_SEED = 2026
EVALUATION_FRACTION = 0.5
FMIN_HZ = 8.0
FMAX_HZ = 30.0
EXPECTED_SESSIONS = ("0", "1", "2", "3", "4", "5")

FORBIDDEN_OUTCOME_KEYS = {
    "accuracy",
    "auc",
    "candidate_advantage",
    "ablation_necessity",
    "effect_delta",
    "prediction_values",
    "predictions",
    "candidate_metric",
    "control_metric",
    "ablated_metric",
    "p_value",
    "bootstrap_ci_lower",
    "bootstrap_ci_upper",
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def iter_keys(value: object):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from iter_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_keys(item)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    output = Path(os.environ.get("KUMAR_AUTHORITY_OUTPUT", "kumar2024-full-authority-freeze"))
    output.mkdir(parents=True, exist_ok=True)

    dataset_spec, dataset, paradigm = build_moabb_longitudinal_dataset(
        "kumar2024",
        fmin=FMIN_HZ,
        fmax=FMAX_HZ,
        resample=None,
    )
    if dataset_spec.expected_session_order != EXPECTED_SESSIONS:
        raise RuntimeError(
            "Kumar2024 pinned session contract drifted before full authority freeze"
        )

    raw = fingerprint_raw_dataset(
        dataset,
        SUBJECTS,
        dataset_key=dataset_spec.key,
        dataset_id=dataset_spec.source_id,
    )
    if raw["subjects"] != list(SUBJECTS):
        raise RuntimeError("raw source fingerprint did not bind the exact 18-subject cohort")
    expected_mapping = {
        str(subject): subject if subject <= 9 else subject + 1 for subject in SUBJECTS
    }
    if raw["selection"]["moabb_subject_to_raw_subject"] != expected_mapping:
        raise RuntimeError("MOABB-to-raw subject mapping drifted")
    write_json(output / "raw-source-fingerprint.json", raw)

    api = _neuros_authority_api()
    cases: list[dict[str, object]] = []
    seen_authority_fingerprints: set[str] = set()
    seen_processed_hashes: set[str] = set()

    for subject in SUBJECTS:
        data = collect_moabb(
            dataset,
            paradigm,
            subjects=[subject],
            dataset_id=dataset_spec.source_id,
        )
        x = np.asarray(data.X)
        if x.ndim != 3:
            raise RuntimeError(f"subject {subject}: expected 3-D EEG epochs, got {x.shape}")
        if x.shape[1] != 22:
            raise RuntimeError(f"subject {subject}: expected 22 EEG channels, got {x.shape[1]}")
        if x.shape[2] != 2561:
            raise RuntimeError(
                f"subject {subject}: expected 2561 samples/epoch, got {x.shape[2]}"
            )
        if not np.isfinite(x).all():
            raise RuntimeError(f"subject {subject}: non-finite EEG values detected")

        observed = api.validate_observed_sessions(
            dataset_spec,
            api.ordered_group_values(data, split_unit="session"),
        )
        if tuple(observed) != EXPECTED_SESSIONS:
            raise RuntimeError(
                f"subject {subject}: session chronology drifted; observed={observed}"
            )

        partition = api.chronological_partition(
            data,
            split_unit="session",
            held_out_value=TARGET_SESSION,
            order=observed,
        )
        split_seed = _stable_seed(
            BASE_SEED,
            dataset_spec.source_id,
            subject,
            TARGET_SESSION,
        )
        split = api.make_nested_calibration_split(
            partition,
            evaluation_fraction=EVALUATION_FRACTION,
            seed=split_seed,
        )
        metadata = dataset_spec.case_metadata(subject)
        metadata.update(
            {
                "held_out_session": TARGET_SESSION,
                "split_seed": int(split_seed),
                "authority_freeze_only": True,
            }
        )
        authority = api.LongitudinalCaseAuthority.from_split(
            split,
            case_id=(
                f"{dataset_spec.source_id}/subject-{subject}/session-{TARGET_SESSION}/"
                f"split-{split_seed}"
            ),
            history_policy="prior",
            observed_group_order=observed,
            case_metadata=metadata,
        )
        restored = authority.restore(data)
        if restored.fingerprint != split.fingerprint:
            raise RuntimeError(
                f"subject {subject}: restored split fingerprint drifted from frozen split"
            )
        if authority.authority_fingerprint in seen_authority_fingerprints:
            raise RuntimeError("distinct subjects produced duplicate authority fingerprints")
        seen_authority_fingerprints.add(authority.authority_fingerprint)
        if authority.processed_data_sha256 in seen_processed_hashes:
            raise RuntimeError("distinct subjects produced duplicate processed-data hashes")
        seen_processed_hashes.add(authority.processed_data_sha256)

        authority_path = output / f"authority-subject-{subject}-session-{TARGET_SESSION}.json"
        authority_mapping = authority.to_dict()
        write_json(authority_path, authority_mapping)

        calibration_pool = sum(
            len(values) for values in authority.calibration_order_by_class.values()
        )
        if calibration_pool <= 0 or len(authority.evaluation_indices) <= 0:
            raise RuntimeError(f"subject {subject}: empty held-out authority partition")

        cases.append(
            {
                "subject": subject,
                "original_protocol": metadata["original_protocol"],
                "input_shape": list(x.shape),
                "dtype": x.dtype.str,
                "observed_sessions": list(observed),
                "held_out_session": TARGET_SESSION,
                "source_history_sessions": list(authority.source_group_values),
                "source_train_samples": len(authority.source_train_indices),
                "evaluation_samples": len(authority.evaluation_indices),
                "calibration_pool_samples": calibration_pool,
                "max_budget_per_class": int(split.max_budget_per_class),
                "split_seed": int(split_seed),
                "authority_fingerprint": authority.authority_fingerprint,
                "partition_fingerprint": authority.partition_fingerprint,
                "calibration_split_fingerprint": authority.calibration_split_fingerprint,
                "processed_data_sha256": authority.processed_data_sha256,
                "raw_subject_fingerprint": raw["by_subject"][str(subject)]["fingerprint"],
            }
        )

    if [case["subject"] for case in cases] != list(SUBJECTS):
        raise RuntimeError("case order/coverage differs from exact 18-subject cohort")

    cohort_authority_identity = {
        "dataset_id": dataset_spec.source_id,
        "subjects": list(SUBJECTS),
        "held_out_session": TARGET_SESSION,
        "evaluation_fraction": EVALUATION_FRACTION,
        "raw_dataset_fingerprint": raw["fingerprint"],
        "case_authorities": [
            {
                "subject": case["subject"],
                "authority_fingerprint": case["authority_fingerprint"],
                "partition_fingerprint": case["partition_fingerprint"],
                "calibration_split_fingerprint": case["calibration_split_fingerprint"],
                "processed_data_sha256": case["processed_data_sha256"],
            }
            for case in cases
        ],
    }
    cohort_authority_fingerprint = hashlib.sha256(
        canonical_json(cohort_authority_identity).encode("utf-8")
    ).hexdigest()

    payload: dict[str, object] = {
        "purpose": "kumar2024_full_cohort_authority_freeze_v1",
        "science_source_sha": os.environ["SCIENCE_SOURCE_SHA"],
        "neuros_source_sha": os.environ["NEUROS_SHA"],
        "dataset_id": dataset_spec.source_id,
        "subjects": list(SUBJECTS),
        "subject_count": len(SUBJECTS),
        "protocol_groups": {"GR": list(range(1, 10)), "PAR": list(range(10, 19))},
        "fmin_hz": FMIN_HZ,
        "fmax_hz": FMAX_HZ,
        "native_resampling": True,
        "expected_sessions": list(EXPECTED_SESSIONS),
        "held_out_session": TARGET_SESSION,
        "evaluation_fraction": EVALUATION_FRACTION,
        "raw_dataset_fingerprint": raw["fingerprint"],
        "raw_file_count": len(raw["files"]),
        "raw_total_bytes": sum(int(item["bytes"]) for item in raw["files"]),
        "cohort_authority_fingerprint": cohort_authority_fingerprint,
        "cases": cases,
        "dataset_structure_inspected": True,
        "class_labels_used_for_stratified_split_construction": True,
        "evidence_assignment_frozen": True,
        "final_evaluation_indices_created": True,
        "e001_executed": False,
        "predictions_computed": False,
        "mechanism_effects_computed": False,
        "control_comparisons_computed": False,
        "confirmatory_outcomes_observed": False,
        "interpretation": (
            "The complete public Kumar2024 cohort, source bytes, session chronology, and exact "
            "session-5 longitudinal evidence assignments were inspected and fingerprinted for "
            "design/preregistration authority. Class labels were used only to construct the "
            "declared stratified calibration/evaluation split. No E001, prediction, mechanism "
            "effect, or control-comparison outcome was computed."
        ),
    }
    canonical = canonical_json(payload)
    payload["authority_freeze_fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()
    write_json(output / "authority-freeze.json", payload)

    leaked = sorted(FORBIDDEN_OUTCOME_KEYS.intersection(iter_keys(payload)))
    if leaked:
        raise RuntimeError(f"outcome fields leaked into authority-freeze payload: {leaked}")

    for path in sorted(output.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        leaked = sorted(FORBIDDEN_OUTCOME_KEYS.intersection(iter_keys(document)))
        if leaked:
            raise RuntimeError(f"outcome fields leaked into {path.name}: {leaked}")

    manifest = {
        path.name: sha256_file(path)
        for path in sorted(output.glob("*.json"))
        if path.name != "sha256-manifest.json"
    }
    write_json(output / "sha256-manifest.json", manifest)

    print(
        json.dumps(
            {
                "authority_freeze_fingerprint": payload["authority_freeze_fingerprint"],
                "cohort_authority_fingerprint": cohort_authority_fingerprint,
                "raw_dataset_fingerprint": raw["fingerprint"],
                "subject_count": len(cases),
                "raw_file_count": len(raw["files"]),
                "dataset_structure_inspected": True,
                "evidence_assignment_frozen": True,
                "mechanism_effects_computed": False,
                "confirmatory_outcomes_observed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
