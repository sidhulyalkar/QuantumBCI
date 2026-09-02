"""Read-side verification for persisted Kumar2024 cohort authority capsules."""

from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

KUMAR2024_AUTHORITY_CAPSULE_SCHEMA = 1
KUMAR2024_AUTHORITY_CAPSULE_ROLE = "kumar2024_full_cohort_authority_capsule_v1"
KUMAR2024_AUTHORITY_CAPSULE_DOMAIN = b"quantumbci.kumar2024-authority-capsule.v1\0"
KUMAR2024_AUTHORITY_FREEZE_PURPOSE = "kumar2024_full_cohort_authority_freeze_v1"
KUMAR2024_DATASET_ID = "moabb-kumar2024"
KUMAR2024_DATASET_KEY = "kumar2024"
KUMAR2024_SUBJECTS = tuple(range(1, 19))
KUMAR2024_EXPECTED_SESSIONS = ("0", "1", "2", "3", "4", "5")
KUMAR2024_HELD_OUT_SESSION = "5"
KUMAR2024_EVALUATION_FRACTION = 0.5
KUMAR2024_FMIN_HZ = 8.0
KUMAR2024_FMAX_HZ = 30.0
KUMAR2024_PROTOCOL_GROUPS = {
    "GR": tuple(range(1, 10)),
    "PAR": tuple(range(10, 19)),
}
KUMAR2024_MOABB_TO_RAW = {
    str(subject): subject if subject <= 9 else subject + 1
    for subject in KUMAR2024_SUBJECTS
}
KUMAR2024_FORBIDDEN_OUTCOME_KEYS = {
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

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHORT_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{16}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_text(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _require_list(name: str, value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _require_text(name: str, value: Any) -> str:
    if value is None:
        raise ValueError(f"{name} must not be null")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _require_bool(name: str, value: Any, expected: bool) -> None:
    if type(value) is not bool or value is not expected:
        raise ValueError(f"{name} must be the JSON boolean {str(expected).lower()}")


def _require_sha256(name: str, value: Any) -> str:
    text = _require_text(name, value)
    if _SHA256_RE.fullmatch(text) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return text


def _require_git_sha(name: str, value: Any) -> str:
    text = _require_text(name, value)
    if _GIT_SHA_RE.fullmatch(text) is None:
        raise ValueError(f"{name} must be a 40-character lowercase Git SHA")
    return text


def _require_short_fingerprint(name: str, value: Any) -> str:
    text = _require_text(name, value)
    if _SHORT_FINGERPRINT_RE.fullmatch(text) is None:
        raise ValueError(f"{name} must be a 16-character lowercase hex fingerprint")
    return text


def _iter_keys(value: Any):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _iter_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_keys(item)


def _expected_source_filenames() -> set[str]:
    return {
        "authority-freeze.json",
        "raw-source-fingerprint.json",
        *{
            f"authority-subject-{subject}-session-{KUMAR2024_HELD_OUT_SESSION}.json"
            for subject in KUMAR2024_SUBJECTS
        },
    }


def fingerprint_kumar2024_authority_capsule_mapping(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("capsule_fingerprint", None)
    return sha256(
        KUMAR2024_AUTHORITY_CAPSULE_DOMAIN + _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _verify_raw_source(
    raw: Mapping[str, Any], freeze: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    value = dict(raw)
    observed_fingerprint = _require_sha256(
        "raw_source_fingerprint.fingerprint", value.pop("fingerprint", None)
    )
    expected_fingerprint = _sha256_text(value)
    if observed_fingerprint != expected_fingerprint:
        raise ValueError("raw source fingerprint is stale or noncanonical")

    if int(raw.get("schema_version", 0)) != 2:
        raise ValueError("raw source fingerprint must use schema_version 2")
    if raw.get("kind") != "kumar2024_selected_raw_source_content_fingerprint":
        raise ValueError("raw source fingerprint has the wrong kind")
    if (
        raw.get("dataset_key") != KUMAR2024_DATASET_KEY
        or raw.get("dataset_id") != KUMAR2024_DATASET_ID
    ):
        raise ValueError("raw source fingerprint has the wrong Kumar2024 dataset identity")
    if raw.get("subjects") != list(KUMAR2024_SUBJECTS):
        raise ValueError("raw source fingerprint must bind the exact 18-subject cohort")

    selection = _require_mapping(
        "raw_source_fingerprint.selection", raw.get("selection")
    )
    expected_include = [
        "Offline/<group>/<subject>/**/*.gdf",
        "Online/<group>/<subject>/**/*.gdf",
    ]
    if (
        selection.get("include") != expected_include
        or selection.get("exclude") != ["Race/**"]
    ):
        raise ValueError(
            "raw source selection no longer matches the declared bar-feedback corpus"
        )
    if selection.get("moabb_subject_to_raw_subject") != KUMAR2024_MOABB_TO_RAW:
        raise ValueError("MOABB-to-raw Kumar2024 subject mapping drifted")

    raw_files = _require_list("raw_source_fingerprint.files", raw.get("files"))
    if not raw_files:
        raise ValueError("raw source fingerprint contains no files")
    file_by_name: dict[str, dict[str, Any]] = {}
    observed_names: list[str] = []
    total_bytes = 0
    for index, item in enumerate(raw_files):
        record = dict(_require_mapping(f"raw files[{index}]", item))
        name = _require_text(f"raw files[{index}].name", record.get("name"))
        if name in file_by_name:
            raise ValueError("raw source file names must be unique")
        if name.startswith("Race/") or not (
            name.startswith("Offline/") or name.startswith("Online/")
        ):
            raise ValueError(
                "raw source fingerprint contains a file outside Offline/Online bar-feedback data"
            )
        if not name.lower().endswith(".gdf"):
            raise ValueError("raw source fingerprint contains a non-GDF file")
        size = int(record.get("bytes", 0))
        if size <= 0:
            raise ValueError("raw source files must have positive byte sizes")
        _require_sha256(f"raw files[{index}].sha256", record.get("sha256"))
        file_by_name[name] = record
        observed_names.append(name)
        total_bytes += size
    if observed_names != sorted(observed_names):
        raise ValueError("raw source files must be serialized in canonical name order")

    by_subject = _require_mapping(
        "raw_source_fingerprint.by_subject", raw.get("by_subject")
    )
    if set(by_subject) != {str(subject) for subject in KUMAR2024_SUBJECTS}:
        raise ValueError(
            "raw source by_subject mapping must cover exactly subjects 1..18"
        )
    for subject in KUMAR2024_SUBJECTS:
        record = _require_mapping(
            f"raw by_subject[{subject}]", by_subject[str(subject)]
        )
        if int(record.get("subject", -1)) != subject:
            raise ValueError("raw source subject identity drifted")
        names = _require_list(
            f"raw by_subject[{subject}].file_names", record.get("file_names")
        )
        if not names or names != sorted(names) or len(names) != len(set(names)):
            raise ValueError(
                "raw source subject file names must be nonempty, unique, and sorted"
            )
        try:
            subject_files = [
                file_by_name[_require_text("raw subject file name", name)]
                for name in names
            ]
        except KeyError as exc:
            raise ValueError(
                "raw source subject references a file absent from the aggregate manifest"
            ) from exc
        identity = {"subject": subject, "files": subject_files}
        if record.get("fingerprint") != _sha256_text(identity):
            raise ValueError(f"raw source subject {subject} fingerprint is stale")

    if int(freeze.get("raw_file_count", -1)) != len(raw_files):
        raise ValueError(
            "authority freeze raw_file_count disagrees with raw source fingerprint"
        )
    if int(freeze.get("raw_total_bytes", -1)) != total_bytes:
        raise ValueError(
            "authority freeze raw_total_bytes disagrees with raw source fingerprint"
        )
    if freeze.get("raw_dataset_fingerprint") != observed_fingerprint:
        raise ValueError("authority freeze does not bind the exact raw dataset fingerprint")
    return file_by_name


def _verify_freeze(freeze: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    value = dict(freeze)
    observed_fingerprint = _require_sha256(
        "authority_freeze.authority_freeze_fingerprint",
        value.pop("authority_freeze_fingerprint", None),
    )
    if observed_fingerprint != _sha256_text(value):
        raise ValueError("authority freeze fingerprint is stale or noncanonical")
    if freeze.get("purpose") != KUMAR2024_AUTHORITY_FREEZE_PURPOSE:
        raise ValueError("authority freeze has the wrong purpose")
    if freeze.get("dataset_id") != KUMAR2024_DATASET_ID:
        raise ValueError("authority freeze has the wrong dataset_id")
    if (
        freeze.get("subjects") != list(KUMAR2024_SUBJECTS)
        or int(freeze.get("subject_count", -1)) != 18
    ):
        raise ValueError("authority freeze must cover exactly Kumar2024 subjects 1..18")
    expected_protocol_groups = {
        key: list(values) for key, values in KUMAR2024_PROTOCOL_GROUPS.items()
    }
    if freeze.get("protocol_groups") != expected_protocol_groups:
        raise ValueError("authority freeze GR/PAR protocol groups drifted")
    if (
        float(freeze.get("fmin_hz")) != KUMAR2024_FMIN_HZ
        or float(freeze.get("fmax_hz")) != KUMAR2024_FMAX_HZ
    ):
        raise ValueError("authority freeze frequency band drifted")
    _require_bool(
        "authority_freeze.native_resampling", freeze.get("native_resampling"), True
    )
    if freeze.get("expected_sessions") != list(KUMAR2024_EXPECTED_SESSIONS):
        raise ValueError("authority freeze session chronology drifted")
    if freeze.get("held_out_session") != KUMAR2024_HELD_OUT_SESSION:
        raise ValueError("authority freeze held-out session drifted")
    if float(freeze.get("evaluation_fraction")) != KUMAR2024_EVALUATION_FRACTION:
        raise ValueError("authority freeze evaluation fraction drifted")

    for key in (
        "dataset_structure_inspected",
        "class_labels_used_for_stratified_split_construction",
        "evidence_assignment_frozen",
        "final_evaluation_indices_created",
    ):
        _require_bool(f"authority_freeze.{key}", freeze.get(key), True)
    for key in (
        "e001_executed",
        "predictions_computed",
        "mechanism_effects_computed",
        "control_comparisons_computed",
        "confirmatory_outcomes_observed",
    ):
        _require_bool(f"authority_freeze.{key}", freeze.get(key), False)

    cases = _require_list("authority_freeze.cases", freeze.get("cases"))
    if len(cases) != 18:
        raise ValueError("authority freeze must contain exactly 18 case summaries")
    case_by_subject: dict[int, Mapping[str, Any]] = {}
    for case in cases:
        mapping = _require_mapping("authority freeze case", case)
        subject = int(mapping.get("subject", -1))
        if subject in case_by_subject or subject not in KUMAR2024_SUBJECTS:
            raise ValueError(
                "authority freeze cases contain duplicate or unknown subjects"
            )
        case_by_subject[subject] = mapping
    if list(case_by_subject) != list(KUMAR2024_SUBJECTS):
        raise ValueError("authority freeze case order must be subjects 1..18")

    cohort_identity = {
        "dataset_id": KUMAR2024_DATASET_ID,
        "subjects": list(KUMAR2024_SUBJECTS),
        "held_out_session": KUMAR2024_HELD_OUT_SESSION,
        "evaluation_fraction": KUMAR2024_EVALUATION_FRACTION,
        "raw_dataset_fingerprint": freeze.get("raw_dataset_fingerprint"),
        "case_authorities": [
            {
                "subject": subject,
                "authority_fingerprint": case_by_subject[subject].get(
                    "authority_fingerprint"
                ),
                "partition_fingerprint": case_by_subject[subject].get(
                    "partition_fingerprint"
                ),
                "calibration_split_fingerprint": case_by_subject[subject].get(
                    "calibration_split_fingerprint"
                ),
                "processed_data_sha256": case_by_subject[subject].get(
                    "processed_data_sha256"
                ),
            }
            for subject in KUMAR2024_SUBJECTS
        ],
    }
    if freeze.get("cohort_authority_fingerprint") != _sha256_text(cohort_identity):
        raise ValueError("cohort authority fingerprint is stale or inconsistent")
    return case_by_subject


def _as_index_set(name: str, raw: Any, n_samples: int) -> set[int]:
    values = _require_list(name, raw)
    parsed = [int(value) for value in values]
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"{name} contains duplicate indices")
    if any(index < 0 or index >= n_samples for index in parsed):
        raise ValueError(f"{name} contains out-of-range indices")
    return set(parsed)


def _verify_authorities(
    authorities: list[Any],
    case_by_subject: Mapping[int, Mapping[str, Any]],
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    if len(authorities) != 18:
        raise ValueError("subject_authorities must contain exactly 18 mappings")
    by_subject_raw = _require_mapping(
        "raw_source_fingerprint.by_subject", raw.get("by_subject")
    )
    seen_authority: set[str] = set()
    seen_processed: set[str] = set()
    by_filename: dict[str, Any] = {}

    for position, item in enumerate(authorities):
        authority = _require_mapping(f"subject_authorities[{position}]", item)
        metadata = _require_mapping(
            "authority.case_metadata", authority.get("case_metadata")
        )
        subject = int(metadata.get("subject", -1))
        expected_subject = KUMAR2024_SUBJECTS[position]
        if subject != expected_subject:
            raise ValueError(
                "subject authorities must be serialized in exact subject 1..18 order"
            )
        protocol = "GR" if subject <= 9 else "PAR"
        if metadata.get("original_protocol") != protocol:
            raise ValueError(f"subject {subject}: original_protocol drifted")
        if metadata.get("held_out_session") != KUMAR2024_HELD_OUT_SESSION:
            raise ValueError(f"subject {subject}: held_out_session drifted")
        _require_bool(
            f"subject {subject}: authority_freeze_only",
            metadata.get("authority_freeze_only"),
            True,
        )
        seed = int(authority.get("seed", -1))
        if int(metadata.get("split_seed", -2)) != seed:
            raise ValueError(
                f"subject {subject}: split seed differs between metadata and authority"
            )
        expected_case_id = (
            f"{KUMAR2024_DATASET_ID}/subject-{subject}/session-5/split-{seed}"
        )
        if authority.get("case_id") != expected_case_id:
            raise ValueError(f"subject {subject}: case_id does not match frozen identity")
        if authority.get("dataset_id") != KUMAR2024_DATASET_ID:
            raise ValueError(f"subject {subject}: dataset_id drifted")
        if (
            authority.get("split_unit") != "session"
            or authority.get("history_policy") != "prior"
        ):
            raise ValueError(f"subject {subject}: longitudinal split policy drifted")
        if authority.get("held_out_values") != [KUMAR2024_HELD_OUT_SESSION]:
            raise ValueError(f"subject {subject}: held_out_values drifted")
        if authority.get("observed_group_order") != list(KUMAR2024_EXPECTED_SESSIONS):
            raise ValueError(f"subject {subject}: observed session order drifted")
        if authority.get("source_group_values") != list(
            KUMAR2024_EXPECTED_SESSIONS[:-1]
        ):
            raise ValueError(f"subject {subject}: source history sessions drifted")
        if float(authority.get("evaluation_fraction")) != KUMAR2024_EVALUATION_FRACTION:
            raise ValueError(f"subject {subject}: evaluation_fraction drifted")

        n_samples = int(authority.get("n_samples", 0))
        shape = _require_list(
            f"subject {subject}: input_shape", authority.get("input_shape")
        )
        if shape != [n_samples, 22, 2561] or n_samples <= 0:
            raise ValueError(f"subject {subject}: frozen input geometry is invalid")
        source = _as_index_set(
            f"subject {subject}: source_train_indices",
            authority.get("source_train_indices"),
            n_samples,
        )
        evaluation = _as_index_set(
            f"subject {subject}: evaluation_indices",
            authority.get("evaluation_indices"),
            n_samples,
        )
        calibration_by_class = _require_mapping(
            f"subject {subject}: calibration_order_by_class",
            authority.get("calibration_order_by_class"),
        )
        if set(calibration_by_class) != {"left_hand", "right_hand"}:
            raise ValueError(f"subject {subject}: calibration classes drifted")
        calibration: set[int] = set()
        for label in ("left_hand", "right_hand"):
            indices = _as_index_set(
                f"subject {subject}: calibration_order_by_class[{label}]",
                calibration_by_class[label],
                n_samples,
            )
            if not indices:
                raise ValueError(f"subject {subject}: empty calibration class {label}")
            if calibration.intersection(indices):
                raise ValueError(f"subject {subject}: calibration classes overlap")
            calibration.update(indices)
        if source & calibration or source & evaluation or calibration & evaluation:
            raise ValueError(
                f"subject {subject}: source/calibration/evaluation partitions overlap"
            )
        if source | calibration | evaluation != set(range(n_samples)):
            raise ValueError(
                f"subject {subject}: frozen partitions do not cover all processed epochs"
            )

        authority_fingerprint = _require_short_fingerprint(
            f"subject {subject}: authority_fingerprint",
            authority.get("authority_fingerprint"),
        )
        partition_fingerprint = _require_short_fingerprint(
            f"subject {subject}: partition_fingerprint",
            authority.get("partition_fingerprint"),
        )
        split_fingerprint = _require_short_fingerprint(
            f"subject {subject}: calibration_split_fingerprint",
            authority.get("calibration_split_fingerprint"),
        )
        processed_sha = _require_sha256(
            f"subject {subject}: processed_data_sha256",
            authority.get("processed_data_sha256"),
        )
        if authority_fingerprint in seen_authority or processed_sha in seen_processed:
            raise ValueError(
                "distinct subjects must not share authority or processed-data fingerprints"
            )
        seen_authority.add(authority_fingerprint)
        seen_processed.add(processed_sha)

        case = case_by_subject[subject]
        expected_case_values = {
            "subject": subject,
            "original_protocol": protocol,
            "input_shape": shape,
            "observed_sessions": list(KUMAR2024_EXPECTED_SESSIONS),
            "held_out_session": KUMAR2024_HELD_OUT_SESSION,
            "source_history_sessions": list(KUMAR2024_EXPECTED_SESSIONS[:-1]),
            "source_train_samples": len(source),
            "evaluation_samples": len(evaluation),
            "calibration_pool_samples": len(calibration),
            "split_seed": seed,
            "authority_fingerprint": authority_fingerprint,
            "partition_fingerprint": partition_fingerprint,
            "calibration_split_fingerprint": split_fingerprint,
            "processed_data_sha256": processed_sha,
            "raw_subject_fingerprint": by_subject_raw[str(subject)].get("fingerprint"),
        }
        for key, expected in expected_case_values.items():
            if case.get(key) != expected:
                raise ValueError(
                    f"subject {subject}: freeze case field {key!r} disagrees with authority"
                )
        if int(case.get("max_budget_per_class", 0)) <= 0:
            raise ValueError(
                f"subject {subject}: max_budget_per_class must be positive"
            )

        filename = f"authority-subject-{subject}-session-5.json"
        by_filename[filename] = dict(authority)
    return by_filename


def verify_kumar2024_authority_capsule_mapping(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a persisted outcome-blind Kumar2024 cohort authority capsule.

    This verifier checks canonical fingerprints, reproduces the producing workflow's JSON
    file hashes from the embedded mappings, validates raw-source and cohort identities,
    and proves that all 18 longitudinal source/calibration/evaluation partitions are
    complete and disjoint. It intentionally does not execute E001 or inspect mechanism
    outcomes.
    """

    value = dict(_require_mapping("capsule", payload))
    if set(value) != {
        "schema_version",
        "artifact_role",
        "purpose",
        "source_artifact",
        "authority_freeze",
        "raw_source_fingerprint",
        "subject_authorities",
        "canonical_component_sha256",
        "claim_boundary",
        "capsule_fingerprint",
    }:
        raise ValueError(
            "Kumar2024 authority capsule has an unexpected top-level key set"
        )
    if int(value.get("schema_version", 0)) != KUMAR2024_AUTHORITY_CAPSULE_SCHEMA:
        raise ValueError("unsupported Kumar2024 authority capsule schema")
    if value.get("artifact_role") != KUMAR2024_AUTHORITY_CAPSULE_ROLE:
        raise ValueError("wrong Kumar2024 authority capsule role")
    if (
        value.get("purpose")
        != "persisted_outcome_blind_design_and_preregistration_authority"
    ):
        raise ValueError("wrong Kumar2024 authority capsule purpose")

    observed_capsule_fingerprint = _require_sha256(
        "capsule_fingerprint", value.get("capsule_fingerprint")
    )
    if (
        observed_capsule_fingerprint
        != fingerprint_kumar2024_authority_capsule_mapping(value)
    ):
        raise ValueError(
            "Kumar2024 authority capsule fingerprint is stale or noncanonical"
        )

    source_artifact = _require_mapping(
        "source_artifact", value.get("source_artifact")
    )
    if source_artifact.get("repository") != "sidhulyalkar/QuantumBCI":
        raise ValueError("source artifact repository drifted")
    if (
        int(source_artifact.get("workflow_run_id", 0)) <= 0
        or int(source_artifact.get("artifact_id", 0)) <= 0
    ):
        raise ValueError("source artifact workflow/artifact IDs must be positive")
    _require_git_sha(
        "source_artifact.workflow_head_sha", source_artifact.get("workflow_head_sha")
    )
    artifact_digest = _require_text(
        "source_artifact.artifact_zip_digest",
        source_artifact.get("artifact_zip_digest"),
    )
    if (
        not artifact_digest.startswith("sha256:")
        or _SHA256_RE.fullmatch(artifact_digest.removeprefix("sha256:")) is None
    ):
        raise ValueError("source artifact ZIP digest must be sha256:<64 lowercase hex>")
    if source_artifact.get("producing_workflow_conclusion") != "success":
        raise ValueError("source artifact must originate from a successful workflow")
    _require_bool(
        "source_artifact.producing_workflow_outcome_blind_guard_passed",
        source_artifact.get("producing_workflow_outcome_blind_guard_passed"),
        True,
    )

    freeze = _require_mapping(
        "authority_freeze", value.get("authority_freeze")
    )
    raw = _require_mapping(
        "raw_source_fingerprint", value.get("raw_source_fingerprint")
    )
    authorities = _require_list(
        "subject_authorities", value.get("subject_authorities")
    )
    case_by_subject = _verify_freeze(freeze)
    _verify_raw_source(raw, freeze)
    authority_by_filename = _verify_authorities(
        authorities, case_by_subject, raw
    )

    if freeze.get("science_source_sha") != source_artifact.get(
        "artifact_name", ""
    ).removeprefix("Kumar2024-full-authority-freeze-"):
        raise ValueError(
            "source artifact name does not bind the exact QuantumBCI science source"
        )
    _require_git_sha(
        "authority_freeze.science_source_sha", freeze.get("science_source_sha")
    )
    _require_git_sha(
        "authority_freeze.neuros_source_sha", freeze.get("neuros_source_sha")
    )

    semantic_files: dict[str, Any] = {
        "authority-freeze.json": dict(freeze),
        "raw-source-fingerprint.json": dict(raw),
        **authority_by_filename,
    }
    expected_filenames = _expected_source_filenames()
    if set(semantic_files) != expected_filenames:
        raise ValueError("embedded source file family is incomplete")

    original_manifest = _require_mapping(
        "source_artifact.original_file_sha256_manifest",
        source_artifact.get("original_file_sha256_manifest"),
    )
    if set(original_manifest) != expected_filenames:
        raise ValueError("source artifact manifest has the wrong file set")
    reproduced_original_manifest = {
        name: sha256(_pretty_json_bytes(document)).hexdigest()
        for name, document in sorted(semantic_files.items())
    }
    if dict(original_manifest) != reproduced_original_manifest:
        raise ValueError(
            "embedded mappings do not reproduce the successful workflow's file manifest"
        )

    component_manifest = _require_mapping(
        "canonical_component_sha256", value.get("canonical_component_sha256")
    )
    expected_components = {
        name: _sha256_text(document)
        for name, document in sorted(semantic_files.items())
    }
    if dict(component_manifest) != expected_components:
        raise ValueError("canonical component fingerprints are stale")

    claim_boundary = _require_mapping(
        "claim_boundary", value.get("claim_boundary")
    )
    expected_claim_boundary = {
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
    }
    if dict(claim_boundary) != expected_claim_boundary:
        raise ValueError("Kumar2024 authority capsule claim boundary drifted")

    leaked = sorted(
        KUMAR2024_FORBIDDEN_OUTCOME_KEYS.intersection(_iter_keys(value))
    )
    if leaked:
        raise ValueError(
            f"outcome-shaped fields leaked into authority capsule: {leaked}"
        )
    return value


def load_kumar2024_authority_capsule(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Kumar2024 authority capsule JSON root must be an object")
    return verify_kumar2024_authority_capsule_mapping(payload)
