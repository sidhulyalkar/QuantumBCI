"""Canonical cached execution path for the real Kumar2024 E001 study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..e001_longitudinal_prepared import run_prepared_longitudinal_e001_case
from ..e001_prepared import prepare_e001_static_features
from ..exporting import verify_run_artifacts
from ..longitudinal import LongitudinalE001CaseResult
from .kumar2024 import (
    Kumar2024StudyConfig,
    _json_dump,
    _neuros_authority_api,
    _sha256_file,
    _stable_seed,
    _write_study_bundle,
    fingerprint_raw_dataset,
)


_ADAPTATION_CONTRACT = {
    "static_feature_scope": "prepared_once_per_participant_tensor",
    "pca_fit_scope": "source_history_only",
    "target_calibration_changes": "readout_only",
    "final_evaluation_in_representation_fit": False,
}


def run_kumar2024_subject(
    data: Any,
    dataset_spec: Any,
    *,
    subject: int,
    config: Kumar2024StudyConfig,
    upstream_dataset_fingerprint: str,
    quantumbci_source_sha: str,
    neuros_source_sha: str,
) -> tuple[tuple[Any, ...], tuple[LongitudinalE001CaseResult, ...]]:
    """Execute one subject while preparing budget-independent features exactly once."""

    api = _neuros_authority_api()
    observed = api.validate_observed_sessions(
        dataset_spec,
        api.ordered_group_values(data, split_unit="session"),
    )
    if len(observed) < 2:
        raise RuntimeError(f"subject {subject} has fewer than two usable sessions")

    if config.held_out_sessions is None:
        targets = tuple(observed[1:])
    else:
        missing = [value for value in config.held_out_sessions if value not in observed]
        if missing:
            raise RuntimeError(
                f"subject {subject} missing requested session(s) {missing}; observed={list(observed)}"
            )
        targets = tuple(config.held_out_sessions)

    x = np.asarray(data.X)
    if x.ndim != 3:
        raise ValueError(
            "Kumar2024 E001 raw-token lane expects MOABB epochs shaped "
            "(samples, channels, time)"
        )
    token_representation = np.transpose(x, (0, 2, 1))
    static = prepare_e001_static_features(
        token_representation,
        np.asarray(data.y),
        center_tokens=True,
        covariance_regularization=config.covariance_regularization,
    )
    representation_id = (
        f"{dataset_spec.source_id}:time-by-channel:band={config.fmin:g}-{config.fmax:g}Hz:"
        f"resample={'native' if config.resample_hz is None else f'{config.resample_hz:g}Hz'}"
    )

    authorities: list[Any] = []
    cases: list[LongitudinalE001CaseResult] = []
    for target in targets:
        partition = api.chronological_partition(
            data,
            split_unit="session",
            held_out_value=target,
            order=observed,
        )
        case_split_seed = _stable_seed(
            config.split_seed,
            dataset_spec.source_id,
            int(subject),
            str(target),
        )
        split = api.make_nested_calibration_split(
            partition,
            evaluation_fraction=config.evaluation_fraction,
            seed=case_split_seed,
        )
        if config.budgets_per_class[-1] > split.max_budget_per_class:
            raise RuntimeError(
                f"strict paired frontier requires {config.budgets_per_class[-1]}/class, but "
                f"subject={subject}, session={target} supports only "
                f"{split.max_budget_per_class}/class"
            )

        metadata = dataset_spec.case_metadata(int(subject))
        metadata.update(
            {
                "held_out_session": str(target),
                "split_seed": int(case_split_seed),
                "pca_fit_scope": "source_history_only",
            }
        )
        authority = api.LongitudinalCaseAuthority.from_split(
            split,
            case_id=(
                f"{dataset_spec.source_id}/subject-{subject}/session-{target}/"
                f"split-{case_split_seed}"
            ),
            history_policy="prior",
            observed_group_order=observed,
            case_metadata=metadata,
        )
        authority.restore(data)
        case = run_prepared_longitudinal_e001_case(
            data,
            authority,
            static,
            representation_id=representation_id,
            budgets_per_class=config.budgets_per_class,
            upstream_dataset_fingerprint=upstream_dataset_fingerprint,
            quantumbci_source_sha=quantumbci_source_sha,
            neuros_source_sha=neuros_source_sha,
            ridge=config.ridge,
        )
        authorities.append(authority)
        cases.append(case)
    return tuple(authorities), tuple(cases)


def _finalize_prepared_bundle(output: Path) -> dict[str, Any]:
    """Make the prepared-feature adaptation contract explicit and re-close the ledger."""

    manifest_path = output / "study_manifest.json"
    ledger_path = output / "evidence_ledger.json"
    representation_path = output / "representation_index.json"
    run_path = output / "run.json"
    report_path = output / "report.md"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["representation_adaptation"] = dict(_ADAPTATION_CONTRACT)
    boundary = list(manifest.get("claim_boundary", []))
    rule = (
        "PCA is fit once on chronological source history and remains frozen across target "
        "calibration budgets; target calibration changes readouts only"
    )
    if rule not in boundary:
        boundary.append(rule)
    manifest["claim_boundary"] = boundary
    _json_dump(manifest_path, manifest)

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["representation_adaptation"] = dict(_ADAPTATION_CONTRACT)
    _json_dump(ledger_path, ledger)

    representation = json.loads(representation_path.read_text(encoding="utf-8"))
    representation["control_preparation"] = dict(_ADAPTATION_CONTRACT)
    _json_dump(representation_path, representation)

    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["representation_adaptation"] = dict(_ADAPTATION_CONTRACT)
    _json_dump(run_path, run)

    report = report_path.read_text(encoding="utf-8").rstrip()
    report += (
        "\n\n## Representation adaptation contract\n\n"
        "Budget-independent density/covariance/operator features are prepared once from the "
        "participant tensor. The flattened PCA control is fit once using chronological source "
        "history only. Target-session calibration examples may update the matched readout, but "
        "they do not refit PCA or any other representation transform. Final evaluation examples "
        "never enter representation fitting.\n"
    )
    report_path.write_text(report, encoding="utf-8")

    hashes = {
        path.name: _sha256_file(path)
        for path in output.iterdir()
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _json_dump(output / "artifact_hashes.json", hashes)
    verification = verify_run_artifacts(output)
    if not verification["valid"]:
        raise RuntimeError(f"prepared study artifact verification failed: {verification}")
    return verification


def run_kumar2024_study(
    output: str | Path,
    *,
    config: Kumar2024StudyConfig,
    quantumbci_source_sha: str,
    neuros_source_sha: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Collect selected Kumar2024 subjects and write a verified cached E001 bundle."""

    try:
        from neuros.foundation_models.moabb_longitudinal import build_moabb_longitudinal_dataset
        from neuros.foundation_models.real_world import collect_moabb
    except ImportError as exc:  # pragma: no cover - optional real-study dependency
        raise ImportError(
            "Kumar2024 execution requires neuros-foundation[evidence] with MOABB."
        ) from exc

    dataset_spec, dataset, paradigm = build_moabb_longitudinal_dataset(
        "kumar2024",
        fmin=config.fmin,
        fmax=config.fmax,
        resample=config.resample_hz,
    )
    raw_fingerprint = fingerprint_raw_dataset(
        dataset,
        config.subjects,
        dataset_key=dataset_spec.key,
        dataset_id=dataset_spec.source_id,
    )

    authorities: list[Any] = []
    cases: list[LongitudinalE001CaseResult] = []
    for subject in config.subjects:
        subject_record = raw_fingerprint["by_subject"].get(str(int(subject)))
        if not isinstance(subject_record, dict) or not subject_record.get("fingerprint"):
            raise RuntimeError(
                f"raw fingerprint manifest lacks participant fingerprint for subject {subject}"
            )
        data = collect_moabb(
            dataset,
            paradigm,
            subjects=[int(subject)],
            dataset_id=dataset_spec.source_id,
        )
        subject_authorities, subject_cases = run_kumar2024_subject(
            data,
            dataset_spec,
            subject=int(subject),
            config=config,
            upstream_dataset_fingerprint=str(subject_record["fingerprint"]),
            quantumbci_source_sha=quantumbci_source_sha,
            neuros_source_sha=neuros_source_sha,
        )
        authorities.extend(subject_authorities)
        cases.extend(subject_cases)

    resolved = Path(output).resolve()
    result = _write_study_bundle(
        resolved,
        config=config,
        dataset_spec=dataset_spec,
        raw_fingerprint=raw_fingerprint,
        authorities=authorities,
        cases=cases,
        quantumbci_source_sha=quantumbci_source_sha,
        neuros_source_sha=neuros_source_sha,
        overwrite=overwrite,
    )
    verification = _finalize_prepared_bundle(resolved)
    result["artifact_verification"] = verification
    result["representation_adaptation"] = dict(_ADAPTATION_CONTRACT)
    return result
