"""Real Kumar2024 E001 study under merged neurOS longitudinal authority.

The acquisition layer is intentionally opt-in: importing QuantumBCI never downloads a
public dataset. ``run_kumar2024_study`` uses neurOS/MOABB only when explicitly invoked,
hashes the original files returned by MOABB ``data_path()``, creates the same deterministic
longitudinal case authority as the neurOS model-ladder protocol, and then evaluates
QuantumBCI's equivalence-first E001 controls on a genuine time-by-channel token surface.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..exporting import verify_run_artifacts
from ..longitudinal import (
    LongitudinalE001CaseResult,
    LongitudinalE001Row,
    evaluate_density_information_gate,
    paired_participant_bootstrap,
    run_longitudinal_e001_case,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _stable_seed(base: int, *parts: Any) -> int:
    """Match the merged neurOS model-ladder split-seed derivation exactly."""

    raw = "|".join([str(base), *(str(part) for part in parts)])
    return int.from_bytes(sha256(raw.encode("utf-8")).digest()[:4], "big")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return _canonical_json(value)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty study result table")
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _package_versions() -> dict[str, str | None]:
    names = (
        "quantum-bci",
        "neuros-core",
        "neuros-models",
        "neuros-foundation",
        "moabb",
        "mne",
        "numpy",
    )
    values: dict[str, str | None] = {}
    for name in names:
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def _flatten_path_values(value: Any) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, os.PathLike)):
        return [Path(value)]
    if isinstance(value, Mapping):
        paths: list[Path] = []
        for item in value.values():
            paths.extend(_flatten_path_values(item))
        return paths
    if isinstance(value, Iterable):
        paths = []
        for item in value:
            paths.extend(_flatten_path_values(item))
        return paths
    return []


def _source_files(value: Any) -> tuple[Path, ...]:
    files: dict[str, Path] = {}
    for raw in _flatten_path_values(value):
        path = raw.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"MOABB data_path returned a missing path: {path}")
        if path.is_file():
            files[str(path)] = path
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file():
                    resolved = child.resolve()
                    files[str(resolved)] = resolved
    if not files:
        raise ValueError(
            "MOABB data_path did not resolve to source files; raw-data fingerprinting "
            "is required for a promoted Kumar2024 study"
        )
    return tuple(sorted(files.values(), key=lambda path: str(path)))


def fingerprint_raw_dataset(
    dataset: Any,
    subjects: Sequence[int],
    *,
    dataset_key: str = "kumar2024",
    dataset_id: str = "moabb-kumar2024",
) -> dict[str, Any]:
    """Hash the original local files returned by MOABB ``data_path``.

    Absolute local paths are never written to the scientific manifest. File content,
    size, and stable relative/basename labels define the fingerprint so moving the
    same downloaded bytes to another machine does not change the dataset identity.
    """

    getter = getattr(dataset, "data_path", None)
    if not callable(getter):
        raise TypeError("dataset must expose callable data_path(subject)")
    normalized_subjects = tuple(sorted(set(int(value) for value in subjects)))
    if not normalized_subjects:
        raise ValueError("subjects must not be empty")

    hash_cache: dict[str, str] = {}
    subject_records: dict[str, Any] = {}
    for subject in normalized_subjects:
        returned = getter(int(subject))
        roots = tuple(path.expanduser().resolve() for path in _flatten_path_values(returned))
        files = _source_files(returned)
        records: list[dict[str, Any]] = []
        for path in files:
            key = str(path)
            digest = hash_cache.get(key)
            if digest is None:
                digest = _sha256_file(path)
                hash_cache[key] = digest

            labels: list[str] = []
            for root in roots:
                if root.is_dir():
                    try:
                        labels.append(str(path.relative_to(root)).replace(os.sep, "/"))
                    except ValueError:
                        continue
                elif root == path:
                    labels.append(path.name)
            label = min(labels, key=lambda item: (len(item), item)) if labels else path.name
            records.append(
                {
                    "name": label,
                    "bytes": int(path.stat().st_size),
                    "sha256": digest,
                }
            )
        records.sort(key=lambda item: (item["name"], item["sha256"], item["bytes"]))
        subject_payload = {
            "subject": int(subject),
            "files": records,
        }
        subject_payload["fingerprint"] = sha256(
            _canonical_json(subject_payload).encode("utf-8")
        ).hexdigest()
        subject_records[str(subject)] = subject_payload

    aggregate_payload = {
        "schema_version": 1,
        "kind": "raw_source_content_fingerprint",
        "dataset_key": str(dataset_key),
        "dataset_id": str(dataset_id),
        "subjects": list(normalized_subjects),
        "by_subject": subject_records,
    }
    aggregate_payload["fingerprint"] = sha256(
        _canonical_json(aggregate_payload).encode("utf-8")
    ).hexdigest()
    return aggregate_payload


@dataclass(frozen=True)
class Kumar2024StudyConfig:
    subjects: tuple[int, ...] = (1, 10)
    held_out_sessions: tuple[str, ...] | None = ("5",)
    budgets_per_class: tuple[int, ...] = (0, 1, 2, 5, 10)
    split_seed: int = 2026
    evaluation_fraction: float = 0.5
    fmin: float = 8.0
    fmax: float = 30.0
    resample_hz: float | None = None
    ridge: float = 1e-3
    covariance_regularization: float = 1e-6

    def __post_init__(self) -> None:
        subjects = tuple(sorted(set(int(value) for value in self.subjects)))
        if len(subjects) < 2:
            raise ValueError("Kumar2024 participant-level study requires at least two subjects")
        if any(value < 1 or value > 18 for value in subjects):
            raise ValueError("Kumar2024 subjects must lie in 1..18")
        budgets = tuple(sorted(set(int(value) for value in self.budgets_per_class)))
        if not budgets or budgets[0] < 0:
            raise ValueError("budgets_per_class must contain non-negative values")
        if 0 not in budgets:
            budgets = (0, *budgets)
        sessions = None
        if self.held_out_sessions is not None:
            sessions = tuple(dict.fromkeys(str(value) for value in self.held_out_sessions))
            if not sessions:
                raise ValueError("held_out_sessions must not be empty when supplied")
            if any(value == "0" for value in sessions):
                raise ValueError("session 0 has no prior-session history and cannot be a prior-policy target")
        if not 0.0 < float(self.evaluation_fraction) < 1.0:
            raise ValueError("evaluation_fraction must lie strictly between 0 and 1")
        if self.fmin <= 0 or self.fmax <= self.fmin:
            raise ValueError("require 0 < fmin < fmax")
        if not np.isfinite(self.ridge) or self.ridge < 0:
            raise ValueError("ridge must be finite and non-negative")
        if not np.isfinite(self.covariance_regularization) or self.covariance_regularization <= 0:
            raise ValueError("covariance_regularization must be finite and positive")
        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(self, "budgets_per_class", budgets)
        object.__setattr__(self, "held_out_sessions", sessions)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "subjects": list(self.subjects),
            "held_out_sessions": None if self.held_out_sessions is None else list(self.held_out_sessions),
            "budgets_per_class": list(self.budgets_per_class),
            "history_policy": "prior",
            "split_seed": int(self.split_seed),
            "evaluation_fraction": float(self.evaluation_fraction),
            "fmin": float(self.fmin),
            "fmax": float(self.fmax),
            "resample_hz": self.resample_hz,
            "ridge": float(self.ridge),
            "covariance_regularization": float(self.covariance_regularization),
        }


def _neuros_authority_api() -> Any:
    try:
        from neuros.foundation_models.longitudinal import (
            chronological_partition,
            make_nested_calibration_split,
            ordered_group_values,
        )
        from neuros.foundation_models.longitudinal_authority import LongitudinalCaseAuthority
        from neuros.foundation_models.moabb_longitudinal import validate_observed_sessions
    except ImportError as exc:  # pragma: no cover - optional real-study dependency
        raise ImportError(
            "Kumar2024 execution requires merged neurOS longitudinal evidence packages. "
            "Install the QuantumBCI neurOS profile and neuros-foundation[evidence]."
        ) from exc

    class API:
        pass

    api = API()
    api.chronological_partition = chronological_partition
    api.make_nested_calibration_split = make_nested_calibration_split
    api.ordered_group_values = ordered_group_values
    api.LongitudinalCaseAuthority = LongitudinalCaseAuthority
    api.validate_observed_sessions = validate_observed_sessions
    return api


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
    """Execute all declared target sessions for one already-collected subject."""

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
        case = run_longitudinal_e001_case(
            data,
            authority,
            token_representation,
            representation_id=representation_id,
            budgets_per_class=config.budgets_per_class,
            upstream_dataset_fingerprint=upstream_dataset_fingerprint,
            quantumbci_source_sha=quantumbci_source_sha,
            neuros_source_sha=neuros_source_sha,
            ridge=config.ridge,
            center_tokens=True,
            covariance_regularization=config.covariance_regularization,
        )
        authorities.append(authority)
        cases.append(case)
    return tuple(authorities), tuple(cases)


def _result_table(cases: Sequence[LongitudinalE001CaseResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        for row in case.rows:
            metadata = dict(row.case_metadata)
            for method, metric in sorted(row.result.metrics.items()):
                rows.append(
                    {
                        "dataset_id": row.dataset_id,
                        "case_id": row.case_id,
                        "subject": metadata.get("subject"),
                        "original_protocol": metadata.get("original_protocol"),
                        "held_out_session": metadata.get(
                            "held_out_session",
                            row.held_out_values[0] if row.held_out_values else None,
                        ),
                        "split_seed": metadata.get("split_seed"),
                        "calibration_per_class": row.calibration_per_class,
                        "method": method,
                        "accuracy": metric.accuracy,
                        "balanced_accuracy": metric.balanced_accuracy,
                        "feature_dimension": row.result.feature_dimensions.get(method),
                        "authority_fingerprint": row.authority_fingerprint,
                        "representation_sha256": row.representation_sha256,
                        "strongest_classical_control": row.result.strongest_classical_control,
                        "density_information_novel": row.result.density_information_novel,
                    }
                )
    return rows


def _prediction_records(cases: Sequence[LongitudinalE001CaseResult]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case in cases:
        for row in case.rows:
            records.append(
                {
                    "case_id": row.case_id,
                    "subject": row.case_metadata.get("subject"),
                    "held_out_session": row.case_metadata.get("held_out_session"),
                    "calibration_per_class": row.calibration_per_class,
                    "test_labels": np.asarray(row.result.test_labels).astype(str).tolist(),
                    "predictions": {
                        name: np.asarray(values).astype(str).tolist()
                        for name, values in sorted(row.result.predictions.items())
                    },
                }
            )
    return records


def _render_report(
    *,
    manifest: Mapping[str, Any],
    result_rows: Sequence[Mapping[str, Any]],
    equivalence_gate: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> str:
    methods = sorted({str(row["method"]) for row in result_rows})
    lines = [
        "# QuantumBCI E001 Kumar2024 study",
        "",
        "## Scientific boundary",
        "",
        "The current density constructor is exactly equivalent to trace-normalized covariance. "
        "This study therefore validates longitudinal evidence behavior and operator/control "
        "semantics; it cannot promote density as carrying additional representation information.",
        "",
        f"- Dataset: `{manifest['dataset_id']}`",
        f"- Participants: {len(manifest['subjects'])}",
        f"- Authority cases: {manifest['authority_cases']}",
        f"- Calibration budgets/class: {manifest['budgets_per_class']}",
        f"- Raw-source fingerprint: `{manifest['raw_dataset_fingerprint']}`",
        f"- QuantumBCI revision: `{manifest['quantumbci_source_sha']}`",
        f"- neurOS revision: `{manifest['neuros_source_sha']}`",
        "",
        "## Mean balanced accuracy across case-budget rows",
        "",
        "| Representation/control | Mean balanced accuracy |",
        "| --- | ---: |",
    ]
    for method in methods:
        values = [
            float(row["balanced_accuracy"])
            for row in result_rows
            if row["method"] == method
        ]
        lines.append(f"| {method} | {np.mean(values):.4f} |")
    lines.extend(
        [
            "",
            "## Equivalence gate",
            "",
            f"- Mathematical equivalence detected: **{equivalence_gate['mathematical_equivalence_detected']}**",
            f"- Density/normalized-covariance prediction identity: **{equivalence_gate['normalized_covariance_prediction_identity']}**",
            f"- Representation-information promotion eligible: **{equivalence_gate['promotion_eligible']}**",
            "",
            "The normalized-covariance participant bootstrap is expected to be exactly zero. "
            "Other controls quantify whether normalization, covariance geometry, pooled statistics, "
            "PCA, or cross-feature deletion change predictive behavior under the same neurOS authority.",
            "",
            "## Participant-level inference",
            "",
        ]
    )
    for control, payload in sorted(bootstrap.items()):
        summaries = payload.get("summaries", [])
        if not summaries:
            continue
        lines.append(f"### {control}")
        lines.append("")
        lines.append("| Budget/class | Participants | Mean density-control delta | 95% bootstrap CI |")
        lines.append("| ---: | ---: | ---: | --- |")
        for item in summaries:
            lines.append(
                f"| {item['calibration_per_class']} | {item['n_units']} | "
                f"{item['observed_mean_delta']:+.4f} | "
                f"[{item['ci_lower']:+.4f}, {item['ci_upper']:+.4f}] |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation ceiling",
            "",
            "A successful run is real-dataset, longitudinal, quantum-inspired evidence only. "
            "It is not evidence for microscopic quantum coherence, entanglement, or physical "
            "quantum computation in neural tissue.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_study_bundle(
    output: Path,
    *,
    config: Kumar2024StudyConfig,
    dataset_spec: Any,
    raw_fingerprint: Mapping[str, Any],
    authorities: Sequence[Any],
    cases: Sequence[LongitudinalE001CaseResult],
    quantumbci_source_sha: str,
    neuros_source_sha: str,
    overwrite: bool,
) -> dict[str, Any]:
    if output.exists():
        if not overwrite and any(output.iterdir()):
            raise FileExistsError(f"study output already contains files: {output}")
        if overwrite:
            shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    all_rows: tuple[LongitudinalE001Row, ...] = tuple(
        row for case in cases for row in case.rows
    )
    if not all_rows:
        raise ValueError("Kumar2024 study produced no E001 rows")
    result_rows = _result_table(cases)
    prediction_records = _prediction_records(cases)
    equivalence_gate = evaluate_density_information_gate(all_rows)

    controls = sorted(set(all_rows[0].result.metrics) - {"density"})
    bootstrap: dict[str, Any] = {}
    for control in controls:
        summaries = paired_participant_bootstrap(
            all_rows,
            control=control,
            inference_key="subject",
            n_resamples=5000,
            seed=config.split_seed,
        )
        bootstrap[control] = {
            "schema_version": 1,
            "control": control,
            "summaries": [item.to_mapping() for item in summaries],
        }

    identity = {
        "schema_version": 1,
        "study": "E001_kumar2024_equivalence_first_longitudinal",
        "dataset_id": dataset_spec.source_id,
        "raw_dataset_fingerprint": raw_fingerprint["fingerprint"],
        "config": config.to_mapping(),
        "quantumbci_source_sha": str(quantumbci_source_sha),
        "neuros_source_sha": str(neuros_source_sha),
        "authority_fingerprints": sorted(
            str(authority.authority_fingerprint) for authority in authorities
        ),
        "case_study_fingerprints": sorted(case.study_fingerprint for case in cases),
    }
    study_fingerprint = sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    run_id = f"E001-kumar2024-{study_fingerprint[:16]}"

    source_revisions = {
        "schema_version": 1,
        "quantumbci_source_sha": str(quantumbci_source_sha),
        "neuros_source_sha": str(neuros_source_sha),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": _package_versions(),
    }
    manifest = {
        "schema_version": 1,
        "evidence_tier": "real_dataset",
        "claim_class": "quantum_inspired",
        "study": identity["study"],
        "dataset_key": dataset_spec.key,
        "dataset_class": dataset_spec.class_name,
        "dataset_id": dataset_spec.source_id,
        "dataset_description": dataset_spec.description,
        "subjects": list(config.subjects),
        "held_out_sessions": (
            None if config.held_out_sessions is None else list(config.held_out_sessions)
        ),
        "budgets_per_class": list(config.budgets_per_class),
        "history_policy": "prior",
        "split_seed_base": int(config.split_seed),
        "evaluation_fraction": float(config.evaluation_fraction),
        "band_hz": [float(config.fmin), float(config.fmax)],
        "resample_hz": config.resample_hz,
        "authority_cases": len(authorities),
        "raw_dataset_fingerprint": raw_fingerprint["fingerprint"],
        "quantumbci_source_sha": str(quantumbci_source_sha),
        "neuros_source_sha": str(neuros_source_sha),
        "study_fingerprint": study_fingerprint,
        "equivalence_gate": equivalence_gate,
        "claim_boundary": [
            "same deterministic neurOS prior-session authority semantics as model ladder",
            "fixed final evaluation examples across calibration budgets",
            "raw source bytes, processed bytes, authority, representation, and source revisions are fingerprinted",
            "density is information-equivalent to normalized covariance for the current constructor",
            "participant-level bootstrap averages repeated cases within participant",
            "real-dataset offline evidence is not physical-quantum or clinical evidence",
        ],
    }

    representation_index = {
        "schema_version": 1,
        "representation": "time-by-channel MOABB epochs",
        "cases": [
            {
                "case_id": case.authority["case_id"],
                "subject": case.authority.get("case_metadata", {}).get("subject"),
                "representation_id": case.representation_id,
                "representation_sha256": case.representation_sha256,
                "study_fingerprint": case.study_fingerprint,
            }
            for case in cases
        ],
    }
    evidence_ledger = {
        "schema_version": 1,
        "study_fingerprint": study_fingerprint,
        "equivalence_gate": equivalence_gate,
        "participant_inference_controls": sorted(bootstrap),
        "information_novelty_promotion_eligible": False,
        "next_scientific_gate": (
            "test a genuinely non-equivalent downstream operator/dynamical/contextual mechanism"
        ),
    }
    run_record = {
        "schema_version": 1,
        "run_id": run_id,
        "title": "QuantumBCI E001 Kumar2024 equivalence-first longitudinal study",
        "experiment_id": "E001_density_geometry",
        "status": "completed",
        "claim_class": "quantum_inspired",
        "evidence_tier": "real_dataset",
        "scientific_fingerprint": study_fingerprint,
        "dataset_id": dataset_spec.source_id,
        "quantumbci_source_sha": str(quantumbci_source_sha),
        "neuros_source_sha": str(neuros_source_sha),
    }

    paths = {
        "run": output / "run.json",
        "manifest": output / "study_manifest.json",
        "revisions": output / "source_revisions.json",
        "dataset": output / "dataset_fingerprint.json",
        "authority": output / "neuros_authority.json",
        "representations": output / "representation_index.json",
        "cases": output / "case_results.json",
        "results": output / "results.csv",
        "predictions": output / "predictions.jsonl",
        "bootstrap": output / "bootstrap_metrics.json",
        "ledger": output / "evidence_ledger.json",
        "report": output / "report.md",
        "hashes": output / "artifact_hashes.json",
    }

    _json_dump(paths["run"], run_record)
    _json_dump(paths["manifest"], manifest)
    _json_dump(paths["revisions"], source_revisions)
    _json_dump(paths["dataset"], raw_fingerprint)
    _json_dump(
        paths["authority"],
        {"schema_version": 1, "cases": [authority.to_dict() for authority in authorities]},
    )
    _json_dump(paths["representations"], representation_index)
    _json_dump(
        paths["cases"],
        {"schema_version": 1, "cases": [case.to_mapping() for case in cases]},
    )
    _write_csv(paths["results"], result_rows)
    with paths["predictions"].open("w", encoding="utf-8") as handle:
        for record in prediction_records:
            handle.write(_canonical_json(record) + "\n")
    _json_dump(paths["bootstrap"], {"schema_version": 1, "controls": bootstrap})
    _json_dump(paths["ledger"], evidence_ledger)
    paths["report"].write_text(
        _render_report(
            manifest=manifest,
            result_rows=result_rows,
            equivalence_gate=equivalence_gate,
            bootstrap=bootstrap,
        ),
        encoding="utf-8",
    )
    hashes = {
        path.name: _sha256_file(path)
        for key, path in paths.items()
        if key != "hashes"
    }
    _json_dump(paths["hashes"], hashes)
    verification = verify_run_artifacts(output)
    if not verification["valid"]:
        raise RuntimeError(f"study artifact verification failed: {verification}")

    return {
        "run_id": run_id,
        "output": str(output),
        "study_fingerprint": study_fingerprint,
        "authority_cases": len(authorities),
        "result_rows": len(result_rows),
        "subjects": list(config.subjects),
        "raw_dataset_fingerprint": raw_fingerprint["fingerprint"],
        "equivalence_promotion_eligible": equivalence_gate["promotion_eligible"],
        "artifact_verification": verification,
    }


def run_kumar2024_study(
    output: str | Path,
    *,
    config: Kumar2024StudyConfig,
    quantumbci_source_sha: str,
    neuros_source_sha: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Download/collect selected Kumar2024 subjects and write a verified E001 bundle."""

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
            upstream_dataset_fingerprint=raw_fingerprint["fingerprint"],
            quantumbci_source_sha=quantumbci_source_sha,
            neuros_source_sha=neuros_source_sha,
        )
        authorities.extend(subject_authorities)
        cases.extend(subject_cases)

    return _write_study_bundle(
        Path(output).resolve(),
        config=config,
        dataset_spec=dataset_spec,
        raw_fingerprint=raw_fingerprint,
        authorities=authorities,
        cases=cases,
        quantumbci_source_sha=quantumbci_source_sha,
        neuros_source_sha=neuros_source_sha,
        overwrite=overwrite,
    )
