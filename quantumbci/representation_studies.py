"""Portable lane bundles for BMRB-Representation.

This module turns already-authority-bound longitudinal E001 case results into a small,
closed-world research artifact. It deliberately does not own dataset acquisition or
foundation-model downloads. Any frozen encoder can produce the representation tensor;
neurOS remains the authority for participant/session splits and processed-data identity.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

import numpy as np

from .exporting import verify_run_artifacts
from .longitudinal import LongitudinalE001CaseResult

E001_REPRESENTATION_LANE_SCHEMA = "quantumbci.e001-representation-lane.v1"


def _required_text(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_frozen_epochs(
    epochs: np.ndarray,
    encoder: Any,
    *,
    sample_rate_hz: float,
) -> np.ndarray:
    """Encode independent epochs with a frozen encoder protocol.

    ``encoder`` must expose ``encode(epoch, sample_rate_hz=...)`` and return either
    ``(tokens, features)`` or a one-item ``(1, tokens, features)`` batch. The helper
    intentionally calls no fit/adapt/train method. Representation fitting or label-aware
    adaptation belongs upstream and must have its own frozen evidence authority.
    """

    values = np.asarray(epochs)
    if values.ndim < 3:
        raise ValueError("epochs must have shape (samples, channels, time) or richer")
    if not np.isfinite(float(sample_rate_hz)) or float(sample_rate_hz) <= 0:
        raise ValueError("sample_rate_hz must be finite and positive")
    encode = getattr(encoder, "encode", None)
    if not callable(encode):
        raise TypeError("encoder must expose callable encode(epoch, sample_rate_hz=...)")

    rows: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None
    for index, epoch in enumerate(values):
        output = np.asarray(encode(epoch, sample_rate_hz=float(sample_rate_hz)))
        if output.ndim == 3:
            if output.shape[0] != 1:
                raise ValueError(
                    "per-epoch frozen encoder returned a multi-item batch; "
                    f"sample={index}, shape={output.shape}"
                )
            output = output[0]
        if output.ndim != 2:
            raise ValueError(
                "frozen encoder output must resolve to (tokens, features); "
                f"sample={index}, shape={output.shape}"
            )
        if output.shape[0] < 2 or output.shape[1] < 2:
            raise ValueError("E001 representations require at least two tokens and features")
        if not np.all(np.isfinite(output)):
            raise ValueError(f"frozen encoder returned non-finite values for sample {index}")
        shape = (int(output.shape[0]), int(output.shape[1]))
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(
                "frozen encoder token/feature shape changed across epochs; "
                f"expected={expected_shape}, observed={shape}, sample={index}"
            )
        rows.append(np.asarray(output, dtype=float))
    if not rows:
        raise ValueError("epochs must contain at least one sample")
    return np.stack(rows)


def write_e001_representation_lane_bundle(
    cases: Iterable[LongitudinalE001CaseResult],
    output_dir: str | Path,
    *,
    study_id: str,
    representation_family: str,
    model_id: str | None = None,
    model_revision: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a verified, portable representation lane from E001 case results."""

    materialized = tuple(cases)
    if not materialized:
        raise ValueError("representation lane bundle requires case results")
    study_name = _required_text("study_id", study_id)
    family = _required_text("representation_family", representation_family)
    model_name = None if model_id is None else _required_text("model_id", model_id)
    model_rev = None if model_revision is None else _required_text("model_revision", model_revision)
    if model_name is not None and model_rev is None:
        raise ValueError("model_revision is required when model_id is supplied")

    representation_ids = {str(case.representation_id) for case in materialized}
    if len(representation_ids) != 1:
        raise ValueError("one representation lane bundle must use one representation_id")
    case_ids = [str(case.authority.get("case_id", "")) for case in materialized]
    if any(not value for value in case_ids):
        raise ValueError("every E001 case must expose authority.case_id")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("representation lane bundle contains duplicate case_id values")

    provenance_keys = ("upstream_dataset_fingerprint", "quantumbci_source_sha", "neuros_source_sha")
    provenance_values: dict[str, set[str]] = {key: set() for key in provenance_keys}
    for case in materialized:
        for key in provenance_keys:
            value = str(case.provenance.get(key, "")).strip()
            if not value:
                raise ValueError(f"case {case.authority.get('case_id')!r} lacks provenance {key!r}")
            provenance_values[key].add(value)
    if any(len(values) != 1 for values in provenance_values.values()):
        raise ValueError("representation lane cases do not share one frozen provenance boundary")

    identity = {
        "schema_version": 1,
        "artifact_role": E001_REPRESENTATION_LANE_SCHEMA,
        "study_id": study_name,
        "representation_id": next(iter(representation_ids)),
        "representation_family": family,
        "model_id": model_name,
        "model_revision": model_rev,
        "case_study_fingerprints": sorted(str(case.study_fingerprint) for case in materialized),
        "provenance": {key: next(iter(values)) for key, values in provenance_values.items()},
        "metadata": dict(metadata or {}),
    }
    scientific_fingerprint = sha256(
        b"quantumbci.e001-representation-lane.v1\0"
        + _canonical_json(identity).encode("utf-8")
    ).hexdigest()

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        if overwrite:
            shutil.rmtree(output)
        elif any(output.iterdir()):
            raise FileExistsError(f"representation lane output already contains files: {output}")
    output.mkdir(parents=True, exist_ok=True)

    run_record = {
        "schema_version": 1,
        "run_id": f"BMRB-representation-lane-{scientific_fingerprint[:16]}",
        "title": f"QuantumBCI E001 representation lane: {study_name}",
        "experiment_id": "E001_density_geometry",
        "status": "completed",
        "claim_class": "quantum_inspired",
        "evidence_tier": "real_dataset_or_frozen_representation",
        "scientific_fingerprint": scientific_fingerprint,
        "representation_family": family,
        "model_id": model_name,
        "model_revision": model_rev,
    }
    manifest = {
        **identity,
        "scientific_fingerprint": scientific_fingerprint,
        "case_count": len(materialized),
        "participant_ids": sorted(
            {
                str(case.authority.get("case_metadata", {}).get("subject"))
                for case in materialized
                if case.authority.get("case_metadata", {}).get("subject") is not None
            }
        ),
        "claim_boundary": [
            "representation is frozen before E001 evaluation",
            "neurOS authority owns participant/session/calibration/evaluation identity",
            "current density constructor remains quantum-inspired",
            "cross-representation recurrence does not imply information novelty",
            "physical-quantum claims require an independent witness protocol",
        ],
    }
    case_payload = {
        "schema_version": 1,
        "artifact_role": "e001_representation_lane_cases",
        "scientific_fingerprint": scientific_fingerprint,
        "cases": [case.to_mapping(include_predictions=False) for case in materialized],
    }
    report_lines = [
        f"# E001 representation lane: {study_name}",
        "",
        f"- Representation: `{identity['representation_id']}`",
        f"- Family: `{family}`",
        f"- Model: `{model_name or 'none/raw'}`",
        f"- Model revision: `{model_rev or 'n/a'}`",
        f"- Cases: {len(materialized)}",
        f"- Scientific fingerprint: `{scientific_fingerprint}`",
        "",
        "This bundle is an authority-bound representation lane for BMRB-Representation. "
        "It records predictive/control behavior in one frozen representation space and makes "
        "no cross-representation or physical-quantum claim by itself.",
        "",
    ]

    files = {
        "run.json": run_record,
        "study_manifest.json": manifest,
        "case_results.json": case_payload,
    }
    for name, payload in files.items():
        (output / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    (output / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    ledger = {
        name: _sha256_file(output / name)
        for name in ("run.json", "study_manifest.json", "case_results.json", "report.md")
    }
    (output / "artifact_hashes.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verification = verify_run_artifacts(output)
    if not verification["valid"]:
        raise RuntimeError(f"representation lane artifact verification failed: {verification}")
    return {
        "run_id": run_record["run_id"],
        "output": str(output),
        "scientific_fingerprint": scientific_fingerprint,
        "case_count": len(materialized),
        "representation_id": identity["representation_id"],
        "representation_family": family,
        "artifact_verification": verification,
    }
