"""Portable research-object and BIDS-aware exports for QuantumBCI run bundles."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import mimetypes
from pathlib import Path
import shutil
from typing import Any
import zipfile


def _package_version() -> str:
    try:
        return version("quantum-bci")
    except PackageNotFoundError:
        return "source-checkout"


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_run_artifacts(run_dir: str | Path) -> dict[str, Any]:
    """Verify one run against its immutable ``artifact_hashes.json`` ledger."""

    directory = Path(run_dir)
    ledger_path = directory / "artifact_hashes.json"
    if not ledger_path.is_file():
        raise FileNotFoundError(f"missing artifact hash ledger: {ledger_path}")
    expected = json.loads(ledger_path.read_text(encoding="utf-8"))
    if not isinstance(expected, dict):
        raise ValueError("artifact_hashes.json must contain a JSON object")
    missing: list[str] = []
    mismatched: list[str] = []
    verified: list[str] = []
    for name, expected_hash in sorted(expected.items()):
        path = directory / name
        if not path.is_file():
            missing.append(name)
            continue
        actual = _hash_file(path)
        if actual != str(expected_hash):
            mismatched.append(name)
        else:
            verified.append(name)
    return {
        "valid": not missing and not mismatched,
        "verified": verified,
        "missing": missing,
        "mismatched": mismatched,
    }


def export_run_ro_crate(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    archive: bool = False,
) -> Path:
    """Export a run as a self-contained minimal RO-Crate 1.3 research object.

    The export uses the RO-Crate 1.3 context and required metadata/root entities.
    QuantumBCI does not claim external validator certification; consumers should run
    their preferred RO-Crate validator when publication workflows require it.
    """

    source = Path(run_dir)
    verification = verify_run_artifacts(source)
    if not verification["valid"]:
        raise ValueError(
            "run artifact verification failed before export: "
            f"missing={verification['missing']} mismatched={verification['mismatched']}"
        )
    run_record = json.loads((source / "run.json").read_text(encoding="utf-8"))
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"export destination already exists: {destination}")
    payload_dir = destination / "data"
    payload_dir.mkdir(parents=True)

    payload_files: list[Path] = []
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        target = payload_dir / path.name
        shutil.copy2(path, target)
        payload_files.append(target)

    has_part = [{"@id": f"data/{path.name}"} for path in payload_files]
    graph: list[dict[str, Any]] = [
        {
            "@id": "ro-crate-metadata.json",
            "@type": "CreativeWork",
            "about": {"@id": "."},
            "conformsTo": {"@id": "https://w3id.org/ro/crate/1.3"},
        },
        {
            "@id": ".",
            "@type": "Dataset",
            "name": str(run_record.get("title") or run_record.get("experiment_id") or "QuantumBCI evidence run"),
            "description": "Portable QuantumBCI research evidence bundle.",
            "datePublished": datetime.now(timezone.utc).date().isoformat(),
            "hasPart": has_part,
            "mentions": {"@id": "https://github.com/sidhulyalkar/QuantumBCI"},
        },
        {
            "@id": "https://github.com/sidhulyalkar/QuantumBCI",
            "@type": "SoftwareApplication",
            "name": "QuantumBCI",
            "version": _package_version(),
            "url": "https://github.com/sidhulyalkar/QuantumBCI",
        },
    ]
    for path in payload_files:
        graph.append(
            {
                "@id": f"data/{path.name}",
                "@type": "File",
                "name": path.name,
                "contentSize": str(path.stat().st_size),
                "encodingFormat": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                "sha256": _hash_file(path),
            }
        )
    metadata = {
        "@context": "https://w3id.org/ro/crate/1.3/context",
        "@graph": graph,
    }
    (destination / "ro-crate-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "ro-crate-preview.html").write_text(
        _crate_preview(run_record, payload_files), encoding="utf-8"
    )

    if not archive:
        return destination
    archive_path = destination.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(destination))
    return archive_path


def _crate_preview(run_record: dict[str, Any], files: list[Path]) -> str:
    items = "".join(f"<li><a href='data/{path.name}'>{path.name}</a></li>" for path in files)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>QuantumBCI Research Object</title><style>body{{font-family:system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;line-height:1.5}}code{{overflow-wrap:anywhere}}</style></head><body><h1>{run_record.get('title') or run_record.get('experiment_id') or 'QuantumBCI evidence run'}</h1><p>Portable research object exported by QuantumBCI {_package_version()}.</p><p>Run: <code>{run_record.get('run_id','unknown')}</code><br>Fingerprint: <code>{run_record.get('scientific_fingerprint','unknown')}</code><br>Claim class: {run_record.get('claim_class','unknown')}<br>Evidence tier: {run_record.get('evidence_tier','unknown')}</p><h2>Payload</h2><ul>{items}</ul><p>Machine-readable metadata: <a href='ro-crate-metadata.json'>ro-crate-metadata.json</a></p></body></html>"""


def export_run_bids_derivative_container(
    run_dir: str | Path,
    bids_root: str | Path,
    *,
    bids_version: str,
    source_dataset_url: str | None = None,
) -> Path:
    """Place a run in a BIDS-aware derivative dataset container.

    This creates the derivative-level ``dataset_description.json`` required for a
    BIDS derivative dataset and preserves the QuantumBCI evidence bundle under an
    ``evidence/`` namespace. The generic QuantumBCI evidence files are not claimed to
    implement a modality-specific standardized BIDS derivative datatype.
    """

    source = Path(run_dir)
    verification = verify_run_artifacts(source)
    if not verification["valid"]:
        raise ValueError("run artifact verification failed before BIDS-aware export")
    version_text = str(bids_version).strip()
    if not version_text:
        raise ValueError("bids_version must be provided explicitly")
    root = Path(bids_root) / "derivatives" / "quantumbci"
    root.mkdir(parents=True, exist_ok=True)
    description_path = root / "dataset_description.json"
    description: dict[str, Any] = {
        "Name": "QuantumBCI evidence derivatives",
        "BIDSVersion": version_text,
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "QuantumBCI",
                "Version": _package_version(),
                "Description": "Falsifiable quantum-inspired neural representation evidence workbench",
            }
        ],
    }
    if source_dataset_url:
        description["SourceDatasets"] = [{"URL": str(source_dataset_url)}]
    if description_path.exists():
        existing = json.loads(description_path.read_text(encoding="utf-8"))
        if existing.get("Name") != description["Name"]:
            raise ValueError(f"refusing to overwrite unrelated derivative dataset: {description_path}")
    description_path.write_text(json.dumps(description, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    run_record = json.loads((source / "run.json").read_text(encoding="utf-8"))
    run_id = str(run_record.get("run_id") or source.name)
    target = root / "evidence" / run_id
    if target.exists():
        raise FileExistsError(f"BIDS-aware evidence target already exists: {target}")
    shutil.copytree(source, target)
    (target / "README.md").write_text(
        "# QuantumBCI evidence bundle\n\n"
        "This directory is stored inside a BIDS derivative dataset container for discovery and provenance. "
        "Its generic evidence files are not asserted to be a modality-specific standardized BIDS derivative datatype.\n",
        encoding="utf-8",
    )
    return target
