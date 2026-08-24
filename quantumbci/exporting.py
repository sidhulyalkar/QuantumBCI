"""Portable research-object and BIDS-aware exports for QuantumBCI run bundles."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import html
from importlib.metadata import PackageNotFoundError, version
import json
import mimetypes
from pathlib import Path
import shutil
from typing import Any
import zipfile


RO_CRATE_VERSION = "1.3"
RO_CRATE_PROFILE = f"https://w3id.org/ro/crate/{RO_CRATE_VERSION}"
RO_CRATE_CONTEXT = f"{RO_CRATE_PROFILE}/context"


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


def _safe_run_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("run_id must not be empty")
    if Path(text).name != text or text in {".", ".."}:
        raise ValueError(f"unsafe run_id: {text!r}")
    if any(ch in text for ch in ("/", "\\", "\x00")):
        raise ValueError(f"unsafe run_id: {text!r}")
    return text


def _validate_ledger_name(name: Any) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError("artifact ledger keys must be non-empty strings")
    if Path(name).name != name or name in {".", "..", "artifact_hashes.json"}:
        raise ValueError(f"unsafe artifact ledger entry: {name!r}")
    return name


def verify_run_artifacts(run_dir: str | Path) -> dict[str, Any]:
    """Verify one run against its ``artifact_hashes.json`` integrity ledger.

    Verification is deliberately closed-world: missing, modified, *or unexpected*
    top-level files invalidate the run. The SHA-256 ledger detects corruption and
    post-run edits; it is not a cryptographic signature and does not establish who
    authored the bundle.
    """

    directory = Path(run_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {directory}")
    ledger_path = directory / "artifact_hashes.json"
    if not ledger_path.is_file():
        raise FileNotFoundError(f"missing artifact hash ledger: {ledger_path}")
    expected_raw = json.loads(ledger_path.read_text(encoding="utf-8"))
    if not isinstance(expected_raw, dict):
        raise ValueError("artifact_hashes.json must contain a JSON object")

    expected: dict[str, str] = {}
    invalid_entries: list[str] = []
    for raw_name, raw_hash in sorted(expected_raw.items(), key=lambda item: str(item[0])):
        try:
            name = _validate_ledger_name(raw_name)
        except ValueError:
            invalid_entries.append(str(raw_name))
            continue
        digest = str(raw_hash).lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            invalid_entries.append(name)
            continue
        expected[name] = digest

    actual_files = {
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    unexpected = sorted(actual_files - set(expected))
    missing: list[str] = []
    mismatched: list[str] = []
    verified: list[str] = []
    for name, expected_hash in sorted(expected.items()):
        path = directory / name
        if not path.is_file():
            missing.append(name)
            continue
        actual = _hash_file(path)
        if actual != expected_hash:
            mismatched.append(name)
        else:
            verified.append(name)
    return {
        "valid": not missing and not mismatched and not unexpected and not invalid_entries,
        "verified": verified,
        "missing": missing,
        "mismatched": mismatched,
        "unexpected": unexpected,
        "invalid_ledger_entries": invalid_entries,
        "integrity_scope": "sha256-checksum-not-authenticity-signature",
    }


def export_run_ro_crate(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    archive: bool = False,
) -> Path:
    """Export a run as a self-contained minimal RO-Crate 1.3 research object.

    The export uses the RO-Crate 1.3 Recommendation context and required
    metadata/root entities. QuantumBCI does not claim external validator
    certification; consumers should run their required RO-Crate validator for
    publication or repository deposition.
    """

    source = Path(run_dir)
    verification = verify_run_artifacts(source)
    if not verification["valid"]:
        raise ValueError(
            "run artifact verification failed before export: "
            f"missing={verification['missing']} mismatched={verification['mismatched']} "
            f"unexpected={verification['unexpected']} "
            f"invalid_ledger_entries={verification['invalid_ledger_entries']}"
        )
    run_record = json.loads((source / "run.json").read_text(encoding="utf-8"))
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"export destination already exists: {destination}")
    archive_path = destination.with_suffix(".zip")
    if archive and archive_path.exists():
        raise FileExistsError(f"export archive already exists: {archive_path}")
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
            "conformsTo": {"@id": RO_CRATE_PROFILE},
        },
        {
            "@id": ".",
            "@type": "Dataset",
            "name": str(run_record.get("title") or run_record.get("experiment_id") or "QuantumBCI evidence run"),
            "description": "Portable QuantumBCI research evidence bundle.",
            "datePublished": datetime.now(timezone.utc).date().isoformat(),
            "conformsTo": {"@id": RO_CRATE_PROFILE},
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
        "@context": RO_CRATE_CONTEXT,
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
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(destination))
    return archive_path


def _crate_preview(run_record: dict[str, Any], files: list[Path]) -> str:
    title = html.escape(str(run_record.get("title") or run_record.get("experiment_id") or "QuantumBCI evidence run"))
    run_id = html.escape(str(run_record.get("run_id", "unknown")))
    fingerprint = html.escape(str(run_record.get("scientific_fingerprint", "unknown")))
    claim_class = html.escape(str(run_record.get("claim_class", "unknown")))
    evidence_tier = html.escape(str(run_record.get("evidence_tier", "unknown")))
    items = "".join(
        f"<li><a href='data/{html.escape(path.name, quote=True)}'>{html.escape(path.name)}</a></li>"
        for path in files
    )
    package_version = html.escape(_package_version())
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>QuantumBCI Research Object</title><style>body{{font-family:system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;line-height:1.5}}code{{overflow-wrap:anywhere}}</style></head><body><h1>{title}</h1><p>Portable research object exported by QuantumBCI {package_version}.</p><p>Run: <code>{run_id}</code><br>Fingerprint: <code>{fingerprint}</code><br>Claim class: {claim_class}<br>Evidence tier: {evidence_tier}</p><h2>Payload</h2><ul>{items}</ul><p>Machine-readable metadata: <a href='ro-crate-metadata.json'>ro-crate-metadata.json</a></p><p><small>SHA-256 checks verify bundle integrity; they are not an authorship signature.</small></p></body></html>"""


def export_run_bids_derivative_container(
    run_dir: str | Path,
    bids_root: str | Path,
    *,
    bids_version: str,
    source_dataset_url: str | None = None,
) -> Path:
    """Place a run in a BIDS-aware derivative dataset container.

    This creates derivative-level ``dataset_description.json`` metadata and
    preserves the QuantumBCI evidence bundle under an ``evidence/`` namespace.
    The generic QuantumBCI evidence files are not claimed to implement a
    modality-specific standardized BIDS derivative datatype.
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
                "CodeURL": "https://github.com/sidhulyalkar/QuantumBCI",
            }
        ],
    }
    if source_dataset_url:
        description["SourceDatasets"] = [{"URL": str(source_dataset_url)}]
    if description_path.exists():
        existing = json.loads(description_path.read_text(encoding="utf-8"))
        if existing.get("Name") != description["Name"] or existing.get("DatasetType") != "derivative":
            raise ValueError(f"refusing to overwrite unrelated derivative dataset: {description_path}")
        if str(existing.get("BIDSVersion", "")) != version_text:
            raise ValueError(
                "existing QuantumBCI derivative container declares a different BIDSVersion: "
                f"{existing.get('BIDSVersion')!r} != {version_text!r}"
            )
        description = existing
        if source_dataset_url:
            sources = list(description.get("SourceDatasets") or [])
            candidate = {"URL": str(source_dataset_url)}
            if candidate not in sources:
                sources.append(candidate)
            description["SourceDatasets"] = sources
    description_path.write_text(json.dumps(description, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    run_record = json.loads((source / "run.json").read_text(encoding="utf-8"))
    run_id = _safe_run_id(run_record.get("run_id") or source.name)
    target = root / "evidence" / run_id
    if target.exists():
        raise FileExistsError(f"BIDS-aware evidence target already exists: {target}")
    shutil.copytree(source, target)
    (target / "README.md").write_text(
        "# QuantumBCI evidence bundle\n\n"
        "This directory is stored inside a BIDS derivative dataset container for discovery and provenance. "
        "Its generic evidence files are not asserted to be a modality-specific standardized BIDS derivative datatype.\n\n"
        "The source run passed its QuantumBCI SHA-256 integrity ledger before export. This is an integrity check, not an authorship signature.\n",
        encoding="utf-8",
    )
    (target / "quantumbci_export.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "container": "bids-aware-derivatives",
                "standardized_modality_derivative": False,
                "bids_version": version_text,
                "source_run_id": run_id,
                "source_scientific_fingerprint": run_record.get("scientific_fingerprint"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return target
