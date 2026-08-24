"""Local-first research workbench, artifact registry, and deterministic smoke study."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Mapping

import numpy as np

from .benchmarking import IndexSplit, benchmark_density_embeddings
from .integrations.neuros import neuros_integration_status


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_hash(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _distribution_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


@dataclass(frozen=True)
class WorkbenchConfig:
    """User-facing local workbench configuration."""

    artifact_root: Path = Path(".quantumbci/runs")
    default_seed: int = 0
    source_sha: str = "working-tree"

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> "WorkbenchConfig":
        if int(value.get("schema_version", 1)) != 1:
            raise ValueError("Unsupported workbench config schema_version")
        root = Path(str(value.get("artifact_root", ".quantumbci/runs"))).expanduser()
        if base_dir is not None and not root.is_absolute():
            root = (base_dir / root).resolve()
        seed = int(value.get("default_seed", 0))
        source_sha = str(value.get("source_sha", "working-tree")).strip()
        if not source_sha:
            raise ValueError("source_sha must not be empty")
        return cls(artifact_root=root, default_seed=seed, source_sha=source_sha)

    @classmethod
    def from_file(cls, path: str | Path) -> "WorkbenchConfig":
        config_path = Path(path)
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Workbench config root must be a JSON object")
        return cls.from_mapping(payload, base_dir=config_path.parent)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_root": str(self.artifact_root),
            "default_seed": int(self.default_seed),
            "source_sha": self.source_sha,
        }


def write_default_config(path: str | Path, *, force: bool = False) -> Path:
    """Create a minimal config suitable for local research work."""

    config_path = Path(path)
    if config_path.exists() and not force:
        raise FileExistsError(f"{config_path} already exists; pass --force to replace it")
    _write_json(config_path, WorkbenchConfig().to_mapping())
    return config_path


def load_config(path: str | Path | None = None) -> WorkbenchConfig:
    """Load a config, defaulting to ``quantumbci.json`` when present."""

    if path is not None:
        return WorkbenchConfig.from_file(path)
    default = Path("quantumbci.json")
    if default.exists():
        return WorkbenchConfig.from_file(default)
    return WorkbenchConfig()


class RunStore:
    """Filesystem-backed run registry with human-readable artifacts."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def ensure(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def create(self, experiment_id: str, fingerprint: str) -> tuple[str, Path]:
        self.ensure()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in experiment_id)
        run_id = f"{timestamp}_{safe_id}_{fingerprint[:10]}"
        path = self.root / run_id
        path.mkdir(parents=False, exist_ok=False)
        return run_id, path

    def records(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        rows: list[dict[str, Any]] = []
        for directory in sorted((p for p in self.root.iterdir() if p.is_dir()), reverse=True):
            record = directory / "run.json"
            if not record.exists():
                continue
            try:
                payload = json.loads(record.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            payload["run_dir"] = str(directory)
            rows.append(payload)
        return rows

    def load(self, run_id: str) -> dict[str, Any]:
        path = self.root / run_id / "run.json"
        if not path.exists():
            raise FileNotFoundError(f"Unknown run id: {run_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["run_dir"] = str(path.parent)
        return payload


@dataclass(frozen=True)
class SmokeResult:
    run_id: str
    run_dir: Path
    scientific_fingerprint: str
    metrics: Mapping[str, Any]


def _embedding_window(
    label: int,
    *,
    subject: int,
    session: int,
    sample: int,
    rng: np.random.Generator,
    tokens: int = 32,
) -> np.ndarray:
    """Generate a correlation-coded latent window for mechanism recovery.

    The two classes have deliberately matched marginal variance but opposite
    feature-0/feature-1 correlation. This is a software/mechanism sanity test,
    not a physiological EEG simulator.
    """

    t = np.linspace(0.0, 2.0 * np.pi, tokens, endpoint=False)
    phase = 0.19 * sample + 0.11 * subject + 0.07 * session
    a = np.sin(t + phase)
    b = np.cos(2.0 * t - 0.5 * phase)
    sign = 1.0 if int(label) == 1 else -1.0
    x = np.stack(
        [a, sign * a, b, np.sin(3.0 * t + 0.25 * phase)],
        axis=1,
    )
    scale = 1.0 + 0.025 * subject + 0.015 * session
    x = x * scale
    x += rng.normal(0.0, 0.025, size=x.shape)
    return x


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_hashes(run_dir: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for path in sorted(run_dir.iterdir()):
        if path.is_file() and path.name != "artifact_hashes.json":
            values[path.name] = _hash_file(path)
    return values


def _render_html_report(run_id: str, fingerprint: str, metrics: Mapping[str, Any]) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{row['subject']}</td>"
        f"<td>{row['density_balanced_accuracy']:.3f}</td>"
        f"<td>{row['diagonal_balanced_accuracy']:.3f}</td>"
        f"<td>{row['offdiagonal_ablated_balanced_accuracy']:.3f}</td>"
        f"<td>{row['density_minus_ablated']:+.3f}</td>"
        "</tr>"
        for row in metrics["per_subject"]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QuantumBCI smoke report</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:980px;margin:40px auto;padding:0 20px;line-height:1.5}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}
.card{{border:1px solid #ddd;border-radius:12px;padding:16px}}
.value{{font-size:2rem;font-weight:700}}
table{{border-collapse:collapse;width:100%;margin-top:24px}}
th,td{{border-bottom:1px solid #ddd;padding:8px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}
code{{overflow-wrap:anywhere}}
.note{{background:#f5f5f5;border-radius:10px;padding:14px}}
</style>
</head>
<body>
<h1>QuantumBCI density smoke</h1>
<p><code>{run_id}</code></p>
<div class="cards">
<div class="card"><div>Density BA</div><div class="value">{metrics['density_balanced_accuracy']:.3f}</div></div>
<div class="card"><div>Diagonal control</div><div class="value">{metrics['diagonal_balanced_accuracy']:.3f}</div></div>
<div class="card"><div>Off-diagonal ablation</div><div class="value">{metrics['offdiagonal_ablated_balanced_accuracy']:.3f}</div></div>
<div class="card"><div>Mechanism delta</div><div class="value">{metrics['density_minus_ablated']:+.3f}</div></div>
</div>
<p class="note"><strong>Interpretation ceiling:</strong> synthetic sanity / quantum-inspired.
This checks that the pipeline can recover a deliberately correlation-coded mechanism. It is not
empirical neural evidence and is not evidence of a physically quantum brain mechanism.</p>
<h2>Participant-like cases</h2>
<table><thead><tr><th>Case</th><th>Density</th><th>Diagonal</th><th>Ablated</th><th>Δ mechanism</th></tr></thead>
<tbody>{rows}</tbody></table>
<h2>Provenance</h2>
<p>Scientific fingerprint: <code>{fingerprint}</code></p>
</body></html>
"""


def run_density_smoke(
    config: WorkbenchConfig,
    *,
    seed: int | None = None,
    subjects: int = 6,
    samples_per_class_session: int = 10,
) -> SmokeResult:
    """Run a deterministic end-to-end density-mechanism sanity study.

    The smoke study intentionally embeds its class signal in cross-feature
    correlation, making off-diagonal deletion a meaningful falsification check.
    It qualifies plumbing and mechanism recovery only; it is never promoted as
    empirical neural evidence.
    """

    resolved_seed = config.default_seed if seed is None else int(seed)
    spec = {
        "schema_version": 1,
        "experiment_id": "SMOKE_density_geometry",
        "claim_class": "quantum_inspired",
        "evidence_tier": "synthetic_sanity",
        "generator": "correlation-coded-latents-v1",
        "subjects": int(subjects),
        "sessions": 3,
        "train_sessions": [0, 1],
        "test_sessions": [2],
        "samples_per_class_session": int(samples_per_class_session),
        "seed": resolved_seed,
        "source_sha": config.source_sha,
    }
    scientific_fingerprint = _json_hash(spec)
    run_id, run_dir = RunStore(config.artifact_root).create(
        spec["experiment_id"], scientific_fingerprint
    )
    started_at = datetime.now(timezone.utc).isoformat()

    rng = np.random.default_rng(resolved_seed)
    predictions: list[dict[str, Any]] = []
    per_subject: list[dict[str, Any]] = []

    for subject in range(int(subjects)):
        windows: list[np.ndarray] = []
        labels: list[int] = []
        sessions: list[int] = []
        for session in range(3):
            for label in (0, 1):
                for sample in range(int(samples_per_class_session)):
                    windows.append(
                        _embedding_window(
                            label,
                            subject=subject,
                            session=session,
                            sample=sample,
                            rng=rng,
                        )
                    )
                    labels.append(label)
                    sessions.append(session)

        embeddings = np.stack(windows)
        y = np.asarray(labels, dtype=int)
        session_array = np.asarray(sessions, dtype=int)
        split = IndexSplit(
            train_indices=np.flatnonzero(session_array < 2),
            test_indices=np.flatnonzero(session_array == 2),
            name=f"subject-{subject}-prior-sessions-to-session-2",
        )
        result = benchmark_density_embeddings(embeddings, y, split)
        row = {
            "subject": subject,
            "density_balanced_accuracy": result.density.balanced_accuracy,
            "diagonal_balanced_accuracy": result.diagonal_control.balanced_accuracy,
            "pooled_balanced_accuracy": result.pooled_control.balanced_accuracy,
            "offdiagonal_ablated_balanced_accuracy": (
                result.offdiagonal_ablation.balanced_accuracy
            ),
            "density_minus_diagonal": result.density_minus_diagonal,
            "density_minus_ablated": result.density_minus_ablation,
        }
        per_subject.append(row)
        test_indices = split.test_indices
        for local_index, dataset_index in enumerate(test_indices.tolist()):
            predictions.append(
                {
                    "subject": subject,
                    "test_session": 2,
                    "dataset_index": dataset_index,
                    "label": int(y[dataset_index]),
                    "density_prediction": int(result.predictions["density"][local_index]),
                    "diagonal_prediction": int(
                        result.predictions["diagonal_control"][local_index]
                    ),
                    "pooled_prediction": int(
                        result.predictions["pooled_control"][local_index]
                    ),
                    "offdiagonal_ablated_prediction": int(
                        result.predictions["offdiagonal_ablation"][local_index]
                    ),
                }
            )

    def mean(name: str) -> float:
        return float(np.mean([float(row[name]) for row in per_subject]))

    metrics = {
        "density_balanced_accuracy": mean("density_balanced_accuracy"),
        "diagonal_balanced_accuracy": mean("diagonal_balanced_accuracy"),
        "pooled_balanced_accuracy": mean("pooled_balanced_accuracy"),
        "offdiagonal_ablated_balanced_accuracy": mean(
            "offdiagonal_ablated_balanced_accuracy"
        ),
        "density_minus_diagonal": mean("density_minus_diagonal"),
        "density_minus_ablated": mean("density_minus_ablated"),
        "per_subject": per_subject,
    }

    _write_json(run_dir / "study_manifest.json", spec)
    _write_json(run_dir / "metrics.json", metrics)
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(_canonical_json(row) + "\n")
    _write_json(
        run_dir / "mechanism.json",
        {
            "claim_class": "quantum_inspired",
            "mechanism": "density_off_diagonal_structure",
            "density_minus_ablated_balanced_accuracy": metrics["density_minus_ablated"],
            "promotion_eligible": False,
            "reason": "synthetic smoke evidence cannot promote an empirical neural claim",
        },
    )
    report = (
        "# QuantumBCI density smoke report\n\n"
        f"- Run: `{run_id}`\n"
        f"- Scientific fingerprint: `{scientific_fingerprint}`\n"
        f"- Density balanced accuracy: {metrics['density_balanced_accuracy']:.3f}\n"
        f"- Diagonal control: {metrics['diagonal_balanced_accuracy']:.3f}\n"
        f"- Pooled control: {metrics['pooled_balanced_accuracy']:.3f}\n"
        f"- Off-diagonal ablation: {metrics['offdiagonal_ablated_balanced_accuracy']:.3f}\n"
        f"- Density minus ablated: {metrics['density_minus_ablated']:+.3f}\n\n"
        "Claim ceiling: `quantum_inspired`; evidence tier: `synthetic_sanity`.\n"
    )
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    (run_dir / "report.html").write_text(
        _render_html_report(run_id, scientific_fingerprint, metrics),
        encoding="utf-8",
    )

    completed_at = datetime.now(timezone.utc).isoformat()
    run_record = {
        "schema_version": 1,
        "run_id": run_id,
        "experiment_id": spec["experiment_id"],
        "status": "completed",
        "evidence_tier": "synthetic_sanity",
        "claim_class": "quantum_inspired",
        "scientific_fingerprint": scientific_fingerprint,
        "source_sha": config.source_sha,
        "seed": resolved_seed,
        "started_at": started_at,
        "completed_at": completed_at,
        "report": "report.html",
        "metrics": {key: value for key, value in metrics.items() if key != "per_subject"},
    }
    _write_json(run_dir / "run.json", run_record)
    _write_json(run_dir / "artifact_hashes.json", _artifact_hashes(run_dir))

    return SmokeResult(
        run_id=run_id,
        run_dir=run_dir,
        scientific_fingerprint=scientific_fingerprint,
        metrics=metrics,
    )


def doctor_report(config: WorkbenchConfig) -> dict[str, Any]:
    """Return a machine-readable readiness report."""

    root = RunStore(config.artifact_root).ensure()
    writable = os.access(root, os.W_OK)
    neuros = neuros_integration_status()
    manifest_count = len(find_manifest_files())
    return {
        "status": "ok" if writable and sys.version_info >= (3, 10) and manifest_count > 0 else "attention",
        "python": {
            "version": platform.python_version(),
            "supported": sys.version_info >= (3, 10),
        },
        "quantumbci": _distribution_version("quantum-bci"),
        "numpy": np.__version__,
        "artifact_root": str(root),
        "artifact_root_writable": bool(writable),
        "optional": {
            "qiskit": _distribution_version("qiskit"),
            "qiskit-aer": _distribution_version("qiskit-aer"),
            **neuros.to_mapping(),
        },
        "source_sha": config.source_sha,
        "experiment_manifests": manifest_count,
    }


def find_manifest_files(extra_dirs: Iterable[str | Path] = ()) -> list[Path]:
    """Discover packaged manifests plus source/explicit overrides.

    Packaged manifests make ``quantumbci experiments list`` useful after a normal
    wheel install. A source checkout or explicitly supplied manifest directory may
    override a packaged manifest with the same filename.
    """

    packaged_registry = Path(__file__).resolve().parent / "experiments" / "manifests"
    directories = [packaged_registry]
    source_registry = Path("experiments/manifests")
    if source_registry.exists():
        directories.append(source_registry)
    directories.extend(Path(directory) for directory in extra_dirs)

    found: dict[str, Path] = {}
    for directory in directories:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            found[path.name] = path
    return sorted(found.values(), key=lambda path: path.name)
