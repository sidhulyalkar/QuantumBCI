"""Portable recipe contracts for running QuantumBCI on external frozen embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import mimetypes
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .benchmarking import IndexSplit, benchmark_density_embeddings
from .claims import ClaimClass
from .workbench import RunStore, WorkbenchConfig


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_hashes(run_dir: Path) -> dict[str, str]:
    return {
        path.name: _hash_file(path)
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }


@dataclass(frozen=True)
class FrozenEmbeddingRecipe:
    """A reproducible, path-resolved benchmark recipe.

    The recipe intentionally supports only the ``quantum_inspired`` claim class.
    Higher-level physical or hardware claims require different evidence contracts.
    """

    id: str
    title: str
    embeddings: Path
    labels: Path
    train_indices: Path
    test_indices: Path
    split_name: str = "explicit"
    ridge: float = 1e-3
    evidence_tier: str = "exploratory"
    source_dataset: str | None = None
    source_model: str | None = None
    notes: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, base_dir: Path) -> "FrozenEmbeddingRecipe":
        if int(value.get("schema_version", 1)) != 1:
            raise ValueError("Unsupported recipe schema_version")
        claim_class = str(value.get("claim_class", ClaimClass.QUANTUM_INSPIRED.value))
        if claim_class != ClaimClass.QUANTUM_INSPIRED.value:
            raise ValueError("frozen-embedding recipes are limited to claim_class=quantum_inspired")
        data = value.get("data")
        if not isinstance(data, Mapping):
            raise ValueError("recipe data must be an object")
        benchmark = value.get("benchmark", {})
        if not isinstance(benchmark, Mapping):
            raise ValueError("recipe benchmark must be an object")

        def resolve(key: str) -> Path:
            raw = str(data.get(key, "")).strip()
            if not raw:
                raise ValueError(f"recipe data.{key} must not be empty")
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = (base_dir / path).resolve()
            return path

        recipe = cls(
            id=str(value.get("id", "")).strip(),
            title=str(value.get("title", "")).strip(),
            embeddings=resolve("embeddings"),
            labels=resolve("labels"),
            train_indices=resolve("train_indices"),
            test_indices=resolve("test_indices"),
            split_name=str(data.get("split_name", "explicit")).strip() or "explicit",
            ridge=float(benchmark.get("ridge", 1e-3)),
            evidence_tier=str(value.get("evidence_tier", "exploratory")).strip() or "exploratory",
            source_dataset=_optional_text(value.get("source_dataset")),
            source_model=_optional_text(value.get("source_model")),
            notes=_optional_text(value.get("notes")),
        )
        recipe.validate()
        return recipe

    def validate(self) -> None:
        if not self.id:
            raise ValueError("recipe id must not be empty")
        if not self.title:
            raise ValueError("recipe title must not be empty")
        if self.ridge < 0:
            raise ValueError("recipe ridge must be non-negative")
        for path in (self.embeddings, self.labels, self.train_indices, self.test_indices):
            if not path.is_file():
                raise FileNotFoundError(f"recipe input does not exist: {path}")

    def input_fingerprints(self) -> dict[str, dict[str, Any]]:
        values: dict[str, dict[str, Any]] = {}
        for name, path in (
            ("embeddings", self.embeddings),
            ("labels", self.labels),
            ("train_indices", self.train_indices),
            ("test_indices", self.test_indices),
        ):
            values[name] = {
                "filename": path.name,
                "sha256": _hash_file(path),
                "bytes": path.stat().st_size,
                "encoding_format": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            }
        return values

    def identity_mapping(self, *, source_sha: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "id": self.id,
            "claim_class": ClaimClass.QUANTUM_INSPIRED.value,
            "evidence_tier": self.evidence_tier,
            "split_name": self.split_name,
            "ridge": self.ridge,
            "source_dataset": self.source_dataset,
            "source_model": self.source_model,
            "input_fingerprints": self.input_fingerprints(),
            "source_sha": source_sha,
        }

    def to_portable_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "id": self.id,
            "title": self.title,
            "claim_class": ClaimClass.QUANTUM_INSPIRED.value,
            "evidence_tier": self.evidence_tier,
            "source_dataset": self.source_dataset,
            "source_model": self.source_model,
            "notes": self.notes,
            "data": {
                "embeddings": self.embeddings.name,
                "labels": self.labels.name,
                "train_indices": self.train_indices.name,
                "test_indices": self.test_indices.name,
                "split_name": self.split_name,
            },
            "benchmark": {"ridge": self.ridge},
        }


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_recipe(path: str | Path) -> FrozenEmbeddingRecipe:
    recipe_path = Path(path)
    payload = json.loads(recipe_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recipe root must be a JSON object")
    return FrozenEmbeddingRecipe.from_mapping(payload, base_dir=recipe_path.parent)


def write_recipe_template(path: str | Path, *, force: bool = False) -> Path:
    recipe_path = Path(path)
    if recipe_path.exists() and not force:
        raise FileExistsError(f"{recipe_path} already exists; pass --force to replace it")
    payload = {
        "schema_version": 1,
        "id": "my-density-study",
        "title": "Density geometry on frozen neural embeddings",
        "claim_class": "quantum_inspired",
        "evidence_tier": "exploratory",
        "source_dataset": "replace-with-dataset-id-or-URL",
        "source_model": "replace-with-model-id-and-revision",
        "data": {
            "embeddings": "embeddings.npy",
            "labels": "labels.npy",
            "train_indices": "train_indices.npy",
            "test_indices": "test_indices.npy",
            "split_name": "subject-exclusive-v1",
        },
        "benchmark": {"ridge": 0.001},
        "notes": "Document preprocessing and evidence authority here.",
    }
    _write_json(recipe_path, payload)
    return recipe_path


@dataclass(frozen=True)
class RecipeRunResult:
    run_id: str
    run_dir: Path
    scientific_fingerprint: str
    metrics: Mapping[str, Any]


def run_recipe(path: str | Path, config: WorkbenchConfig) -> RecipeRunResult:
    """Execute one frozen-embedding recipe into the normal QuantumBCI RunStore."""

    recipe = load_recipe(path)
    identity = recipe.identity_mapping(source_sha=config.source_sha)
    scientific_fingerprint = sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    run_id, run_dir = RunStore(config.artifact_root).create(recipe.id, scientific_fingerprint)
    started_at = datetime.now(timezone.utc).isoformat()

    embeddings = np.load(recipe.embeddings, allow_pickle=False)
    labels = np.load(recipe.labels, allow_pickle=False)
    train_indices = np.load(recipe.train_indices, allow_pickle=False)
    test_indices = np.load(recipe.test_indices, allow_pickle=False)
    split = IndexSplit(train_indices=train_indices, test_indices=test_indices, name=recipe.split_name)
    result = benchmark_density_embeddings(embeddings, labels, split, ridge=recipe.ridge)
    metrics = result.to_mapping(include_predictions=False)

    _write_json(run_dir / "recipe.json", recipe.to_portable_mapping())
    _write_json(run_dir / "inputs.json", {
        **identity,
        "array_shapes": {
            "embeddings": list(np.asarray(embeddings).shape),
            "labels": list(np.asarray(labels).shape),
            "train_indices": list(np.asarray(train_indices).shape),
            "test_indices": list(np.asarray(test_indices).shape),
        },
    })
    _write_json(run_dir / "metrics.json", metrics)

    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for offset, dataset_index in enumerate(split.test_indices.tolist()):
            row = {
                "dataset_index": int(dataset_index),
                "label": str(result.test_labels[offset]),
                **{
                    f"{name}_prediction": str(np.asarray(values)[offset])
                    for name, values in result.predictions.items()
                },
            }
            handle.write(_canonical_json(row) + "\n")

    report = _render_recipe_report(recipe, scientific_fingerprint, metrics)
    (run_dir / "report.md").write_text(report["markdown"], encoding="utf-8")
    (run_dir / "report.html").write_text(report["html"], encoding="utf-8")

    completed_at = datetime.now(timezone.utc).isoformat()
    _write_json(run_dir / "run.json", {
        "schema_version": 1,
        "run_id": run_id,
        "experiment_id": recipe.id,
        "title": recipe.title,
        "status": "completed",
        "claim_class": ClaimClass.QUANTUM_INSPIRED.value,
        "evidence_tier": recipe.evidence_tier,
        "scientific_fingerprint": scientific_fingerprint,
        "source_sha": config.source_sha,
        "source_dataset": recipe.source_dataset,
        "source_model": recipe.source_model,
        "started_at": started_at,
        "completed_at": completed_at,
        "report": "report.html",
        "metrics": {
            "density_balanced_accuracy": metrics["density"]["balanced_accuracy"],
            "diagonal_balanced_accuracy": metrics["diagonal_control"]["balanced_accuracy"],
            "pooled_balanced_accuracy": metrics["pooled_control"]["balanced_accuracy"],
            "offdiagonal_ablated_balanced_accuracy": metrics["offdiagonal_ablation"]["balanced_accuracy"],
            "density_minus_diagonal": metrics["density_minus_diagonal"],
            "density_minus_ablation": metrics["density_minus_ablation"],
        },
    })
    _write_json(run_dir / "artifact_hashes.json", _artifact_hashes(run_dir))
    return RecipeRunResult(
        run_id=run_id,
        run_dir=run_dir,
        scientific_fingerprint=scientific_fingerprint,
        metrics=metrics,
    )


def _render_recipe_report(
    recipe: FrozenEmbeddingRecipe,
    fingerprint: str,
    metrics: Mapping[str, Any],
) -> dict[str, str]:
    density = float(metrics["density"]["balanced_accuracy"])
    diagonal = float(metrics["diagonal_control"]["balanced_accuracy"])
    pooled = float(metrics["pooled_control"]["balanced_accuracy"])
    ablated = float(metrics["offdiagonal_ablation"]["balanced_accuracy"])
    delta = float(metrics["density_minus_ablation"])
    markdown = (
        f"# {recipe.title}\n\n"
        f"- Study id: `{recipe.id}`\n"
        f"- Scientific fingerprint: `{fingerprint}`\n"
        f"- Evidence tier: `{recipe.evidence_tier}`\n"
        f"- Source dataset: `{recipe.source_dataset or 'unspecified'}`\n"
        f"- Source model: `{recipe.source_model or 'unspecified'}`\n"
        f"- Density balanced accuracy: {density:.3f}\n"
        f"- Diagonal control: {diagonal:.3f}\n"
        f"- Pooled control: {pooled:.3f}\n"
        f"- Off-diagonal intervention: {ablated:.3f}\n"
        f"- Density minus intervention: {delta:+.3f}\n\n"
        "Claim ceiling: `quantum_inspired`. Recipe metadata and an explicit split do not, by themselves, establish causal or physical-quantum evidence.\n"
    )
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{recipe.title}</title><style>body{{font-family:system-ui,sans-serif;max-width:960px;margin:40px auto;padding:0 20px;line-height:1.5}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}.card{{border:1px solid #ddd;border-radius:12px;padding:16px}}.value{{font-size:2rem;font-weight:700}}code{{overflow-wrap:anywhere}}.note{{padding:14px;background:#f5f5f5;border-radius:10px}}</style></head><body><h1>{recipe.title}</h1><p><code>{fingerprint}</code></p><div class="grid"><div class="card"><div>Density BA</div><div class="value">{density:.3f}</div></div><div class="card"><div>Diagonal</div><div class="value">{diagonal:.3f}</div></div><div class="card"><div>Pooled</div><div class="value">{pooled:.3f}</div></div><div class="card"><div>Ablated</div><div class="value">{ablated:.3f}</div></div><div class="card"><div>Mechanism Δ</div><div class="value">{delta:+.3f}</div></div></div><h2>Provenance</h2><p>Dataset: {recipe.source_dataset or 'unspecified'}<br>Model: {recipe.source_model or 'unspecified'}<br>Split: {recipe.split_name}<br>Evidence tier: {recipe.evidence_tier}</p><p class="note"><strong>Interpretation ceiling:</strong> quantum-inspired. A good fit or ablation effect is not evidence that neural tissue is physically quantum.</p></body></html>"""
    return {"markdown": markdown, "html": html}
