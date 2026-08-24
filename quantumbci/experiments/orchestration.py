"""Deterministic planning for QuantumBCI experiment DAGs."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .manifest import ExperimentManifest, ManifestError, StageSpec


def topological_layers(manifest: ExperimentManifest) -> tuple[tuple[StageSpec, ...], ...]:
    """Return deterministic parallelizable layers of the experiment DAG."""

    stage_by_id = {stage.id: stage for stage in manifest.stages}
    remaining = {stage.id: set(stage.depends_on) for stage in manifest.stages}
    resolved: set[str] = set()
    layers: list[tuple[StageSpec, ...]] = []
    while remaining:
        ready_ids = sorted(stage_id for stage_id, deps in remaining.items() if deps <= resolved)
        if not ready_ids:
            raise ManifestError("Experiment DAG could not be topologically ordered")
        layer = tuple(stage_by_id[stage_id] for stage_id in ready_ids)
        layers.append(layer)
        resolved.update(ready_ids)
        for stage_id in ready_ids:
            remaining.pop(stage_id)
    return tuple(layers)


def build_plan(manifest: ExperimentManifest, source_sha: str) -> dict[str, Any]:
    """Build a portable run *plan* for a validated manifest.

    This is intentionally a plan id, not the final scientific run id. The latter must
    additionally bind the dataset fingerprint and split registry after data-contract
    stages execute.
    """

    source_sha = source_sha.strip()
    if not source_sha:
        raise ValueError("source_sha must not be empty")
    identity = f"{manifest.digest}:{source_sha}"
    plan_id = sha256(identity.encode("utf-8")).hexdigest()
    layers = topological_layers(manifest)
    return {
        "schema_version": 1,
        "experiment_id": manifest.id,
        "manifest_digest": manifest.digest,
        "source_sha": source_sha,
        "plan_id": plan_id,
        "claim_class": manifest.claim_class.value,
        "layers": [[stage.id for stage in layer] for layer in layers],
        "stages": {
            stage.id: {
                "kind": stage.kind,
                "depends_on": list(stage.depends_on),
                "command": list(stage.command),
                "resources": dict(stage.resources),
                "produces": list(stage.produces),
                "status": "pending",
            }
            for stage in manifest.stages
        },
        "decision_gates": list(manifest.decision_gates),
        "final_run_id_note": (
            "Bind dataset_fingerprint and split_registry to this plan before assigning "
            "the final scientific run id."
        ),
    }


def render_plan(manifest: ExperimentManifest, source_sha: str) -> str:
    """Render the DAG as a compact human-readable execution plan."""

    plan = build_plan(manifest, source_sha)
    lines = [
        f"{manifest.id}: {manifest.title}",
        f"claim ceiling: {manifest.claim_class.value}",
        f"manifest: {manifest.digest[:12]}",
        f"source: {source_sha}",
        f"plan: {plan['plan_id'][:12]}",
    ]
    for index, layer in enumerate(topological_layers(manifest), start=1):
        stage_names = ", ".join(stage.id for stage in layer)
        lines.append(f"layer {index}: {stage_names}")
    return "\n".join(lines)


def materialize_plan(
    manifest: ExperimentManifest,
    source_sha: str,
    output_dir: str | Path,
) -> Path:
    """Write a stable JSON plan ledger without executing experiment commands."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "plan.json"
    payload = build_plan(manifest, source_sha)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
