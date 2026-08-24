"""Machine-readable experiment contracts.

The manifest layer deliberately does not execute commands. It validates the scientific
DAG and produces stable content hashes that can be consumed by local, cluster, cloud,
or QPU executors without making any one platform part of the evidence model.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from ..claims import ClaimClass


class ManifestError(ValueError):
    """Raised when an experiment manifest violates the orchestration contract."""


@dataclass(frozen=True)
class StageSpec:
    """One node in an experiment DAG."""

    id: str
    kind: str
    depends_on: tuple[str, ...]
    command: tuple[str, ...]
    resources: Mapping[str, Any]
    produces: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StageSpec":
        required = ("id", "kind", "depends_on", "command", "resources", "produces")
        missing = [key for key in required if key not in value]
        if missing:
            raise ManifestError(f"Stage is missing required fields: {', '.join(missing)}")
        stage = cls(
            id=str(value["id"]).strip(),
            kind=str(value["kind"]).strip(),
            depends_on=tuple(str(item) for item in value["depends_on"]),
            command=tuple(str(item) for item in value["command"]),
            resources=dict(value["resources"]),
            produces=tuple(str(item) for item in value["produces"]),
        )
        if not stage.id or not stage.kind:
            raise ManifestError("Stage id and kind must be non-empty")
        if not stage.command:
            raise ManifestError(f"Stage {stage.id!r} must declare a tokenized command")
        if len(set(stage.depends_on)) != len(stage.depends_on):
            raise ManifestError(f"Stage {stage.id!r} repeats a dependency")
        if len(set(stage.produces)) != len(stage.produces):
            raise ManifestError(f"Stage {stage.id!r} repeats an output artifact")
        return stage

    def to_mapping(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "depends_on": list(self.depends_on),
            "command": list(self.command),
            "resources": dict(self.resources),
            "produces": list(self.produces),
        }


@dataclass(frozen=True)
class ExperimentManifest:
    """Validated scientific and computational contract for an experiment."""

    schema_version: int
    id: str
    title: str
    claim_class: ClaimClass
    objective: str
    datasets: tuple[str, ...]
    encoders: tuple[str, ...]
    primary_metrics: tuple[str, ...]
    stages: tuple[StageSpec, ...]
    decision_gates: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExperimentManifest":
        required = (
            "schema_version",
            "id",
            "title",
            "claim_class",
            "objective",
            "datasets",
            "encoders",
            "primary_metrics",
            "stages",
            "decision_gates",
        )
        missing = [key for key in required if key not in value]
        if missing:
            raise ManifestError(f"Manifest is missing required fields: {', '.join(missing)}")
        try:
            claim_class = ClaimClass(str(value["claim_class"]))
        except ValueError as exc:
            options = ", ".join(item.value for item in ClaimClass)
            raise ManifestError(f"Unknown claim_class. Expected one of: {options}") from exc

        manifest = cls(
            schema_version=int(value["schema_version"]),
            id=str(value["id"]).strip(),
            title=str(value["title"]).strip(),
            claim_class=claim_class,
            objective=str(value["objective"]).strip(),
            datasets=tuple(str(item) for item in value["datasets"]),
            encoders=tuple(str(item) for item in value["encoders"]),
            primary_metrics=tuple(str(item) for item in value["primary_metrics"]),
            stages=tuple(StageSpec.from_mapping(item) for item in value["stages"]),
            decision_gates=tuple(str(item).strip() for item in value["decision_gates"]),
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ManifestError(f"Unsupported manifest schema_version={self.schema_version}")
        for name, value in (
            ("id", self.id),
            ("title", self.title),
            ("objective", self.objective),
            ("datasets", self.datasets),
            ("encoders", self.encoders),
            ("primary_metrics", self.primary_metrics),
            ("stages", self.stages),
            ("decision_gates", self.decision_gates),
        ):
            if not value:
                raise ManifestError(f"Manifest {name} must not be empty")

        stage_ids = [stage.id for stage in self.stages]
        if len(set(stage_ids)) != len(stage_ids):
            raise ManifestError("Stage ids must be unique")
        known = set(stage_ids)
        for stage in self.stages:
            if stage.id in stage.depends_on:
                raise ManifestError(f"Stage {stage.id!r} cannot depend on itself")
            unknown = sorted(set(stage.depends_on) - known)
            if unknown:
                raise ManifestError(
                    f"Stage {stage.id!r} references unknown dependencies: {', '.join(unknown)}"
                )

        artifact_owner: dict[str, str] = {}
        for stage in self.stages:
            for artifact in stage.produces:
                if not artifact:
                    raise ManifestError(f"Stage {stage.id!r} declares an empty artifact name")
                previous = artifact_owner.get(artifact)
                if previous is not None:
                    raise ManifestError(
                        f"Artifact {artifact!r} is produced by both {previous!r} and {stage.id!r}"
                    )
                artifact_owner[artifact] = stage.id

        # Kahn validation is repeated in orchestration to keep this object independently safe.
        remaining = {stage.id: set(stage.depends_on) for stage in self.stages}
        resolved: set[str] = set()
        while remaining:
            ready = sorted(stage_id for stage_id, deps in remaining.items() if deps <= resolved)
            if not ready:
                cycle = ", ".join(sorted(remaining))
                raise ManifestError(f"Experiment DAG contains a cycle involving: {cycle}")
            resolved.update(ready)
            for stage_id in ready:
                remaining.pop(stage_id)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "title": self.title,
            "claim_class": self.claim_class.value,
            "objective": self.objective,
            "datasets": list(self.datasets),
            "encoders": list(self.encoders),
            "primary_metrics": list(self.primary_metrics),
            "stages": [stage.to_mapping() for stage in self.stages],
            "decision_gates": list(self.decision_gates),
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _resolve_manifest_path(reference: str | Path) -> Path:
    """Resolve an explicit path or a bundled experiment ID/filename.

    This lets an installed wheel use ``E001_density_geometry`` directly instead
    of requiring knowledge of internal package paths. Explicit existing paths
    always win.
    """

    direct = Path(reference)
    if direct.exists():
        return direct

    text = str(reference).strip()
    if not text:
        raise FileNotFoundError("manifest reference must not be empty")
    names = [text]
    if not text.endswith(".json"):
        names.append(f"{text}.json")

    registries = (
        Path(__file__).resolve().parent / "manifests",
        Path("experiments/manifests"),
    )
    for registry in registries:
        for name in names:
            candidate = registry / name
            if candidate.exists():
                return candidate

    # Final lookup by the declared experiment id, useful if a future filename
    # differs from its manifest id.
    for registry in registries:
        if not registry.exists():
            continue
        for candidate in sorted(registry.glob("*.json")):
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and str(payload.get("id", "")) == text:
                return candidate

    raise FileNotFoundError(
        f"Unknown experiment manifest {text!r}. Use 'quantumbci experiments list' "
        "to inspect bundled experiment IDs."
    )


def load_manifest(path: str | Path) -> ExperimentManifest:
    """Load and validate one JSON experiment manifest or bundled experiment ID."""

    manifest_path = _resolve_manifest_path(path)
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Invalid JSON in {manifest_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError("Manifest root must be a JSON object")
    return ExperimentManifest.from_mapping(value)
