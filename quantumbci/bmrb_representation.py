"""Executable cross-representation BMRB bundles."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from typing import Any, Mapping

from .exporting import verify_run_artifacts
from .representation_conservation import (
    RepresentationConservationPolicy,
    RepresentationConservationResult,
    RepresentationEffectCase,
    build_representation_necessity_profile,
    evaluate_representation_conservation,
)
from .representation_studies import E001_REPRESENTATION_LANE_SCHEMA
from .recapitulation import MechanismNecessityProfile

BMRB_REPRESENTATION_SCHEMA = 1
BMRB_REPRESENTATION_BENCHMARK = "BMRB_REPRESENTATION_V1"


def _required_text(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _strict_bool(name: str, value: Any) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a JSON boolean")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _metric_balanced_accuracy(benchmark: Mapping[str, Any], method: str) -> float:
    metrics = benchmark.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("E001 benchmark is missing metrics mapping")
    metric = metrics.get(method)
    if not isinstance(metric, Mapping):
        raise ValueError(f"E001 benchmark is missing method {method!r}")
    value = float(metric.get("balanced_accuracy"))
    if not (-1e100 < value < 1e100):
        raise ValueError(f"balanced_accuracy for {method!r} must be finite")
    return value


@dataclass(frozen=True)
class BMRBRepresentationLaneSpec:
    lane_id: str
    artifact_dir: Path
    artifact_ledger_sha256: str
    scientific_fingerprint: str
    source_representation_id: str
    representation_family: str
    model_id: str | None
    model_revision: str | None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "artifact_dir": self.artifact_dir.name,
            "artifact_ledger_sha256": self.artifact_ledger_sha256,
            "scientific_fingerprint": self.scientific_fingerprint,
            "source_representation_id": self.source_representation_id,
            "representation_family": self.representation_family,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
        }


@dataclass(frozen=True)
class BMRBRepresentationBundle:
    study_id: str
    mechanism_id: str
    policy: RepresentationConservationPolicy
    lanes: tuple[BMRBRepresentationLaneSpec, ...]
    conservation: RepresentationConservationResult
    profile: MechanismNecessityProfile
    source_fingerprint: str

    @property
    def artifact_fingerprint(self) -> str:
        return sha256(
            b"quantumbci.bmrb-representation-artifact.v1\0"
            + _canonical_json(self._mapping_without_artifact_fingerprint()).encode("utf-8")
        ).hexdigest()

    def _mapping_without_artifact_fingerprint(self) -> dict[str, Any]:
        return {
            "schema_version": BMRB_REPRESENTATION_SCHEMA,
            "artifact_role": "bmrb_representation_bundle",
            "benchmark": BMRB_REPRESENTATION_BENCHMARK,
            "study_id": self.study_id,
            "mechanism_id": self.mechanism_id,
            "policy": self.policy.to_mapping(),
            "lanes": [lane.to_mapping() for lane in self.lanes],
            "representation_conservation": self.conservation.to_mapping(),
            "mechanism_profile": self.profile.to_mapping(),
            "source_fingerprint": self.source_fingerprint,
            "physical_quantum_promotion_eligible": False,
        }

    def to_mapping(self) -> dict[str, Any]:
        return {
            **self._mapping_without_artifact_fingerprint(),
            "artifact_fingerprint": self.artifact_fingerprint,
        }


def _resolve_lane_dir(root: Path, raw: Any, *, label: str) -> Path:
    value = Path(_required_text(label, raw)).expanduser()
    if not value.is_absolute():
        value = root / value
    value = value.resolve()
    if not value.is_dir():
        raise FileNotFoundError(f"{label} not found: {value}")
    return value


def _load_lane(
    *,
    lane_id: str,
    artifact_dir: Path,
    mechanism_id: str,
    participant_key: str,
) -> tuple[BMRBRepresentationLaneSpec, tuple[RepresentationEffectCase, ...]]:
    verification = verify_run_artifacts(artifact_dir)
    if not verification["valid"]:
        raise ValueError(
            f"representation lane {lane_id!r} failed artifact verification: {verification}"
        )
    run = _load_json(artifact_dir / "run.json", label="representation lane run")
    manifest = _load_json(
        artifact_dir / "study_manifest.json", label="representation lane manifest"
    )
    case_payload = _load_json(
        artifact_dir / "case_results.json", label="representation lane cases"
    )
    if manifest.get("artifact_role") != E001_REPRESENTATION_LANE_SCHEMA:
        raise ValueError(
            f"representation lane {lane_id!r} has unsupported artifact_role: "
            f"{manifest.get('artifact_role')!r}"
        )
    scientific_fingerprint = _required_text(
        "lane scientific_fingerprint", manifest.get("scientific_fingerprint")
    )
    if run.get("scientific_fingerprint") != scientific_fingerprint:
        raise ValueError("representation lane run/manifest scientific fingerprint mismatch")
    if case_payload.get("scientific_fingerprint") != scientific_fingerprint:
        raise ValueError("representation lane cases/manifest scientific fingerprint mismatch")
    if case_payload.get("artifact_role") != "e001_representation_lane_cases":
        raise ValueError("representation lane case_results has the wrong artifact_role")

    source_representation_id = _required_text(
        "lane representation_id", manifest.get("representation_id")
    )
    representation_family = _required_text(
        "lane representation_family", manifest.get("representation_family")
    )
    model_id = manifest.get("model_id")
    model_revision = manifest.get("model_revision")
    if model_id is not None:
        model_id = _required_text("lane model_id", model_id)
        model_revision = _required_text("lane model_revision", model_revision)
    elif model_revision is not None:
        raise ValueError("representation lane model_revision cannot be set without model_id")
    if representation_family == "foundation_model" and model_id is None:
        raise ValueError("foundation_model representation lanes must pin model_id and model_revision")

    raw_cases = case_payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("representation lane must contain case results")
    parsed: list[RepresentationEffectCase] = []
    for case_index, case in enumerate(raw_cases):
        if not isinstance(case, Mapping):
            raise ValueError(f"lane case[{case_index}] must be an object")
        if int(case.get("schema_version", 0)) != 2:
            raise ValueError("BMRB-Representation v1 requires LongitudinalE001CaseResult schema 2")
        if case.get("representation_id") != source_representation_id:
            raise ValueError("case representation_id does not match lane manifest")
        representation_sha = _required_text(
            "case representation_sha256", case.get("representation_sha256")
        )
        case_fingerprint = _required_text("case study_fingerprint", case.get("study_fingerprint"))
        authority = case.get("authority")
        if not isinstance(authority, Mapping):
            raise ValueError("representation case is missing authority mapping")
        case_id = _required_text("authority.case_id", authority.get("case_id"))
        authority_fingerprint = _required_text(
            "authority.authority_fingerprint", authority.get("authority_fingerprint")
        )
        case_metadata = authority.get("case_metadata")
        if not isinstance(case_metadata, Mapping):
            raise ValueError("representation case authority lacks case_metadata")
        participant = _required_text(
            f"authority.case_metadata[{participant_key!r}]", case_metadata.get(participant_key)
        )
        held_out_values = authority.get("held_out_values")
        if not isinstance(held_out_values, list) or not held_out_values:
            raise ValueError("representation case authority lacks held_out_values")
        occasion = str(case_metadata.get("held_out_session", held_out_values[0])).strip()
        if not occasion:
            raise ValueError("representation case occasion/session id must not be empty")

        rows = case.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError("representation case must contain E001 rows")
        seen_budgets: set[int] = set()
        for row_index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ValueError(f"representation row[{row_index}] must be an object")
            if row.get("case_id") != case_id:
                raise ValueError("representation row case_id does not match case authority")
            if row.get("authority_fingerprint") != authority_fingerprint:
                raise ValueError("representation row authority_fingerprint mismatch")
            if row.get("representation_id") != source_representation_id:
                raise ValueError("representation row representation_id mismatch")
            if row.get("representation_sha256") != representation_sha:
                raise ValueError("representation row representation_sha256 mismatch")
            budget = int(row.get("calibration_per_class", -1))
            if budget < 0:
                raise ValueError("calibration_per_class must be non-negative")
            if budget in seen_budgets:
                raise ValueError(
                    f"duplicate calibration budget {budget} in representation case {case_id!r}"
                )
            seen_budgets.add(budget)
            benchmark = row.get("benchmark")
            if not isinstance(benchmark, Mapping):
                raise ValueError("representation row is missing benchmark mapping")
            strongest = _required_text(
                "benchmark.strongest_classical_control",
                benchmark.get("strongest_classical_control"),
            )
            information_novel = _strict_bool(
                "benchmark.density_information_novel",
                benchmark.get("density_information_novel"),
            )
            parsed.append(
                RepresentationEffectCase(
                    participant_id=participant,
                    occasion_id=occasion,
                    case_id=case_id,
                    calibration_per_class=budget,
                    representation_id=lane_id,
                    representation_family=representation_family,
                    source_representation_id=source_representation_id,
                    model_id=model_id,
                    model_revision=model_revision,
                    mechanism_id=mechanism_id,
                    authority_fingerprint=authority_fingerprint,
                    representation_sha256=representation_sha,
                    source_fingerprint=case_fingerprint,
                    candidate_metric=_metric_balanced_accuracy(benchmark, "density"),
                    strongest_control_metric=_metric_balanced_accuracy(benchmark, strongest),
                    ablated_metric=_metric_balanced_accuracy(benchmark, "offdiagonal_ablation"),
                    higher_is_better=True,
                    information_novel=information_novel,
                )
            )

    lane_spec = BMRBRepresentationLaneSpec(
        lane_id=lane_id,
        artifact_dir=artifact_dir,
        artifact_ledger_sha256=_sha256_file(artifact_dir / "artifact_hashes.json"),
        scientific_fingerprint=scientific_fingerprint,
        source_representation_id=source_representation_id,
        representation_family=representation_family,
        model_id=model_id,
        model_revision=model_revision,
    )
    return lane_spec, tuple(parsed)


def load_bmrb_representation_manifest(
    path: str | Path,
) -> tuple[
    str,
    str,
    RepresentationConservationPolicy,
    tuple[BMRBRepresentationLaneSpec, ...],
    tuple[RepresentationEffectCase, ...],
    dict[str, Any],
]:
    manifest_path = Path(path).expanduser().resolve()
    payload = _load_json(manifest_path, label="BMRB representation manifest")
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("BMRB representation manifest schema_version must be 1")
    study_id = _required_text("study_id", payload.get("study_id"))
    mechanism_id = _required_text("mechanism_id", payload.get("mechanism_id"))
    participant_key = _required_text("participant_key", payload.get("participant_key", "subject"))
    policy_payload = payload.get("policy")
    if not isinstance(policy_payload, Mapping):
        raise ValueError("BMRB representation manifest is missing policy")
    policy = RepresentationConservationPolicy.from_mapping(policy_payload)
    raw_lanes = payload.get("lanes")
    if not isinstance(raw_lanes, list) or len(raw_lanes) < 2:
        raise ValueError("BMRB representation manifest requires at least two lanes")

    specs: list[BMRBRepresentationLaneSpec] = []
    cases: list[RepresentationEffectCase] = []
    lane_ids: set[str] = set()
    for index, raw in enumerate(raw_lanes):
        if not isinstance(raw, Mapping):
            raise ValueError(f"lanes[{index}] must be an object")
        lane_id = _required_text(f"lanes[{index}].lane_id", raw.get("lane_id"))
        if lane_id in lane_ids:
            raise ValueError(f"duplicate representation lane_id: {lane_id!r}")
        lane_dir = _resolve_lane_dir(
            manifest_path.parent,
            raw.get("artifact_dir"),
            label=f"lanes[{index}].artifact_dir",
        )
        spec, lane_cases = _load_lane(
            lane_id=lane_id,
            artifact_dir=lane_dir,
            mechanism_id=mechanism_id,
            participant_key=participant_key,
        )
        expected_fingerprint = raw.get("scientific_fingerprint")
        if expected_fingerprint is not None and str(expected_fingerprint) != spec.scientific_fingerprint:
            raise ValueError(
                f"representation lane {lane_id!r} scientific_fingerprint does not match manifest"
            )
        specs.append(spec)
        cases.extend(lane_cases)
        lane_ids.add(lane_id)
    if policy.reference_representation_id not in lane_ids:
        raise ValueError("policy reference_representation_id is not a declared lane")
    return study_id, mechanism_id, policy, tuple(specs), tuple(cases), dict(payload)


def build_bmrb_representation_bundle(manifest_path: str | Path) -> BMRBRepresentationBundle:
    study_id, mechanism_id, policy, lanes, cases, manifest = load_bmrb_representation_manifest(
        manifest_path
    )
    conservation = evaluate_representation_conservation(cases, policy=policy)
    profile = build_representation_necessity_profile(conservation)
    identity = {
        "schema_version": BMRB_REPRESENTATION_SCHEMA,
        "study_id": study_id,
        "mechanism_id": mechanism_id,
        "policy": policy.to_mapping(),
        "lanes": [lane.to_mapping() for lane in lanes],
        "conservation_source_fingerprint": conservation.source_fingerprint,
        "manifest_metadata": manifest.get("metadata", {}),
    }
    source_fingerprint = sha256(
        b"quantumbci.bmrb-representation.v1\0"
        + _canonical_json(identity).encode("utf-8")
    ).hexdigest()
    return BMRBRepresentationBundle(
        study_id=study_id,
        mechanism_id=mechanism_id,
        policy=policy,
        lanes=lanes,
        conservation=conservation,
        profile=profile,
        source_fingerprint=source_fingerprint,
    )


def verify_bmrb_representation_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    if int(value.get("schema_version", 0)) != BMRB_REPRESENTATION_SCHEMA:
        raise ValueError("unsupported BMRB representation schema")
    if value.get("artifact_role") != "bmrb_representation_bundle":
        raise ValueError("BMRB representation artifact has the wrong artifact_role")
    if value.get("benchmark") != BMRB_REPRESENTATION_BENCHMARK:
        raise ValueError("BMRB representation artifact has the wrong benchmark id")
    fingerprint = _required_text("artifact_fingerprint", value.get("artifact_fingerprint"))
    scientific = {key: item for key, item in value.items() if key != "artifact_fingerprint"}
    expected = sha256(
        b"quantumbci.bmrb-representation-artifact.v1\0"
        + _canonical_json(scientific).encode("utf-8")
    ).hexdigest()
    if expected != fingerprint:
        raise ValueError("BMRB representation artifact fingerprint mismatch")
    return value


def render_bmrb_representation_html(bundle: BMRBRepresentationBundle) -> str:
    result = bundle.conservation
    profile = bundle.profile
    promotion = "none" if profile.promotion_ceiling is None else profile.promotion_ceiling.name.lower()
    gate_rows = "".join(
        "<tr>"
        f"<td>{escape(gate.tier.name.lower())}</td>"
        f"<td>{escape(gate.id)}</td>"
        f"<td><strong>{escape(gate.status.value)}</strong></td>"
        f"<td>{escape(gate.summary)}</td>"
        "</tr>"
        for gate in profile.ordered_gates
    )
    lane_rows = "".join(
        "<tr>"
        f"<td>{escape(lane.representation_id)}</td>"
        f"<td>{escape(lane.representation_family)}</td>"
        f"<td>{escape(lane.model_id or 'raw / none')}</td>"
        f"<td>{lane.mean_candidate_advantage:+.4f}</td>"
        f"<td>{lane.mean_ablation_necessity:+.4f}</td>"
        f"<td>{lane.participant_positive_fraction:.3f}</td>"
        f"<td>{lane.information_novel_fraction:.3f}</td>"
        "</tr>"
        for lane in result.lanes
    )
    correlation_rows = "".join(
        "<tr>"
        f"<td>{escape(lane_id)}</td>"
        f"<td>{'n/a' if value is None else f'{value:.3f}'}</td>"
        "</tr>"
        for lane_id, value in sorted(result.pairwise_reference_correlations.items())
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(bundle.study_id)} · BMRB-Representation</title>
<style>
body {{ font-family: ui-sans-serif,system-ui,sans-serif; max-width:1180px; margin:40px auto; padding:0 24px; line-height:1.45; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin:20px 0 28px; }}
.card {{ border:1px solid #bbb; border-radius:10px; padding:14px; }} .card b {{ display:block;font-size:1.3rem;margin-top:4px; }}
table {{ width:100%;border-collapse:collapse;margin:14px 0 30px; }} th,td {{ border-bottom:1px solid #ccc;padding:9px 8px;text-align:left;vertical-align:top; }}
.small {{ opacity:.75;font-size:.92rem; }} code {{ font-family:ui-monospace,monospace; }}
</style></head><body>
<h1>BMRB-Representation · {escape(bundle.study_id)}</h1>
<p>The same authority-bound cases are compared across frozen representation spaces. Conservation and information novelty are reported separately.</p>
<div class="cards">
<div class="card">Representations<b>{result.representation_count}</b></div>
<div class="card">Families<b>{result.representation_family_count}</b></div>
<div class="card">Participants<b>{result.participant_count}</b></div>
<div class="card">Direction match<b>{result.direction_match_fraction:.3f}</b></div>
<div class="card">Novel lane fraction<b>{result.information_novel_representation_fraction:.3f}</b></div>
<div class="card">Promotion ceiling<b>{escape(promotion)}</b></div>
</div>
<h2>Mechanism necessity ladder</h2><table><thead><tr><th>Tier</th><th>Gate</th><th>Status</th><th>Interpretation</th></tr></thead><tbody>{gate_rows}</tbody></table>
<h2>Representation lanes</h2><table><thead><tr><th>Lane</th><th>Family</th><th>Model</th><th>Candidate advantage</th><th>Ablation necessity</th><th>Participant positive</th><th>Information novel</th></tr></thead><tbody>{lane_rows}</tbody></table>
<h2>Reference effect correlations</h2><table><thead><tr><th>Lane</th><th>Pearson r</th></tr></thead><tbody>{correlation_rows}</tbody></table>
<p class="small">Correlation is descriptive only and is unavailable for fewer than three participants or degenerate participant effects. Promotion is controlled by the preregistered sign, ablation, participant, family and adversary gates instead.</p>
<p class="small"><code>source_fingerprint={escape(bundle.source_fingerprint)}</code></p>
<p class="small"><code>artifact_fingerprint={escape(bundle.artifact_fingerprint)}</code></p>
</body></html>"""


def write_bmrb_representation_bundle(
    bundle: BMRBRepresentationBundle,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "bmrb_representation.json"
    html_path = root / "report.html"
    json_path.write_text(
        json.dumps(bundle.to_mapping(), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    verify_bmrb_representation_mapping(_load_json(json_path, label="BMRB representation output"))
    html_path.write_text(render_bmrb_representation_html(bundle), encoding="utf-8")
    return json_path, html_path
