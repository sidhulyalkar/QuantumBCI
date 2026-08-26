"""End-to-end causal promotion bundles for BMRB-Dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from typing import Any, Mapping

from .causal_recapitulation import (
    CausalCaseEvidence,
    CausalNecessityPolicy,
    CausalNecessityResult,
    attach_causal_evidence,
    causal_case_from_neuros_mechint,
    evaluate_causal_necessity,
)
from .matched_recovery import (
    MatchedClassicalRecoveryEvidence,
    matched_classical_recovery_from_mapping,
)
from .recapitulation import MechanismNecessityProfile, mechanism_profile_from_mapping


def _required_text(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class BMRBCausalCaseSpec:
    participant_id: str
    occasion_id: str
    case_id: str
    dose_response_path: Path
    dose_response_sha256: str
    faithfulness_path: Path
    faithfulness_sha256: str
    matched_recovery_path: Path
    matched_recovery_sha256: str
    matched_recovery: MatchedClassicalRecoveryEvidence

    def to_mapping(self) -> dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "occasion_id": self.occasion_id,
            "case_id": self.case_id,
            "dose_response_artifact": self.dose_response_path.name,
            "dose_response_sha256": self.dose_response_sha256,
            "faithfulness_artifact": self.faithfulness_path.name,
            "faithfulness_sha256": self.faithfulness_sha256,
            "matched_recovery_artifact": self.matched_recovery_path.name,
            "matched_recovery_sha256": self.matched_recovery_sha256,
            "matched_recovery": {
                "classical_model_id": self.matched_recovery.classical_model_id,
                "information_set_id": self.matched_recovery.information_set_id,
                "metric_name": self.matched_recovery.metric_name,
                "classical_recovery_fraction": self.matched_recovery.recovery_fraction,
                "source_fingerprint": self.matched_recovery.source_fingerprint,
            },
        }


@dataclass(frozen=True)
class BMRBCausalBundle:
    study_id: str
    upstream_bmrb_path: Path
    upstream_bmrb_sha256: str
    upstream_source_fingerprint: str
    policy: CausalNecessityPolicy
    case_specs: tuple[BMRBCausalCaseSpec, ...]
    causal_result: CausalNecessityResult
    profile: MechanismNecessityProfile
    source_fingerprint: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_role": "bmrb_causal_bundle",
            "benchmark": "BMRB_DYNAMICS_V1",
            "study_id": self.study_id,
            "upstream_bmrb": self.upstream_bmrb_path.name,
            "upstream_bmrb_sha256": self.upstream_bmrb_sha256,
            "upstream_source_fingerprint": self.upstream_source_fingerprint,
            "policy": self.policy.to_mapping(),
            "cases": [item.to_mapping() for item in self.case_specs],
            "causal_evidence": self.causal_result.to_mapping(),
            "mechanism_profile": self.profile.to_mapping(),
            "source_fingerprint": self.source_fingerprint,
            "physical_quantum_promotion_eligible": False,
        }


def _resolve_file(root: Path, raw: Any, *, label: str) -> Path:
    value = Path(_required_text(label, raw)).expanduser()
    if not value.is_absolute():
        value = root / value
    value = value.resolve()
    if not value.is_file():
        raise FileNotFoundError(f"{label} not found: {value}")
    return value


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _load_upstream_profile(path: Path) -> tuple[dict[str, Any], MechanismNecessityProfile]:
    payload = _load_json_object(path, label="upstream BMRB artifact")
    if payload.get("artifact_role") != "bmrb_dynamics_bundle":
        raise ValueError("upstream BMRB artifact has the wrong artifact_role")
    if payload.get("benchmark") != "BMRB_DYNAMICS_V1":
        raise ValueError("upstream BMRB artifact is not BMRB_DYNAMICS_V1")
    profile_payload = payload.get("mechanism_profile")
    if not isinstance(profile_payload, Mapping):
        raise ValueError("upstream BMRB artifact is missing mechanism_profile")
    profile = mechanism_profile_from_mapping(profile_payload)
    return payload, profile


def _load_recovery(
    path: Path,
    *,
    study_id: str,
    participant_id: str,
    occasion_id: str,
    case_id: str,
) -> MatchedClassicalRecoveryEvidence:
    payload = _load_json_object(path, label="matched recovery artifact")
    evidence = matched_classical_recovery_from_mapping(payload)
    expected = {
        "study_id": study_id,
        "participant_id": participant_id,
        "occasion_id": occasion_id,
        "case_id": case_id,
    }
    for field, value in expected.items():
        if getattr(evidence, field) != value:
            raise ValueError(
                f"matched recovery {field} mismatch: {getattr(evidence, field)!r} != {value!r}"
            )
    return evidence


def load_bmrb_causal_manifest(
    path: str | Path,
) -> tuple[
    str,
    Path,
    CausalNecessityPolicy,
    tuple[BMRBCausalCaseSpec, ...],
    dict[str, Any],
]:
    manifest_path = Path(path).expanduser().resolve()
    payload = _load_json_object(manifest_path, label="BMRB causal manifest")
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("BMRB causal manifest schema_version must be 1")
    study_id = _required_text("study_id", payload.get("study_id"))
    upstream_path = _resolve_file(
        manifest_path.parent,
        payload.get("upstream_bmrb"),
        label="upstream_bmrb",
    )
    policy_payload = payload.get("policy")
    if not isinstance(policy_payload, Mapping):
        raise ValueError("BMRB causal manifest is missing policy")
    policy = CausalNecessityPolicy.from_mapping(policy_payload)
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) < 2:
        raise ValueError("BMRB causal manifest requires at least two case entries")

    cases: list[BMRBCausalCaseSpec] = []
    keys: set[tuple[str, str]] = set()
    case_ids: set[str] = set()
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, Mapping):
            raise ValueError(f"cases[{index}] must be an object")
        participant = _required_text(
            f"cases[{index}].participant_id", raw.get("participant_id")
        )
        occasion = _required_text(
            f"cases[{index}].occasion_id", raw.get("occasion_id")
        )
        case_id = _required_text(f"cases[{index}].case_id", raw.get("case_id"))
        if (participant, occasion) in keys:
            raise ValueError(
                "BMRB causal manifest permits one case per participant/occasion pair"
            )
        if case_id in case_ids:
            raise ValueError(f"duplicate causal case_id: {case_id!r}")
        dose = _resolve_file(
            manifest_path.parent,
            raw.get("dose_response_artifact"),
            label=f"cases[{index}].dose_response_artifact",
        )
        faith = _resolve_file(
            manifest_path.parent,
            raw.get("faithfulness_artifact"),
            label=f"cases[{index}].faithfulness_artifact",
        )
        recovery_path = _resolve_file(
            manifest_path.parent,
            raw.get("matched_recovery_artifact"),
            label=f"cases[{index}].matched_recovery_artifact",
        )
        recovery = _load_recovery(
            recovery_path,
            study_id=study_id,
            participant_id=participant,
            occasion_id=occasion,
            case_id=case_id,
        )
        cases.append(
            BMRBCausalCaseSpec(
                participant_id=participant,
                occasion_id=occasion,
                case_id=case_id,
                dose_response_path=dose,
                dose_response_sha256=_sha256_file(dose),
                faithfulness_path=faith,
                faithfulness_sha256=_sha256_file(faith),
                matched_recovery_path=recovery_path,
                matched_recovery_sha256=_sha256_file(recovery_path),
                matched_recovery=recovery,
            )
        )
        keys.add((participant, occasion))
        case_ids.add(case_id)
    return study_id, upstream_path, policy, tuple(cases), dict(payload)


def build_bmrb_causal_bundle(manifest_path: str | Path) -> BMRBCausalBundle:
    study_id, upstream_path, policy, case_specs, manifest = load_bmrb_causal_manifest(
        manifest_path
    )
    upstream_payload, upstream_profile = _load_upstream_profile(upstream_path)
    upstream_source_fingerprint = _required_text(
        "upstream source_fingerprint", upstream_payload.get("source_fingerprint")
    )
    mechanism_id = upstream_profile.mechanism_id

    causal_cases: list[CausalCaseEvidence] = []
    information_sets = {case.matched_recovery.information_set_id for case in case_specs}
    if len(information_sets) != 1:
        raise ValueError(
            "all matched-classical recovery cases must use one declared information_set_id"
        )
    for case in case_specs:
        if case.matched_recovery.mechanism_id != mechanism_id:
            raise ValueError(
                "matched recovery mechanism_id does not match upstream BMRB mechanism"
            )
        dose_payload = _load_json_object(
            case.dose_response_path, label="dose-response artifact"
        )
        faith_payload = _load_json_object(
            case.faithfulness_path, label="faithfulness artifact"
        )
        causal_cases.append(
            causal_case_from_neuros_mechint(
                participant_id=case.participant_id,
                occasion_id=case.occasion_id,
                case_id=case.case_id,
                mechanism_id=mechanism_id,
                dose_response=dose_payload,
                faithfulness=faith_payload,
                matched_recovery=case.matched_recovery.as_causal_recovery(),
            )
        )

    result = evaluate_causal_necessity(causal_cases, policy=policy)
    updated_profile = attach_causal_evidence(upstream_profile, result)
    upstream_sha256 = _sha256_file(upstream_path)
    identity = {
        "schema_version": 1,
        "study_id": study_id,
        "upstream_bmrb_sha256": upstream_sha256,
        "upstream_source_fingerprint": upstream_source_fingerprint,
        "policy": policy.to_mapping(),
        "cases": [case.to_mapping() for case in case_specs],
        "causal_source_fingerprint": result.source_fingerprint,
        "manifest_metadata": manifest.get("metadata", {}),
    }
    source_fingerprint = sha256(
        b"quantumbci.bmrb-causal.v1\0" + _canonical_json(identity).encode("utf-8")
    ).hexdigest()
    return BMRBCausalBundle(
        study_id=study_id,
        upstream_bmrb_path=upstream_path,
        upstream_bmrb_sha256=upstream_sha256,
        upstream_source_fingerprint=upstream_source_fingerprint,
        policy=policy,
        case_specs=case_specs,
        causal_result=result,
        profile=updated_profile,
        source_fingerprint=source_fingerprint,
    )


def render_bmrb_causal_html(bundle: BMRBCausalBundle) -> str:
    profile = bundle.profile
    promotion = (
        "none"
        if profile.promotion_ceiling is None
        else profile.promotion_ceiling.name.lower()
    )
    gate_rows = "".join(
        "<tr>"
        f"<td>{escape(gate.tier.name.lower())}</td>"
        f"<td>{escape(gate.id)}</td>"
        f"<td><strong>{escape(gate.status.value)}</strong></td>"
        f"<td>{escape(gate.summary)}</td>"
        "</tr>"
        for gate in profile.ordered_gates
    )
    participant_rows = "".join(
        "<tr>"
        f"<td>{escape(row.participant_id)}</td>"
        f"<td>{row.direction_match_fraction:.3f}</td>"
        f"<td>{row.dose_response_pass_fraction:.3f}</td>"
        f"<td>{row.faithfulness_pass_fraction:.3f}</td>"
        f"<td>{row.mean_necessity_fraction:.3f}</td>"
        f"<td>{row.mean_joint_random_percentile:.3f}</td>"
        f"<td>{row.mean_classical_recovery_fraction:.3f}</td>"
        "</tr>"
        for row in bundle.causal_result.participants
    )
    recovery_rows = "".join(
        "<tr>"
        f"<td>{escape(case.participant_id)}</td>"
        f"<td>{escape(case.matched_recovery.classical_model_id)}</td>"
        f"<td>{escape(case.matched_recovery.metric_name)}</td>"
        f"<td>{case.matched_recovery.ablation_loss:.6g}</td>"
        f"<td>{case.matched_recovery.restored_loss:.6g}</td>"
        f"<td>{case.matched_recovery.recovery_fraction:.3f}</td>"
        "</tr>"
        for case in bundle.case_specs
    )
    information_set_id = bundle.case_specs[0].matched_recovery.information_set_id
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(bundle.study_id)} · BMRB causal necessity</title>
<style>
body {{ font-family: ui-sans-serif,system-ui,sans-serif; max-width:1180px; margin:40px auto; padding:0 24px; line-height:1.45; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin:20px 0 28px; }}
.card {{ border:1px solid #bbb; border-radius:10px; padding:14px; }} .card b {{ display:block;font-size:1.3rem;margin-top:4px; }}
table {{ width:100%;border-collapse:collapse;margin:14px 0 30px; }} th,td {{ border-bottom:1px solid #ccc;padding:9px 8px;text-align:left;vertical-align:top; }}
.small {{ opacity:.75;font-size:.92rem; }} code {{ font-family:ui-monospace,monospace; }}
</style></head><body>
<h1>BMRB causal necessity · {escape(bundle.study_id)}</h1>
<p>Intervention direction, held-out faithfulness, ablation necessity and matched-classical recovery are evaluated together. Scientific criteria and promotion eligibility remain separate.</p>
<div class="cards">
<div class="card">Participants<b>{len(bundle.causal_result.participants)}</b></div>
<div class="card">Criteria passed<b>{str(bundle.causal_result.scientific_criteria_passed).lower()}</b></div>
<div class="card">Policy preregistered<b>{str(bundle.policy.preregistered).lower()}</b></div>
<div class="card">Causal promotion eligible<b>{str(bundle.causal_result.promotion_eligible).lower()}</b></div>
<div class="card">Promotion ceiling<b>{escape(promotion)}</b></div>
<div class="card">First falsifier<b>{escape(profile.first_failing_gate or 'none')}</b></div>
</div>
<h2>Mechanism necessity ladder</h2><table><thead><tr><th>Tier</th><th>Gate</th><th>Status</th><th>Interpretation</th></tr></thead><tbody>{gate_rows}</tbody></table>
<h2>Participant-balanced causal evidence</h2><table><thead><tr><th>Participant</th><th>Direction</th><th>Dose pass</th><th>Faithfulness pass</th><th>Necessity</th><th>Random percentile</th><th>Classical recovery</th></tr></thead><tbody>{participant_rows}</tbody></table>
<h2>Matched-classical recovery evidence</h2><p class="small">Information set: <code>{escape(information_set_id)}</code>. Recovery is derived from baseline, ablated and recovered metrics; it is not supplied as a free scalar.</p>
<table><thead><tr><th>Participant</th><th>Classical model</th><th>Metric</th><th>Ablation loss</th><th>Restored loss</th><th>Recovery fraction</th></tr></thead><tbody>{recovery_rows}</tbody></table>
<p class="small">Lower matched-classical recovery is stronger necessity evidence. Every recovery row has an independent source fingerprint and file SHA-256.</p>
<p class="small"><code>source_fingerprint={escape(bundle.source_fingerprint)}</code></p>
</body></html>"""


def write_bmrb_causal_bundle(
    bundle: BMRBCausalBundle,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "bmrb_causal.json"
    html_path = root / "report.html"
    json_path.write_text(
        json.dumps(bundle.to_mapping(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    html_path.write_text(render_bmrb_causal_html(bundle), encoding="utf-8")
    return json_path, html_path
