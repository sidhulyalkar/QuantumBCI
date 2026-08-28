"""Executable Brain Mechanism Recapitulation Benchmark bundles."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .claims import ClaimClass
from .recapitulation import (
    EvidenceGate,
    EvidenceTier,
    GateStatus,
    MechanismNecessityProfile,
    bmrb_dynamics_signature,
    mechanism_profile_from_mapping,
)
from .reliability import (
    DEFAULT_RELIABILITY_BOOTSTRAP_RESAMPLES,
    DEFAULT_RELIABILITY_SEED,
    RepeatedCaseEstimate,
    RepeatedCaseReliabilityBundle,
    audit_repeated_case_reliability,
    estimates_from_stability_artifact,
)


DEFAULT_E002_RELIABILITY_ESTIMATES = (
    "omega_x",
    "omega_z",
    "gamma_dephasing",
    "gamma_relaxation",
    "canonical_structure_residual",
    "canonical_minus_affine_one_step_rmse",
    "canonical_minus_affine_rollout_rmse",
    "direct_minus_nonlinear_mean_nll",
    "direct_minus_nonlinear_one_step_rmse",
)
BMRB_DYNAMICS_ARTIFACT_SCHEMA = 2


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


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


def _source_fingerprint(source_identity: Mapping[str, Any]) -> str:
    return sha256(
        b"quantumbci.bmrb-dynamics.v1\0"
        + _canonical_json(source_identity).encode("utf-8")
    ).hexdigest()


def _artifact_fingerprint(payload: Mapping[str, Any]) -> str:
    return sha256(
        b"quantumbci.bmrb-dynamics-artifact.v2\0"
        + _canonical_json(payload).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class BMRBCaseSpec:
    participant_id: str
    occasion_id: str
    case_id: str
    artifact_path: Path
    artifact_sha256: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "occasion_id": self.occasion_id,
            "case_id": self.case_id,
            "artifact": self.artifact_path.name,
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True)
class BMRBDynamicsBundle:
    study_id: str
    case_specs: tuple[BMRBCaseSpec, ...]
    reliability: RepeatedCaseReliabilityBundle
    profile: MechanismNecessityProfile
    source_identity: Mapping[str, Any]
    source_fingerprint: str

    def _artifact_core(self) -> dict[str, Any]:
        return {
            "schema_version": BMRB_DYNAMICS_ARTIFACT_SCHEMA,
            "artifact_role": "bmrb_dynamics_bundle",
            "benchmark": "BMRB_DYNAMICS_V1",
            "study_id": self.study_id,
            "source_identity": dict(self.source_identity),
            "source_fingerprint": self.source_fingerprint,
            "cases": [case.to_mapping() for case in self.case_specs],
            "reliability": self.reliability.to_mapping(),
            "mechanism_profile": self.profile.to_mapping(),
            "claim_ceiling": self.profile.claim_class.value,
        }

    @property
    def artifact_fingerprint(self) -> str:
        return _artifact_fingerprint(self._artifact_core())

    def to_mapping(self) -> dict[str, Any]:
        core = self._artifact_core()
        return {**core, "artifact_fingerprint": self.artifact_fingerprint}


def verify_bmrb_dynamics_mapping(payload: Mapping[str, Any]) -> MechanismNecessityProfile:
    """Verify a serialized v0.16+ BMRB-Dynamics artifact before scientific reuse.

    v0.15 artifacts remain valid terminal reports, but they did not serialize the
    complete source identity needed to independently recompute their source
    fingerprint. A causal stage therefore requires regeneration with v0.16+.
    """

    if payload.get("artifact_role") != "bmrb_dynamics_bundle":
        raise ValueError("upstream BMRB artifact has the wrong artifact_role")
    if payload.get("benchmark") != "BMRB_DYNAMICS_V1":
        raise ValueError("upstream BMRB artifact is not BMRB_DYNAMICS_V1")
    if int(payload.get("schema_version", 0)) < BMRB_DYNAMICS_ARTIFACT_SCHEMA:
        raise ValueError(
            "upstream BMRB-Dynamics artifact predates the self-verifying v0.16 schema; "
            "regenerate it with quantumbci-bmrb dynamics before causal promotion"
        )
    source_identity = payload.get("source_identity")
    if not isinstance(source_identity, Mapping):
        raise ValueError("upstream BMRB artifact is missing source_identity")
    claimed_source = _required_text(
        "upstream source_fingerprint", payload.get("source_fingerprint")
    )
    if _source_fingerprint(source_identity) != claimed_source:
        raise ValueError("upstream BMRB source fingerprint mismatch")

    core = {
        key: value for key, value in payload.items() if key != "artifact_fingerprint"
    }
    claimed_artifact = _required_text(
        "upstream artifact_fingerprint", payload.get("artifact_fingerprint")
    )
    if _artifact_fingerprint(core) != claimed_artifact:
        raise ValueError("upstream BMRB artifact fingerprint mismatch")

    study_id = _required_text("upstream study_id", payload.get("study_id"))
    if source_identity.get("study_id") != study_id:
        raise ValueError("upstream BMRB source_identity study_id mismatch")
    cases = payload.get("cases")
    if source_identity.get("cases") != cases:
        raise ValueError("upstream BMRB source_identity cases mismatch")
    reliability = payload.get("reliability")
    if not isinstance(reliability, Mapping):
        raise ValueError("upstream BMRB artifact is missing reliability")
    if source_identity.get("reliability_source_fingerprint") != reliability.get(
        "source_fingerprint"
    ):
        raise ValueError("upstream BMRB reliability fingerprint linkage mismatch")
    profile_payload = payload.get("mechanism_profile")
    if not isinstance(profile_payload, Mapping):
        raise ValueError("upstream BMRB artifact is missing mechanism_profile")
    profile = mechanism_profile_from_mapping(profile_payload)
    if payload.get("claim_ceiling") != profile.claim_class.value:
        raise ValueError("upstream BMRB claim_ceiling mismatch")
    return profile


def load_bmrb_case_manifest(path: str | Path) -> tuple[str, tuple[BMRBCaseSpec, ...], dict[str, Any]]:
    """Load a local manifest of qualified case-level v0.14 artifacts."""

    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("BMRB case manifest must be a JSON object")
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("BMRB case manifest schema_version must be 1")
    study_id = _required_text("study_id", payload.get("study_id"))
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) < 2:
        raise ValueError("BMRB case manifest requires at least two cases")

    cases: list[BMRBCaseSpec] = []
    case_ids: set[str] = set()
    participant_occasions: set[tuple[str, str]] = set()
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, Mapping):
            raise ValueError(f"cases[{index}] must be an object")
        participant = _required_text(
            f"cases[{index}].participant_id", raw.get("participant_id")
        )
        occasion = _required_text(f"cases[{index}].occasion_id", raw.get("occasion_id"))
        case_id = _required_text(f"cases[{index}].case_id", raw.get("case_id"))
        if case_id in case_ids:
            raise ValueError(f"duplicate case_id {case_id!r}")
        pair = (participant, occasion)
        if pair in participant_occasions:
            raise ValueError(
                "BMRB v1 accepts one case artifact per participant/occasion pair; "
                f"duplicate={pair}"
            )
        artifact_value = _required_text(f"cases[{index}].artifact", raw.get("artifact"))
        artifact_path = Path(artifact_value).expanduser()
        if not artifact_path.is_absolute():
            artifact_path = manifest_path.parent / artifact_path
        artifact_path = artifact_path.resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(f"case artifact not found: {artifact_path}")
        cases.append(
            BMRBCaseSpec(
                participant_id=participant,
                occasion_id=occasion,
                case_id=case_id,
                artifact_path=artifact_path,
                artifact_sha256=_sha256_file(artifact_path),
            )
        )
        case_ids.add(case_id)
        participant_occasions.add(pair)
    return study_id, tuple(cases), dict(payload)


def _read_stability_artifact(case: BMRBCaseSpec) -> dict[str, Any]:
    payload = json.loads(case.artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"case artifact {case.case_id!r} must contain a JSON object")
    return payload


def _build_e002_profile(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    reliability: RepeatedCaseReliabilityBundle,
) -> MechanismNecessityProfile:
    if not artifacts:
        raise ValueError("BMRB-Dynamics requires case artifacts")
    claim_classes = {str(artifact.get("claim_class")) for artifact in artifacts}
    if claim_classes != {ClaimClass.QUANTUM_INSPIRED.value}:
        raise ValueError(
            "BMRB-Dynamics v1 expects quantum_inspired E002 stability artifacts; "
            f"observed={sorted(claim_classes)}"
        )
    if not all(bool(artifact.get("predictive_adversary_ladder_complete", False)) for artifact in artifacts):
        raise ValueError("every case must complete the predictive adversary ladder")
    if not all(artifact.get("stability_gate_defined") is False for artifact in artifacts):
        raise ValueError("v0.15 expects v0.14 artifacts with no universal stability gate")

    novelty = [bool(artifact.get("dynamical_information_novel", False)) for artifact in artifacts]
    if not any(novelty):
        adversary_status = GateStatus.FAIL
        adversary_summary = (
            "Every declared case reports dynamical_information_novel=false after the matched "
            "predictive adversary ladder. Current evidence therefore falsifies promotion of the "
            "canonical open-system parameterization as uniquely necessary dynamics information."
        )
    else:
        adversary_status = GateStatus.CHARACTERIZED
        adversary_summary = (
            "At least one case reports surviving dynamical information, but BMRB v1 does not "
            "invent a pooled adversary-survival threshold. A preregistered cross-case decision "
            "rule is required before promotion."
        )

    gates = (
        EvidenceGate(
            id="descriptive_contract",
            tier=EvidenceTier.DESCRIPTIVE,
            status=GateStatus.PASS,
            summary="All declared cases are qualified fixed-evaluation E002 stability artifacts.",
            threshold="all declared case artifacts pass execution and preserve fixed evaluation",
        ),
        EvidenceGate(
            id="predictive_sufficiency",
            tier=EvidenceTier.PREDICTIVE,
            status=GateStatus.CHARACTERIZED,
            summary=(
                "Held-out one-step, rollout and predictive-density comparisons are available for "
                "every case, but no cross-case predictive promotion threshold is declared here."
            ),
            evidence_ref="v0.9-v0.14 case artifacts",
        ),
        EvidenceGate(
            id="matched_classical_adversaries",
            tier=EvidenceTier.ADVERSARY_SURVIVING,
            status=adversary_status,
            summary=adversary_summary,
            evidence_ref="dynamical_information_novel",
        ),
        EvidenceGate(
            id="source_resampling_stability",
            tier=EvidenceTier.SOURCE_STABILITY,
            status=GateStatus.CHARACTERIZED,
            summary=(
                "Trajectory-block source bootstrap intervals, sign consistency, predictive-gain "
                "survival and nonlinear-selection frequencies are available per case. No universal "
                "stability pass threshold is defined."
            ),
            evidence_ref="bootstrap_stability_evidence",
        ),
        EvidenceGate(
            id="repeated_case_reliability",
            tier=EvidenceTier.REPEATED_CASE,
            status=GateStatus.CHARACTERIZED,
            summary=(
                "Population recurrence and repeated-participant reliability are summarized across "
                "the declared cases. ICC is computed only for complete balanced panels. No universal "
                "reliability pass threshold is defined."
            ),
            evidence_ref=reliability.source_fingerprint,
        ),
        EvidenceGate(
            id="causal_intervention_and_ablation",
            tier=EvidenceTier.CAUSAL_MECHANISTIC,
            status=GateStatus.NOT_RUN,
            summary=(
                "Mechanism-specific intervention direction, dose response and ablation recovery "
                "controls are not yet present in this bundle."
            ),
        ),
        EvidenceGate(
            id="physical_quantum_witness",
            tier=EvidenceTier.PHYSICAL_QUANTUM,
            status=GateStatus.NOT_APPLICABLE,
            summary=(
                "The E002 canonical open-system model is quantum-inspired. Physical-quantum "
                "promotion requires an independent substrate/witness protocol and cannot be inferred "
                "from this benchmark."
            ),
        ),
    )
    return MechanismNecessityProfile(
        mechanism_id="lindblad_latent_dynamics",
        claim_class=ClaimClass.QUANTUM_INSPIRED,
        signature=bmrb_dynamics_signature(),
        gates=gates,
        metadata={
            "case_count": len(artifacts),
            "participant_count": reliability.participant_count,
            "reliability_source_fingerprint": reliability.source_fingerprint,
        },
    )


def build_bmrb_dynamics_bundle(
    manifest_path: str | Path,
    *,
    estimate_names: Sequence[str] = DEFAULT_E002_RELIABILITY_ESTIMATES,
    n_resamples: int = DEFAULT_RELIABILITY_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_RELIABILITY_SEED,
) -> BMRBDynamicsBundle:
    study_id, case_specs, manifest = load_bmrb_case_manifest(manifest_path)
    rows: list[RepeatedCaseEstimate] = []
    artifacts: list[dict[str, Any]] = []
    for case in case_specs:
        artifact = _read_stability_artifact(case)
        artifacts.append(artifact)
        rows.extend(
            estimates_from_stability_artifact(
                artifact,
                participant_id=case.participant_id,
                occasion_id=case.occasion_id,
                case_id=case.case_id,
                artifact_sha256=case.artifact_sha256,
                estimate_names=estimate_names,
            )
        )

    reliability = audit_repeated_case_reliability(
        rows,
        study_id=study_id,
        n_resamples=n_resamples,
        seed=seed,
    )
    profile = _build_e002_profile(artifacts, reliability=reliability)
    source_identity = {
        "schema_version": 1,
        "study_id": study_id,
        "manifest": {
            "case_count": len(case_specs),
            "declared_metadata": manifest.get("metadata", {}),
        },
        "cases": [case.to_mapping() for case in case_specs],
        "reliability_source_fingerprint": reliability.source_fingerprint,
        "estimate_names": list(estimate_names),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }
    fingerprint = _source_fingerprint(source_identity)
    return BMRBDynamicsBundle(
        study_id=study_id,
        case_specs=case_specs,
        reliability=reliability,
        profile=profile,
        source_identity=source_identity,
        source_fingerprint=fingerprint,
    )


def render_bmrb_dynamics_html(bundle: BMRBDynamicsBundle) -> str:
    profile = bundle.profile
    gate_rows = []
    for gate in profile.ordered_gates:
        gate_rows.append(
            "<tr>"
            f"<td>{escape(gate.tier.name.lower())}</td>"
            f"<td>{escape(gate.id)}</td>"
            f"<td><strong>{escape(gate.status.value)}</strong></td>"
            f"<td>{escape(gate.summary)}</td>"
            "</tr>"
        )

    reliability_rows = []
    for result in bundle.reliability.results:
        icc = "n/a" if result.icc is None else f"{result.icc.value:.3f}"
        reliability_rows.append(
            "<tr>"
            f"<td>{escape(result.estimate_name)}</td>"
            f"<td>{result.grand_mean:.6g}</td>"
            f"<td>{result.population_sign_consistency:.3f}</td>"
            f"<td>[{result.bootstrap_ci_low:.6g}, {result.bootstrap_ci_high:.6g}]</td>"
            f"<td>{escape(icc)}</td>"
            "</tr>"
        )

    promotion = (
        "none"
        if profile.promotion_ceiling is None
        else profile.promotion_ceiling.name.lower()
    )
    failing = profile.first_failing_gate or "none"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(bundle.study_id)} · BMRB-Dynamics</title>
<style>
body {{ font-family: ui-sans-serif, system-ui, sans-serif; max-width: 1180px; margin: 40px auto; padding: 0 24px; line-height: 1.45; }}
h1, h2 {{ letter-spacing: -0.02em; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(190px,1fr)); gap: 12px; margin: 20px 0 28px; }}
.card {{ border: 1px solid #bbb; border-radius: 10px; padding: 14px; }}
.card b {{ display: block; font-size: 1.35rem; margin-top: 4px; }}
table {{ width: 100%; border-collapse: collapse; margin: 14px 0 30px; }}
th, td {{ border-bottom: 1px solid #ccc; padding: 9px 8px; text-align: left; vertical-align: top; }}
code {{ font-family: ui-monospace, monospace; }}
.small {{ opacity: .75; font-size: .92rem; }}
</style>
</head>
<body>
<h1>BMRB-Dynamics · {escape(bundle.study_id)}</h1>
<p>{escape(profile.signature.description)}</p>
<div class="cards">
<div class="card">Cases<b>{len(bundle.case_specs)}</b></div>
<div class="card">Participants<b>{bundle.reliability.participant_count}</b></div>
<div class="card">Evidence coverage<b>{escape(profile.evidence_coverage_tier.name.lower())}</b></div>
<div class="card">Promotion ceiling<b>{escape(promotion)}</b></div>
<div class="card">First falsifier<b>{escape(failing)}</b></div>
</div>
<h2>Mechanism necessity ladder</h2>
<table><thead><tr><th>Tier</th><th>Gate</th><th>Status</th><th>Interpretation</th></tr></thead>
<tbody>{''.join(gate_rows)}</tbody></table>
<h2>Repeated-case mechanism quantities</h2>
<table><thead><tr><th>Quantity</th><th>Participant-weighted mean</th><th>Sign consistency</th><th>Hierarchical bootstrap 95% interval</th><th>ICC(A,1)</th></tr></thead>
<tbody>{''.join(reliability_rows)}</tbody></table>
<p class="small">ICC is shown only for a complete balanced participant × occasion panel and measures reproducibility of individual differences, not population recurrence. Evidence coverage is not the same as promotion.</p>
<p class="small"><code>source_fingerprint={escape(bundle.source_fingerprint)}</code><br><code>artifact_fingerprint={escape(bundle.artifact_fingerprint)}</code></p>
</body>
</html>
"""


def write_bmrb_dynamics_bundle(
    bundle: BMRBDynamicsBundle,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "bmrb_dynamics.json"
    html_path = root / "report.html"
    json_path.write_text(
        json.dumps(bundle.to_mapping(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    html_path.write_text(render_bmrb_dynamics_html(bundle), encoding="utf-8")
    return json_path, html_path
