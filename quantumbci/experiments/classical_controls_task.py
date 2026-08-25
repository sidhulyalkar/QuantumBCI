"""Materialize the v0.10 E002 classical-control ladder under frozen evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..classical_dynamics import run_extended_classical_controls
from ..trajectory_authority import load_trajectory_contract_descriptor


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _transition_sha256(pairs: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(pairs, dtype=np.int64).reshape(-1, 2))
    digest = hashlib.sha256()
    digest.update(b"quantumbci.trajectory-transitions.v1\0")
    digest.update(str(values.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def _expected_identity(data: Any, authority: Any) -> dict[str, str]:
    return {
        "authority_fingerprint": authority.authority_fingerprint,
        "data_sha256": data.data_sha256,
        "fit_transition_sha256": _transition_sha256(authority.transition_pairs(data, "fit")),
        "evaluation_transition_sha256": _transition_sha256(
            authority.transition_pairs(data, "evaluation")
        ),
    }


def _verify_trajectory_index(
    trajectory_index: dict[str, Any],
    *,
    data: Any,
    authority: Any,
) -> None:
    if trajectory_index.get("experiment") != "E002":
        raise ValueError("trajectory index is not an E002 artifact")
    if trajectory_index.get("artifact_role") != "trajectory_evidence_authority":
        raise ValueError("trajectory index has the wrong artifact role")
    if trajectory_index.get("authority") != authority.to_dict(data=data):
        raise ValueError("trajectory index authority differs from reconstructed descriptor authority")
    shared = trajectory_index.get("shared_tensor_contract")
    if not isinstance(shared, dict):
        raise ValueError("trajectory index is missing shared_tensor_contract")
    if str(shared.get("data_sha256")) != data.data_sha256:
        raise ValueError("trajectory index state SHA-256 differs from descriptor data")
    if not bool(shared.get("required_for_all_model_lanes", False)):
        raise ValueError("trajectory index does not require a shared tensor across model lanes")


def _verify_matched_baseline(
    matched: dict[str, Any],
    *,
    identity: dict[str, str],
) -> None:
    if matched.get("experiment") != "E002":
        raise ValueError("matched baseline is not an E002 artifact")
    if matched.get("artifact_role") != "matched_dynamics_baseline":
        raise ValueError("matched baseline has the wrong artifact role")
    if matched.get("status") != "pass" or not bool(matched.get("same_evidence_verified", False)):
        raise ValueError("matched baseline did not pass same-evidence verification")
    if float(matched.get("authoritative_ridge", float("nan"))) != 0.0:
        raise ValueError("v0.10 requires the unregularized v0.9 matched baseline")
    if not bool(matched.get("regularization_geometry_matched", False)):
        raise ValueError("matched baseline did not certify regularization geometry")
    if bool(matched.get("dynamical_information_novel", True)):
        raise ValueError("matched baseline incorrectly claims dynamical-information novelty")
    if bool(matched.get("physical_quantum_promotion_eligible", True)):
        raise ValueError("matched baseline incorrectly permits physical-quantum promotion")

    for field, expected in identity.items():
        if str(matched.get(field)) != expected:
            raise ValueError(f"matched baseline {field} differs from current trajectory authority")
    for lane_name in ("affine", "canonical"):
        lane = matched.get(lane_name)
        if not isinstance(lane, dict):
            raise ValueError(f"matched baseline is missing {lane_name} lane")
        for field, expected in identity.items():
            if str(lane.get(field)) != expected:
                raise ValueError(
                    f"matched baseline {lane_name}.{field} differs from current trajectory authority"
                )


def _metric(mapping: dict[str, Any], name: str) -> float:
    value = mapping.get(name)
    if value is None:
        raise ValueError(f"required metric {name!r} is missing")
    return float(value)


def build_classical_controls_artifact(
    *,
    descriptor_path: Path,
    trajectory_index_path: Path,
    matched_path: Path,
) -> dict[str, Any]:
    data, authority = load_trajectory_contract_descriptor(descriptor_path)
    trajectory_index = _read_json(trajectory_index_path, label="trajectory index artifact")
    matched = _read_json(matched_path, label="matched dynamics artifact")
    _verify_trajectory_index(trajectory_index, data=data, authority=authority)
    identity = _expected_identity(data, authority)
    _verify_matched_baseline(matched, identity=identity)

    controls = run_extended_classical_controls(data, authority)
    control_payload = controls.to_mapping()
    for field, expected in identity.items():
        if str(control_payload.get(field)) != expected:
            raise RuntimeError(f"classical controls produced mismatched {field}")

    canonical = matched["canonical"]["evaluation_metrics"]
    affine = matched["affine"]["evaluation_metrics"]
    full = control_payload["controls"]["full_var1"]["evaluation_metrics"]
    diagonal = control_payload["controls"]["diagonal_ar1"]["evaluation_metrics"]
    persistence = control_payload["controls"]["persistence"]["evaluation_metrics"]

    best_one_step = min(
        (
            ("persistence", _metric(persistence, "one_step_rmse")),
            ("diagonal_ar1", _metric(diagonal, "one_step_rmse")),
            ("full_var1_affine", _metric(full, "one_step_rmse")),
        ),
        key=lambda item: item[1],
    )
    best_rollout = min(
        (
            ("persistence", _metric(persistence, "rollout_rmse")),
            ("diagonal_ar1", _metric(diagonal, "rollout_rmse")),
            ("full_var1_affine", _metric(full, "rollout_rmse")),
        ),
        key=lambda item: item[1],
    )

    return {
        **control_payload,
        "status": "pass",
        "artifact_role": "extended_classical_dynamics_controls",
        "schema_version": 2,
        "descriptor_name": descriptor_path.name,
        "trajectory_index_name": trajectory_index_path.name,
        "matched_baseline_name": matched_path.name,
        "upstream_matched_baseline_verified": True,
        "comparisons_to_v0_9": {
            "canonical_minus_full_var1_one_step_rmse": (
                _metric(canonical, "one_step_rmse") - _metric(full, "one_step_rmse")
            ),
            "canonical_minus_full_var1_rollout_rmse": (
                _metric(canonical, "rollout_rmse") - _metric(full, "rollout_rmse")
            ),
            "continuous_affine_minus_full_var1_one_step_rmse": (
                _metric(affine, "one_step_rmse") - _metric(full, "one_step_rmse")
            ),
            "continuous_affine_minus_full_var1_rollout_rmse": (
                _metric(affine, "rollout_rmse") - _metric(full, "rollout_rmse")
            ),
            "canonical_beats_full_var1_one_step": (
                _metric(canonical, "one_step_rmse") < _metric(full, "one_step_rmse")
            ),
            "canonical_beats_full_var1_rollout": (
                _metric(canonical, "rollout_rmse") < _metric(full, "rollout_rmse")
            ),
            "best_extended_one_step_model": best_one_step[0],
            "best_extended_rollout_model": best_rollout[0],
        },
        "linear_observed_control_stage_complete": True,
        "probabilistic_latent_state_space_control_required": True,
        "switching_state_control_required": True,
        "flexible_nonlinear_control_required_when_powered": True,
        "intervention_stage_eligible": False,
        "physical_quantum_promotion_eligible": False,
        "interpretation_ceiling": (
            "The v0.10 artifact expands the classical linear observed-state ladder. VAR(1), "
            "direct discrete affine transition, and the fully observed identity-observation "
            "LDS mean are one model class here. A distinct Kalman claim requires an explicit "
            "latent observation/noise contract and probabilistic scoring."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", default="trajectory_contract.json")
    parser.add_argument("--trajectory-index", default="trajectory_index.json")
    parser.add_argument("--matched", default="matched_dynamics.json")
    parser.add_argument("--output", default="classical_controls.json")
    args = parser.parse_args(argv)

    output = Path(args.output)
    try:
        payload = build_classical_controls_artifact(
            descriptor_path=Path(args.descriptor),
            trajectory_index_path=Path(args.trajectory_index),
            matched_path=Path(args.matched),
        )
        _write_json(output, payload)
        print(json.dumps({**payload, "artifact": str(output)}, sort_keys=True))
        return 0
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
