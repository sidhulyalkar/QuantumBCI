"""Materialize the v0.11 E002 probabilistic state-space control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..classical_dynamics import DIRECT_DISCRETE_ESTIMATOR_ID, run_extended_classical_controls
from ..probabilistic_ssm import run_probabilistic_state_space_control
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


def _mean_model_sha256(transition: np.ndarray, intercept: np.ndarray) -> str:
    payload = {
        "transition": np.asarray(transition, dtype=float).tolist(),
        "intercept": np.asarray(intercept, dtype=float).reshape(-1).tolist(),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(b"quantumbci.e002.mean-model.v1\0" + canonical).hexdigest()


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
        raise ValueError("trajectory index does not require one shared tensor across model lanes")


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
        raise ValueError("v0.11 requires the unregularized v0.9 matched baseline")
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


def _verify_classical_controls(
    classical: dict[str, Any],
    *,
    identity: dict[str, str],
    data: Any,
    authority: Any,
) -> dict[str, Any]:
    if classical.get("experiment") != "E002":
        raise ValueError("classical controls artifact is not E002")
    if classical.get("artifact_role") != "extended_classical_dynamics_controls":
        raise ValueError("classical controls artifact has the wrong artifact role")
    if classical.get("status") != "pass":
        raise ValueError("classical controls artifact did not pass")
    if not bool(classical.get("upstream_matched_baseline_verified", False)):
        raise ValueError("classical controls did not verify the v0.9 matched baseline")
    if not bool(classical.get("linear_observed_control_stage_complete", False)):
        raise ValueError("v0.10 observed-state linear control stage is incomplete")
    if not bool(classical.get("probabilistic_latent_state_space_control_required", False)):
        raise ValueError("classical controls do not require the probabilistic state-space stage")
    if bool(classical.get("intervention_stage_eligible", True)):
        raise ValueError("classical controls incorrectly permit intervention promotion")
    if bool(classical.get("physical_quantum_promotion_eligible", True)):
        raise ValueError("classical controls incorrectly permit physical-quantum promotion")
    for field, expected in identity.items():
        if str(classical.get(field)) != expected:
            raise ValueError(f"classical controls {field} differs from current trajectory authority")

    controls = classical.get("controls")
    if not isinstance(controls, dict):
        raise ValueError("classical controls artifact is missing controls")
    full_var = controls.get("full_var1")
    if not isinstance(full_var, dict):
        raise ValueError("classical controls artifact is missing full_var1")
    if full_var.get("model_id") != "full_var1_affine":
        raise ValueError("v0.11 requires the v0.10 full_var1_affine mean model")
    if full_var.get("estimator_id") != DIRECT_DISCRETE_ESTIMATOR_ID:
        raise ValueError("v0.10 full VAR estimator identity is unexpected")
    if int(full_var.get("parameter_count", -1)) != 12:
        raise ValueError("v0.11 requires the full 12-parameter v0.10 VAR mean model")
    for field, expected in identity.items():
        if str(full_var.get(field)) != expected:
            raise ValueError(
                f"classical controls full_var1.{field} differs from current trajectory authority"
            )

    # Reconstruct the v0.10 control artifact independently from the frozen tensor and
    # authority. This prevents a hand-edited full-VAR transition from becoming the
    # authoritative v0.11 mean merely because its surrounding hashes were left intact.
    expected_controls = run_extended_classical_controls(data, authority).to_mapping()["controls"]
    expected_full_var = expected_controls["full_var1"]
    if full_var != expected_full_var:
        raise ValueError(
            "classical controls full_var1 differs from independent reconstruction under current authority"
        )

    transition = np.asarray(full_var.get("transition"), dtype=float)
    intercept = np.asarray(full_var.get("intercept"), dtype=float).reshape(-1)
    if transition.shape != (3, 3) or intercept.shape != (3,):
        raise ValueError("v0.11 E002 probabilistic control requires a 3D full VAR mean model")
    if not np.all(np.isfinite(transition)) or not np.all(np.isfinite(intercept)):
        raise ValueError("v0.10 full VAR mean model contains non-finite values")
    return full_var


def build_probabilistic_ssm_artifact(
    *,
    descriptor_path: Path,
    trajectory_index_path: Path,
    matched_path: Path,
    classical_controls_path: Path,
) -> dict[str, Any]:
    data, authority = load_trajectory_contract_descriptor(descriptor_path)
    trajectory_index = _read_json(trajectory_index_path, label="trajectory index artifact")
    matched = _read_json(matched_path, label="matched dynamics artifact")
    classical = _read_json(classical_controls_path, label="classical controls artifact")

    _verify_trajectory_index(trajectory_index, data=data, authority=authority)
    identity = _expected_identity(data, authority)
    _verify_matched_baseline(matched, identity=identity)
    full_var = _verify_classical_controls(
        classical,
        identity=identity,
        data=data,
        authority=authority,
    )

    transition = np.asarray(full_var["transition"], dtype=float)
    intercept = np.asarray(full_var["intercept"], dtype=float)
    mean_model_sha256 = _mean_model_sha256(transition, intercept)
    result = run_probabilistic_state_space_control(
        data,
        authority,
        transition,
        intercept,
    )
    payload = result.to_mapping()
    for field, expected in identity.items():
        if str(payload.get(field)) != expected:
            raise RuntimeError(f"probabilistic state-space control produced mismatched {field}")
    if not np.array_equal(np.asarray(payload["transition"], dtype=float), transition):
        raise RuntimeError("probabilistic control refit or changed the frozen v0.10 transition")
    if not np.array_equal(np.asarray(payload["intercept"], dtype=float), intercept):
        raise RuntimeError("probabilistic control refit or changed the frozen v0.10 intercept")

    return {
        **payload,
        "schema_version": 2,
        "status": "pass",
        "artifact_role": "probabilistic_latent_state_space_control",
        "descriptor_name": descriptor_path.name,
        "trajectory_index_name": trajectory_index_path.name,
        "matched_baseline_name": matched_path.name,
        "classical_controls_name": classical_controls_path.name,
        "upstream_trajectory_authority_verified": True,
        "upstream_matched_baseline_verified": True,
        "upstream_classical_controls_verified": True,
        "mean_transition_source": "v0.10:controls.full_var1",
        "mean_model_sha256": mean_model_sha256,
        "probabilistic_latent_state_space_control_complete": True,
        "switching_state_control_required": True,
        "flexible_nonlinear_control_required_when_powered": True,
        "bootstrap_stability_required": True,
        "intervention_stage_eligible": False,
        "dynamical_information_novel": False,
        "physical_quantum_promotion_eligible": False,
        "interpretation_ceiling": (
            "This artifact establishes a calibrated classical latent-noise control with a "
            "fixed identity observation matrix and the exact frozen v0.10 VAR mean model. "
            "Any gain is evidence for filtering or uncertainty modeling, not quantum novelty. "
            "Switching-state, nonlinear, stability, and intervention gates remain open."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", default="trajectory_contract.json")
    parser.add_argument("--trajectory-index", default="trajectory_index.json")
    parser.add_argument("--matched", default="matched_dynamics.json")
    parser.add_argument("--classical-controls", default="classical_controls.json")
    parser.add_argument("--output", default="probabilistic_ssm.json")
    args = parser.parse_args(argv)

    output = Path(args.output)
    try:
        payload = build_probabilistic_ssm_artifact(
            descriptor_path=Path(args.descriptor),
            trajectory_index_path=Path(args.trajectory_index),
            matched_path=Path(args.matched),
            classical_controls_path=Path(args.classical_controls),
        )
        _write_json(output, payload)
        print(json.dumps({**payload, "artifact": str(output)}, sort_keys=True))
        return 0
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
