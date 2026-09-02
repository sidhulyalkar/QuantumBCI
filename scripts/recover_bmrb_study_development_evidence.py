#!/usr/bin/env python3
"""Recover and analyze the complete BMRB study development surface.

This script consumes the already-computed 24 development shards, recomposes them with the
qualified v1 implementation, verifies the complete artifact, and emits an applicability-aware
development evidence capsule. It never executes the evaluation partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from quantumbci.bmrb_study_operating_artifacts import verify_bmrb_study_operating_mapping
from quantumbci.bmrb_study_operating_shards import merge_bmrb_study_operating_shards
from quantumbci.bmrb_study_operating import (
    BMRBStudyOperatingPolicy,
    recommended_development_grid,
)

SCIENCE_SOURCE_SHA = "681ea12c436fce121ba74de6f877a8267e94dd3f"
SOURCE_WORKFLOW_RUN_ID = 33467852855
SOURCE_WORKFLOW_HEAD_SHA = "4838be9857a42b0633dfd53c78cdc62f1a45df5d"
STUDY_ID = "bmrb-study-development-v1"
REPLICATES_PER_CELL = 8
BOOTSTRAP_RESAMPLES = 100

GENERIC_METRICS = (
    "observed_replication_pass_rate",
    "decision_error_rate",
    "context_specific_match_rate",
    "sensitivity_warning_match_rate",
    "mean_successful_replication_margin",
    "mean_study_effect_range",
)


def _mean(rows: list[dict[str, object]], metric: str) -> float:
    return float(mean(float(row[metric]) for row in rows))


def _summarize(rows: list[dict[str, object]]) -> dict[str, float]:
    return {metric: _mean(rows, metric) for metric in GENERIC_METRICS}


def _wilson(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denominator
    radius = z * ((p * (1.0 - p) + z2 / (4.0 * trials)) / trials) ** 0.5 / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _policy() -> BMRBStudyOperatingPolicy:
    return BMRBStudyOperatingPolicy(
        study_id=STUDY_ID,
        source_sha=SCIENCE_SOURCE_SHA,
        partition="development",
        grid=recommended_development_grid(),
        replicates_per_cell=REPLICATES_PER_CELL,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
    )


def _load_shards(shard_dir: Path) -> list[dict[str, object]]:
    paths = sorted(shard_dir.glob("shard-*.json"))
    expected = [f"shard-{index:02d}.json" for index in range(24)]
    observed = [path.name for path in paths]
    if len(paths) != 24:
        raise RuntimeError(f"expected 24 exact shards, found {len(paths)}")
    if observed != expected:
        raise RuntimeError(f"unexpected shard family: observed={observed} expected={expected}")
    payloads: list[dict[str, object]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a JSON object")
        payloads.append(payload)
    return payloads


def _recompose(shard_payloads: list[dict[str, object]]) -> dict[str, object]:
    result = merge_bmrb_study_operating_shards(_policy(), shard_payloads)
    payload = result.to_mapping()
    verify_bmrb_study_operating_mapping(payload)
    if len(payload["cells"]) != 648:
        raise RuntimeError("recomposed operating surface is not the exact 648-cell grid")
    if payload["policy"]["source_sha"] != SCIENCE_SOURCE_SHA:
        raise RuntimeError("science source SHA drifted during recomposition")
    if payload["policy"]["partition"] != "development":
        raise RuntimeError("only development evidence may be recovered")
    if payload["qualification_defined"] is not False:
        raise RuntimeError("development evidence must not define qualification")
    if payload["evaluation_partition_executed"] is not False:
        raise RuntimeError("evaluation partition must remain unexecuted")
    if payload["physical_quantum_promotion_eligible"] is not False:
        raise RuntimeError("physical-quantum promotion must remain false")
    return payload


def _analyze(result: dict[str, object]) -> dict[str, object]:
    cells = result["cells"]
    assert isinstance(cells, list)

    by_scenario: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_scenario_n: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    by_scenario_cross: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for cell in cells:
        assert isinstance(cell, dict)
        scenario_id = str(cell["scenario_id"])
        by_scenario[scenario_id].append(cell)
        by_scenario_n[(scenario_id, int(cell["participant_count"]))].append(cell)
        by_scenario_cross[(scenario_id, float(cell["cross_study_effect_scale"]))].append(cell)

    def select(ids: set[str]) -> list[dict[str, object]]:
        return [cell for cell in cells if str(cell["scenario_id"]) in ids]

    pure_null = {"homogeneous-null-3", "homogeneous-null-4"}
    homogeneous_positive = {"homogeneous-positive-3", "homogeneous-positive-4"}
    contextual_veto = {"primary-only-positive-4", "primary-fail-replications-positive-4"}
    conflicted_positive = {"fragile-one-conflict-4", "redundant-one-conflict-5"}

    primary_fail = by_scenario["primary-fail-replications-positive-4"]
    fragile = by_scenario["fragile-one-conflict-4"]

    worst = sorted(
        cells,
        key=lambda row: (
            float(row["decision_error_rate"]),
            1.0 - float(row["sensitivity_warning_match_rate"]),
        ),
        reverse=True,
    )[:20]
    worst_cells = [
        {
            key: row[key]
            for key in (
                "scenario_id",
                "participant_count",
                "within_study_heterogeneity_scale",
                "measurement_noise_scale",
                "cross_study_effect_scale",
                "observed_replication_pass_rate",
                "decision_error_rate",
                "sensitivity_warning_match_rate",
                "pass_rate_ci_lower",
                "pass_rate_ci_upper",
            )
        }
        for row in worst
    ]

    policy = result["policy"]
    assert isinstance(policy, dict)
    return {
        "purpose": "applicability_aware_development_analysis_v1",
        "scientific_partition": "development",
        "science_source_sha": policy["source_sha"],
        "policy_fingerprint": policy["policy_fingerprint"],
        "artifact_fingerprint": result["artifact_fingerprint"],
        "cell_count": len(cells),
        "replicates_per_cell": policy["replicates_per_cell"],
        "bootstrap_resamples": policy["bootstrap_resamples"],
        "evaluation_partition_executed": False,
        "qualification_defined": False,
        "physical_quantum_promotion_eligible": False,
        "legacy_aggregate_mapping": {
            "values": result["aggregate"],
            "interpretation": (
                "Retained for v1 fidelity only. False-promotion and recovery aggregates mix "
                "scientifically distinct scenario classes and are not classical Type-I error "
                "or power estimands."
            ),
        },
        "estimand_classes": {
            "pure_null_broad_promotion_rate": _mean(
                select(pure_null), "observed_replication_pass_rate"
            ),
            "homogeneous_positive_broad_recovery_rate": _mean(
                select(homogeneous_positive), "observed_replication_pass_rate"
            ),
            "contextual_or_failed_primary_broad_promotion_rate": _mean(
                select(contextual_veto), "observed_replication_pass_rate"
            ),
            "conflicted_positive_broad_recovery_rate": _mean(
                select(conflicted_positive), "observed_replication_pass_rate"
            ),
            "failed_primary_role_protection_rate": _mean(
                primary_fail, "primary_role_protection_rate"
            ),
            "fragile_conflict_detection_rate": _mean(
                fragile, "fragile_claim_detection_rate"
            ),
        },
        "by_scenario": {
            scenario_id: _summarize(rows)
            for scenario_id, rows in sorted(by_scenario.items())
        },
        "by_scenario_and_participant_count": {
            f"{scenario_id}|n={participant_count}": _summarize(rows)
            for (scenario_id, participant_count), rows in sorted(by_scenario_n.items())
        },
        "by_scenario_and_cross_study_effect_scale": {
            f"{scenario_id}|cross={scale:g}": _summarize(rows)
            for (scenario_id, scale), rows in sorted(by_scenario_cross.items())
        },
        "metric_applicability": {
            "primary_role_protection_rate": (
                "Interpreted only for primary-fail-replications-positive-4; v1 stores 1.0 in "
                "non-applicable scenarios."
            ),
            "fragile_claim_detection_rate": (
                "Interpreted only for fragile-one-conflict-4 here; v1 stores 1.0 in many "
                "non-applicable scenarios."
            ),
            "cross_study_effect_scale": (
                "No global marginal is reported because the v1 axis is inert for all-null "
                "scenarios and only perturbs positive-labelled candidate effects."
            ),
            "sensitivity_warning_under_null": (
                "Direction agreement references the observed primary sign; near a true zero "
                "effect that sign is noise, so no-warning match is not well posed without "
                "magnitude gating."
            ),
        },
        "coarse_rate_resolution": {
            "step": 0.125,
            "wilson_95_for_0_of_8": _wilson(0, 8),
            "wilson_95_for_1_of_8": _wilson(1, 8),
            "wilson_95_for_8_of_8": _wilson(8, 8),
            "interpretation": (
                "Eight outer replicates per cell map gross regimes; they do not provide "
                "precision tail-probability calibration."
            ),
        },
        "worst_20_cells_by_decision_error_then_warning_mismatch": worst_cells,
        "claim_boundary": (
            "Development-only deterministic software evidence. It does not define acceptance "
            "thresholds, execute final evaluation, validate biological truth, establish universal "
            "generalization, or authorize physical-quantum promotion."
        ),
    }


def _report(analysis: dict[str, object]) -> str:
    estimands = analysis["estimand_classes"]
    assert isinstance(estimands, dict)
    lines = [
        "# BMRB study development evidence v1",
        "",
        "Complete **648-cell development-only** study operating surface recomposed from the 24 ",
        f"successful shards of Actions run `{SOURCE_WORKFLOW_RUN_ID}`.",
        "",
        "## Authority",
        "",
        f"- science source: `{analysis['science_source_sha']}`",
        f"- policy fingerprint: `{analysis['policy_fingerprint']}`",
        f"- artifact fingerprint: `{analysis['artifact_fingerprint']}`",
        "- outer replicates per cell: **8**",
        "- participant bootstrap resamples: **100**",
        "- final evaluation executed: **false**",
        "- qualification defined: **false**",
        "- physical-quantum promotion eligible: **false**",
        "",
        "## Applicability-aware descriptive estimands",
        "",
        f"- pure-null broad promotion: **{float(estimands['pure_null_broad_promotion_rate']):.6f}**",
        f"- homogeneous-positive broad recovery: **{float(estimands['homogeneous_positive_broad_recovery_rate']):.6f}**",
        f"- contextual/failed-primary broad promotion: **{float(estimands['contextual_or_failed_primary_broad_promotion_rate']):.6f}**",
        f"- conflicted-positive broad recovery: **{float(estimands['conflicted_positive_broad_recovery_rate']):.6f}**",
        f"- failed-primary role protection: **{float(estimands['failed_primary_role_protection_rate']):.6f}**",
        f"- fragile-conflict detection: **{float(estimands['fragile_conflict_detection_rate']):.6f}**",
        "",
        "These are coarse development summaries, not registered acceptance criteria. With 8 ",
        "replicates, cell-level rates move in increments of 0.125.",
        "",
        "## Critical interpretation",
        "",
        "The legacy v1 false-promotion/recovery aggregates mix distinct scenario classes. ",
        "Non-applicable protection/detection metrics are not globally averaged here. No global ",
        "`cross_study_effect_scale` marginal is reported because that axis is inert for all-null ",
        "scenarios. The null direction-agreement warning behavior is treated as a v1 diagnostic ",
        "semantics issue, not something to tune away after seeing this surface.",
        "",
        "## Claim boundary",
        "",
        "Development-only deterministic software evidence. This capsule does not define final ",
        "thresholds, execute the evaluation partition, establish biological truth, prove ",
        "population/task universality, or authorize a physical-quantum mechanism claim.",
        "",
    ]
    return "\n".join(lines)


def _provenance(result: dict[str, object]) -> dict[str, object]:
    policy = result["policy"]
    assert isinstance(policy, dict)
    return {
        "source_workflow_run_id": SOURCE_WORKFLOW_RUN_ID,
        "source_workflow_head_sha": SOURCE_WORKFLOW_HEAD_SHA,
        "science_source_sha": SCIENCE_SOURCE_SHA,
        "recovery_workflow_run_id": int(os.environ.get("GITHUB_RUN_ID", "0")),
        "recovery_workflow_commit_sha": os.environ.get("GITHUB_SHA"),
        "runner_os": os.environ.get("RUNNER_OS"),
        "runner_arch": os.environ.get("RUNNER_ARCH"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pip": subprocess.check_output(
            ["python", "-m", "pip", "--version"], text=True
        ).strip(),
        "artifact_fingerprint": result["artifact_fingerprint"],
        "policy_fingerprint": policy["policy_fingerprint"],
        "note": (
            "Execution provenance supplements, but does not alter, the qualified v1 scientific "
            "fingerprint."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    shard_payloads = _load_shards(args.shards)
    result = _recompose(shard_payloads)
    args.output.mkdir(parents=True, exist_ok=True)

    result_path = args.output / "development-result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verify_bmrb_study_operating_mapping(
        json.loads(result_path.read_text(encoding="utf-8"))
    )

    analysis = _analyze(result)
    (args.output / "development-analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "execution-provenance.json").write_text(
        json.dumps(_provenance(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = _report(analysis)
    (args.output / "README.md").write_text(report, encoding="utf-8")

    manifest = {
        path.name: _sha256(path)
        for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "sha256-manifest.json"
    }
    (args.output / "sha256-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with Path(step_summary).open("a", encoding="utf-8") as handle:
            handle.write(report)


if __name__ == "__main__":
    main()
