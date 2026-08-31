"""CLI for resumable BMRB study-level development operating runs.

This command intentionally has no evaluation-partition option. Partial shard files are
execution artifacts only; only a complete verified merge yields the ordinary study-level
operating result schema.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .bmrb_study_operating import (
    BMRBStudyOperatingPolicy,
    qualification_smoke_grid,
    recommended_development_grid,
)
from .bmrb_study_operating_artifacts import verify_bmrb_study_operating_mapping
from .bmrb_study_operating_shards import (
    load_bmrb_study_operating_shard,
    merge_bmrb_study_operating_shards,
    plan_bmrb_study_operating_shards,
    run_bmrb_study_operating_shard,
    write_bmrb_study_operating_shard,
)


def _add_policy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument(
        "--grid",
        choices=("recommended", "smoke"),
        default="recommended",
        help="Development grid only. The evaluation partition is not exposed.",
    )
    parser.add_argument("--replicates-per-cell", type=int, default=8)
    parser.add_argument("--bootstrap-resamples", type=int, default=100)


def _policy_from_args(args: argparse.Namespace) -> BMRBStudyOperatingPolicy:
    grid = (
        recommended_development_grid()
        if args.grid == "recommended"
        else qualification_smoke_grid()
    )
    return BMRBStudyOperatingPolicy(
        study_id=args.study_id,
        source_sha=args.source_sha,
        partition="development",
        grid=grid,
        replicates_per_cell=args.replicates_per_cell,
        bootstrap_resamples=args.bootstrap_resamples,
    )


def _write_json(path: str | Path, payload: object) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run resumable development shards for BMRB study operating curves."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Create a deterministic complete shard plan.")
    _add_policy_arguments(plan)
    plan.add_argument("--cells-per-shard", type=int, required=True)
    plan.add_argument("--output", required=True)

    run = subparsers.add_parser("run", help="Run one deterministic development shard.")
    _add_policy_arguments(run)
    run.add_argument("--start-cell", type=int, required=True)
    run.add_argument("--stop-cell", type=int, required=True)
    run.add_argument("--output", required=True)

    merge = subparsers.add_parser(
        "merge",
        help="Merge an exact complete shard family into one verified operating result.",
    )
    _add_policy_arguments(merge)
    merge.add_argument("--output", required=True)
    merge.add_argument("shards", nargs="+")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    policy = _policy_from_args(args)

    if args.command == "plan":
        plan = plan_bmrb_study_operating_shards(
            policy,
            cells_per_shard=args.cells_per_shard,
        )
        _write_json(args.output, plan.to_mapping())
        return 0

    if args.command == "run":
        shard = run_bmrb_study_operating_shard(
            policy,
            start_cell=args.start_cell,
            stop_cell=args.stop_cell,
        )
        write_bmrb_study_operating_shard(args.output, shard)
        return 0

    if args.command == "merge":
        shards = tuple(
            load_bmrb_study_operating_shard(path, policy=policy) for path in args.shards
        )
        result = merge_bmrb_study_operating_shards(policy, shards)
        payload = result.to_mapping()
        verify_bmrb_study_operating_mapping(payload)
        _write_json(args.output, payload)
        return 0

    raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
