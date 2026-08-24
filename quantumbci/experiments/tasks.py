"""Reserved stage-task entry point for the v0.3 implementation sequence.

The experiment manifests intentionally name future task commands now so the scientific
DAG can be reviewed before dataset/model code exists. Calling a task that has not been
implemented fails explicitly instead of silently fabricating an artifact.
"""

from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("experiment")
    parser.add_argument("qualifier", nargs="?")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            {
                "status": "not_implemented",
                "task": args.task,
                "experiment": args.experiment,
                "qualifier": args.qualifier,
                "message": (
                    "This manifest stage is a reviewed orchestration contract, but its "
                    "dataset/model task has not been implemented in v0.3 planning yet."
                ),
            },
            sort_keys=True,
        )
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
