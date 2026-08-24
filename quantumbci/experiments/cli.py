"""CLI for validating and planning QuantumBCI experiment manifests."""

from __future__ import annotations

import argparse
import json

from .manifest import load_manifest
from .orchestration import build_plan, materialize_plan, render_plan


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("manifest", help="Path to an experiment JSON manifest")
    value.add_argument("--source-sha", required=True, help="Git source revision bound to the plan")
    value.add_argument("--json", action="store_true", help="Print the full plan as JSON")
    value.add_argument("--output-dir", help="Write plan.json to this directory")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    if args.output_dir:
        path = materialize_plan(manifest, args.source_sha, args.output_dir)
        print(path)
        return 0
    if args.json:
        print(json.dumps(build_plan(manifest, args.source_sha), indent=2, sort_keys=True))
    else:
        print(render_plan(manifest, args.source_sha))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
