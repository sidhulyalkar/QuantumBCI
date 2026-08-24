#!/usr/bin/env python3
"""Compatibility wrapper for the installed ``quantumbci-kumar2024`` command."""

from quantumbci.studies.kumar2024_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
