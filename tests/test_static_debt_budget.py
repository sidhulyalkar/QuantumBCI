"""Keep intentionally tolerated static debt bounded and reviewable.

This is a ratchet, not a claim that the existing codebase is perfectly typed. New broad
exception handling or new files containing type suppressions must be justified explicitly
rather than entering the scientific codebase invisibly.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "quantumbci"

# The optional neurOS registry is an external integration boundary. neurOS owns a richer
# adapter-error taxonomy than QuantumBCI can depend on without importing the optional
# runtime, so one translation catch remains deliberate here.
ALLOWED_EXCEPTION_CATCH_COUNTS = {
    "quantumbci/integrations/neuros.py": 1,
}

# These files currently carry narrowly coded typing suppressions around trajectory-role /
# NumPy typing seams. The ratchet prevents the debt from spreading to new modules. Each
# suppression must remain error-code-qualified rather than a blanket ``type: ignore``.
ALLOWED_TYPE_IGNORE_FILES = {
    "quantumbci/classical_dynamics.py",
    "quantumbci/dynamics_fitting.py",
    "quantumbci/nonlinear_dynamics.py",
    "quantumbci/probabilistic_ssm.py",
    "quantumbci/switching_dynamics.py",
    "quantumbci/trajectory_authority.py",
}

QUALIFIED_TYPE_IGNORE = re.compile(r"#\s*type:\s*ignore\[[^\]]+\]")


def _python_files() -> tuple[Path, ...]:
    return tuple(sorted(PACKAGE_ROOT.rglob("*.py")))


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=_relative(path))


def test_no_bare_except_or_baseexception_catches_in_production() -> None:
    failures: list[str] = []
    for path in _python_files():
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.ExceptHandler):
                continue
            if node.type is None:
                failures.append(f"{_relative(path)}:{node.lineno}: bare except")
            elif isinstance(node.type, ast.Name) and node.type.id == "BaseException":
                failures.append(f"{_relative(path)}:{node.lineno}: catches BaseException")
    assert not failures, "\n".join(failures)


def test_broad_exception_catches_match_explicit_budget() -> None:
    observed: dict[str, int] = {}
    for path in _python_files():
        count = 0
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.ExceptHandler):
                continue
            if isinstance(node.type, ast.Name) and node.type.id == "Exception":
                count += 1
        if count:
            observed[_relative(path)] = count
    assert observed == ALLOWED_EXCEPTION_CATCH_COUNTS


def test_type_suppression_debt_cannot_spread_to_new_modules() -> None:
    observed_files: set[str] = set()
    unqualified: list[str] = []
    for path in _python_files():
        relative = _relative(path)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "type: ignore" not in line:
                continue
            observed_files.add(relative)
            if QUALIFIED_TYPE_IGNORE.search(line) is None:
                unqualified.append(f"{relative}:{line_number}: {line.strip()}")

    unexpected = sorted(observed_files - ALLOWED_TYPE_IGNORE_FILES)
    assert not unexpected, f"new modules contain type suppressions: {unexpected}"
    assert not unqualified, "unqualified type suppressions:\n" + "\n".join(unqualified)


def test_static_debt_allowlists_do_not_accumulate_stale_entries() -> None:
    """Removing the last suppression/catch from a file should shrink the allowlist too."""

    exception_files: set[str] = set()
    ignore_files: set[str] = set()
    for path in _python_files():
        relative = _relative(path)
        tree = _tree(path)
        if any(
            isinstance(node, ast.ExceptHandler)
            and isinstance(node.type, ast.Name)
            and node.type.id == "Exception"
            for node in ast.walk(tree)
        ):
            exception_files.add(relative)
        if "type: ignore" in path.read_text(encoding="utf-8"):
            ignore_files.add(relative)

    assert exception_files == set(ALLOWED_EXCEPTION_CATCH_COUNTS)
    assert ignore_files == ALLOWED_TYPE_IGNORE_FILES
