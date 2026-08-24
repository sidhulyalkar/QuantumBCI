from __future__ import annotations

from pathlib import Path

import pytest

from quantumbci.studies.kumar2024 import Kumar2024StudyConfig, fingerprint_raw_dataset


class _Dataset:
    def __init__(self, roots: dict[int, Path]) -> None:
        self.roots = roots

    def data_path(self, subject: int):
        return [self.roots[int(subject)]]


def _source_tree(root: Path, subjects: tuple[int, ...]) -> dict[int, Path]:
    roots: dict[int, Path] = {}
    for subject in subjects:
        directory = root / f"subject-{subject}"
        directory.mkdir(parents=True)
        (directory / "session-a.txt").write_text(
            f"subject={subject}\nsession=a\nvalue=123\n",
            encoding="utf-8",
        )
        nested = directory / "nested"
        nested.mkdir()
        (nested / "session-b.txt").write_text(
            f"subject={subject}\nsession=b\nvalue=907\n",
            encoding="utf-8",
        )
        roots[subject] = directory
    return roots


def test_raw_dataset_fingerprint_is_content_addressed_not_absolute_path_addressed(tmp_path: Path) -> None:
    subjects = (1, 10)
    first_roots = _source_tree(tmp_path / "machine-a", subjects)
    second_roots = _source_tree(tmp_path / "another" / "machine-b", subjects)

    first = fingerprint_raw_dataset(_Dataset(first_roots), subjects)
    second = fingerprint_raw_dataset(_Dataset(second_roots), subjects)

    assert first["fingerprint"] == second["fingerprint"]
    assert first["by_subject"]["1"]["fingerprint"] == second["by_subject"]["1"]["fingerprint"]
    assert all("machine-a" not in item["name"] for item in first["by_subject"]["1"]["files"])


def test_raw_dataset_fingerprint_changes_when_source_content_changes(tmp_path: Path) -> None:
    subjects = (1, 10)
    roots = _source_tree(tmp_path / "source", subjects)
    dataset = _Dataset(roots)
    before = fingerprint_raw_dataset(dataset, subjects)
    (roots[10] / "session-a.txt").write_text("changed source content\n", encoding="utf-8")
    after = fingerprint_raw_dataset(dataset, subjects)
    assert before["fingerprint"] != after["fingerprint"]
    assert before["by_subject"]["1"]["fingerprint"] == after["by_subject"]["1"]["fingerprint"]
    assert before["by_subject"]["10"]["fingerprint"] != after["by_subject"]["10"]["fingerprint"]


def test_kumar_config_canonicalizes_frontier_and_rejects_invalid_study() -> None:
    config = Kumar2024StudyConfig(
        subjects=(10, 1, 10),
        held_out_sessions=("5", "5"),
        budgets_per_class=(5, 0, 2, 2),
    )
    assert config.subjects == (1, 10)
    assert config.held_out_sessions == ("5",)
    assert config.budgets_per_class == (0, 2, 5)

    with pytest.raises(ValueError, match="at least two subjects"):
        Kumar2024StudyConfig(subjects=(1,))
    with pytest.raises(ValueError, match="session 0"):
        Kumar2024StudyConfig(subjects=(1, 10), held_out_sessions=("0",))
    with pytest.raises(ValueError, match="1..18"):
        Kumar2024StudyConfig(subjects=(1, 19))
