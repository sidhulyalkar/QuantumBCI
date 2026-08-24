from __future__ import annotations

from pathlib import Path

import pytest

from quantumbci.studies.kumar2024 import Kumar2024StudyConfig, fingerprint_raw_dataset


class _Dataset:
    def __init__(self, root: Path) -> None:
        self.root = root

    def data_path(self, subject: int):
        # Mirrors current MOABB Kumar2024: every subject resolves to one shared
        # extracted dataset root; subject-specific selection happens afterward.
        assert 1 <= int(subject) <= 18
        return [self.root]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _kumar_tree(root: Path) -> Path:
    # MOABB subjects 1..9 map directly to raw GR participants. MOABB subject 10
    # maps to raw subject 11 in the PAR cohort because the upstream release skips
    # one participant in the public bar-feedback subset.
    for group, raw_subject in (("GR", 1), ("PAR", 11)):
        suffix = f"{raw_subject:02d}"
        for session in range(1, 3):
            _write(
                root
                / "Offline"
                / group
                / f"Subject_{suffix}_Offline"
                / f"subject-{raw_subject}-offline-{session}.gdf",
                f"raw_subject={raw_subject}\nkind=offline\nsession={session}\n",
            )
        for session in range(1, 6):
            _write(
                root
                / "Online"
                / group
                / f"Subject_{suffix}_Online"
                / f"subject-{raw_subject}-online-{session}.gdf",
                f"raw_subject={raw_subject}\nkind=online\nsession={session}\n",
            )

    # Present in the public archive, but not consumed by the MOABB bar-feedback
    # loader used for this study. Changes here must not alter scientific identity.
    _write(root / "Race" / "Subject_01" / "unused-racing.gdf", "racing-v1\n")
    _write(root / "README.txt", "archive metadata that is not an EEG epoch source\n")
    return root


def test_raw_dataset_fingerprint_is_content_addressed_not_absolute_path_addressed(
    tmp_path: Path,
) -> None:
    subjects = (1, 10)
    first_root = _kumar_tree(tmp_path / "machine-a" / "Kumar2024")
    second_root = _kumar_tree(tmp_path / "another" / "machine-b" / "Kumar2024")

    first = fingerprint_raw_dataset(_Dataset(first_root), subjects)
    second = fingerprint_raw_dataset(_Dataset(second_root), subjects)

    assert first["schema_version"] == 2
    assert first["fingerprint"] == second["fingerprint"]
    assert first["by_subject"]["1"]["fingerprint"] == second["by_subject"]["1"]["fingerprint"]
    assert first["by_subject"]["10"]["fingerprint"] == second["by_subject"]["10"]["fingerprint"]
    assert first["selection"]["moabb_subject_to_raw_subject"] == {"1": 1, "10": 11}
    assert first["selection"]["exclude"] == ["Race/**"]
    assert all(not item["name"].startswith("Race/") for item in first["files"])
    assert all("machine-a" not in item["name"] for item in first["files"])


def test_raw_dataset_fingerprint_changes_only_for_selected_subject_content(
    tmp_path: Path,
) -> None:
    subjects = (1, 10)
    root = _kumar_tree(tmp_path / "source")
    dataset = _Dataset(root)
    before = fingerprint_raw_dataset(dataset, subjects)

    selected = (
        root
        / "Online"
        / "PAR"
        / "Subject_11_Online"
        / "subject-11-online-5.gdf"
    )
    selected.write_text("changed selected EEG bytes\n", encoding="utf-8")
    after = fingerprint_raw_dataset(dataset, subjects)

    assert before["fingerprint"] != after["fingerprint"]
    assert before["by_subject"]["1"]["fingerprint"] == after["by_subject"]["1"]["fingerprint"]
    assert before["by_subject"]["10"]["fingerprint"] != after["by_subject"]["10"]["fingerprint"]


def test_racing_game_files_do_not_change_bar_feedback_fingerprint(tmp_path: Path) -> None:
    root = _kumar_tree(tmp_path / "source")
    dataset = _Dataset(root)
    before = fingerprint_raw_dataset(dataset, (1, 10))

    (root / "Race" / "Subject_01" / "unused-racing.gdf").write_text(
        "completely different racing bytes\n",
        encoding="utf-8",
    )
    after = fingerprint_raw_dataset(dataset, (1, 10))

    assert before["fingerprint"] == after["fingerprint"]
    assert before["by_subject"] == after["by_subject"]


def test_raw_dataset_fingerprint_fails_if_selected_subject_files_are_missing(
    tmp_path: Path,
) -> None:
    root = _kumar_tree(tmp_path / "source")
    missing = root / "Online" / "PAR" / "Subject_11_Online"
    for path in missing.glob("*.gdf"):
        path.unlink()
    for path in (root / "Offline" / "PAR" / "Subject_11_Offline").glob("*.gdf"):
        path.unlink()

    with pytest.raises(FileNotFoundError, match="MOABB subject 10"):
        fingerprint_raw_dataset(_Dataset(root), (1, 10))


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
