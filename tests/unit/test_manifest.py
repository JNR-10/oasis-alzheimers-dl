"""Tests for the cohort manifest I/O contract (ADR-0008 / issue #14)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from oasis_dl.data.manifest import COLUMNS, read_manifest, write_manifest


def _frame(**overrides: object) -> pd.DataFrame:
    row = {
        "subject_id": "S1",
        "session_id": "S1_d0000",
        "partition": "train",
        "days_from_entry": 0,
        "scan_path": "/nonexistent/S1.nii.gz",
        "label": 0,
        "is_target_visit": True,
        "exclusion_reason": "",
    }
    row.update(overrides)
    return pd.DataFrame([row])


class TestWriteManifest:
    def test_writes_csv_with_schema_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "manifests" / "cohort.csv"
        write_manifest(_frame(), path)
        loaded = pd.read_csv(path)
        assert list(loaded.columns) == list(COLUMNS)

    def test_refuses_to_overwrite(self, tmp_path: Path) -> None:
        path = tmp_path / "cohort.csv"
        write_manifest(_frame(), path)
        with pytest.raises(FileExistsError, match="overwrite"):
            write_manifest(_frame(subject_id="S2"), path)
        assert pd.read_csv(path).iloc[0]["subject_id"] == "S1"

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        frame = _frame().drop(columns=["exclusion_reason"])
        with pytest.raises(ValueError, match="exclusion_reason"):
            write_manifest(frame, tmp_path / "cohort.csv")

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "cohort.csv"
        write_manifest(_frame(), path)
        assert path.exists()


class TestReadManifest:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "cohort.csv"
        original = _frame(
            exclusion_reason="sampled_out",
            is_target_visit=False,
            label="",
        )
        write_manifest(original, path)
        loaded = read_manifest(path)
        assert list(loaded.columns) == list(COLUMNS)
        assert loaded.iloc[0]["subject_id"] == "S1"
        assert loaded.iloc[0]["exclusion_reason"] == "sampled_out"

    def test_absent_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_manifest(tmp_path / "missing.csv")

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.csv"
        pd.DataFrame({"subject_id": ["S1"]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="missing columns"):
            read_manifest(path)
