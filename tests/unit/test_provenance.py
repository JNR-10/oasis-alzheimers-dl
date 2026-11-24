"""Run provenance — ADR-0008.

A result whose configuration cannot be recovered does not go in the report. The
properties here are mostly about *when* and *how honestly* the record is written:

- Captured at run **start**, so a crashed run still leaves a record of what was
  attempted. A record written at the end only ever describes runs that succeeded,
  which are exactly the ones least in need of explaining.
- A dirty checkout is flagged. It is fine while exploring and disqualifying for a
  reported number, because the code that produced it no longer exists anywhere.
- Unknowable facts are reported as unknown. A missing git binary must not come back
  as "clean".
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from oasis_dl.utils.provenance import (
    RunProvenance,
    capture,
    git_commit,
    git_is_dirty,
    gpu_name,
    package_versions,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A real git checkout with one commit."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "tracked.txt").write_text("one\n")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-qm", "first")
    return tmp_path


class TestGitCommit:
    def test_it_returns_the_current_hash(self, repo: Path) -> None:
        assert len(git_commit(repo)) >= 7
        assert all(c in "0123456789abcdef" for c in git_commit(repo))

    def test_outside_a_checkout_it_says_unknown(self, tmp_path: Path) -> None:
        """Never a plausible-looking placeholder hash."""
        assert git_commit(tmp_path / "not-a-repo") == "unknown"


class TestGitIsDirty:
    def test_a_clean_checkout_is_clean(self, repo: Path) -> None:
        assert git_is_dirty(repo) is False

    def test_a_modified_tracked_file_is_dirty(self, repo: Path) -> None:
        (repo / "tracked.txt").write_text("two\n")
        assert git_is_dirty(repo) is True

    def test_a_staged_change_is_dirty(self, repo: Path) -> None:
        (repo / "tracked.txt").write_text("three\n")
        _git(repo, "add", "tracked.txt")
        assert git_is_dirty(repo) is True

    def test_an_unverifiable_checkout_is_reported_dirty(self, tmp_path: Path) -> None:
        """The safe answer. Reporting "clean" for a checkout we could not inspect would
        let an unreproducible run into the report wearing a clean flag."""
        assert git_is_dirty(tmp_path / "not-a-repo") is True


class TestEnvironment:
    def test_the_libraries_that_can_change_a_result_are_recorded(self) -> None:
        versions = package_versions()
        for package in ("torch", "monai", "numpy", "nibabel", "pandas"):
            assert package in versions, package

    def test_a_missing_package_is_omitted_not_guessed(self) -> None:
        """Every recorded version must be real."""
        assert all(value and value != "unknown" for value in package_versions().values())

    def test_gpu_name_is_none_on_cpu(self) -> None:
        import torch

        if not torch.cuda.is_available():
            assert gpu_name() is None


class TestCapture:
    def test_it_fills_every_field(self, tmp_path: Path, repo: Path) -> None:
        record = capture(
            run_id="r1",
            experiment="baseline_single_scan",
            seed=42,
            manifest_path=tmp_path / "manifest.parquet",
            config={"model": {"encoder": {"backbone": "vit_base_16"}}},
            repo=repo,
        )
        assert record.run_id == "r1"
        assert record.experiment == "baseline_single_scan"
        assert record.seed == 42
        assert record.git_commit
        assert record.python_version and record.platform
        assert record.packages
        assert record.config["model"]["encoder"]["backbone"] == "vit_base_16"

    def test_the_timestamp_is_utc_and_parseable(self, tmp_path: Path, repo: Path) -> None:
        record = capture("r1", "exp", 0, tmp_path / "m.parquet", repo=repo)
        parsed = datetime.fromisoformat(record.timestamp)
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0

    def test_it_warns_on_a_dirty_checkout(self, tmp_path: Path, repo: Path) -> None:
        """Flagged *and* warned. A field nobody reads is not a control."""
        (repo / "tracked.txt").write_text("changed\n")
        with pytest.warns(UserWarning, match="dirty"):
            record = capture("r1", "exp", 0, tmp_path / "m.parquet", repo=repo)
        assert record.git_dirty is True

    def test_a_clean_checkout_does_not_warn(self, tmp_path: Path, repo: Path) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert capture("r1", "exp", 0, tmp_path / "m.parquet", repo=repo).git_dirty is False


class TestWrite:
    def test_it_creates_parent_directories(self, tmp_path: Path, repo: Path) -> None:
        target = tmp_path / "deep" / "nested" / "provenance.json"
        capture("r1", "exp", 0, tmp_path / "m.parquet", repo=repo).write(target)
        assert target.exists()

    def test_the_record_round_trips_through_json(self, tmp_path: Path, repo: Path) -> None:
        original = capture(
            "r1",
            "exp",
            7,
            tmp_path / "m.parquet",
            config={"training": {"max_epochs": 60}},
            repo=repo,
        )
        target = tmp_path / "provenance.json"
        original.write(target)
        loaded = json.loads(target.read_text())
        assert loaded["run_id"] == "r1"
        assert loaded["seed"] == 7
        assert loaded["config"]["training"]["max_epochs"] == 60
        assert RunProvenance(**loaded) == original

    def test_the_manifest_path_is_recorded_as_text(self, tmp_path: Path, repo: Path) -> None:
        """ADR-0008: which cohort this trained on. A Path is not JSON."""
        record = capture("r1", "exp", 0, tmp_path / "cohort.parquet", repo=repo)
        record.write(tmp_path / "p.json")
        assert "cohort.parquet" in json.loads((tmp_path / "p.json").read_text())["manifest_path"]
