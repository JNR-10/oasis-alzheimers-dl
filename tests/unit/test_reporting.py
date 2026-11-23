"""Results tables generated from run records — #12, ADR-0008.

``AGENTS.md``: results tables are generated from recorded runs, never typed. A
hand-copied number is one that can silently stop matching the run that produced it, and
in a report whose whole claim is methodological care that is the worst available failure.

Three properties here are about refusing to flatter:

- A **crashed run** is announced, not silently skipped. A table that quietly drops the
  runs that failed describes a different experiment from the one that was run.
- A **dirty** or **mixed-commit** group is flagged, because the code that produced those
  numbers cannot be recovered.
- The **interval kind is named**. Across-seed spread and a bootstrap over one fitted
  model measure different things, and a column that silently mixed them would invite
  exactly the comparison the protocol forbids.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oasis_dl.evaluation.bootstrap import ConfidenceInterval
from oasis_dl.evaluation.metrics import compute_metrics
from oasis_dl.evaluation.reporting import (
    collect_runs,
    results_table,
    write_results_table,
    write_run_results,
)


def _write_run(
    root: Path,
    *,
    arm: str,
    seed: int,
    roc_auc: float,
    commit: str = "a" * 40,
    dirty: bool = False,
    results: bool = True,
    interval: tuple[float, float] = (0.6, 0.9),
) -> Path:
    run_dir = root / f"{arm}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "provenance.json").write_text(
        json.dumps(
            {
                "run_id": f"{arm}-{seed}",
                "experiment": arm,
                "git_commit": commit,
                "git_dirty": dirty,
                "seed": seed,
                "timestamp": "2026-11-23T00:00:00+00:00",
                "manifest_path": "cohort.parquet",
                "python_version": "3.11.0",
                "platform": "test",
                "packages": {"torch": "2.2.2"},
                "gpu": None,
                "config": {},
            }
        )
    )
    if results:
        (run_dir / "results.json").write_text(
            json.dumps(
                {
                    "arm": arm,
                    "partition": "test",
                    "n_subjects": 60,
                    "n_visits": 142,
                    "metrics": {"roc_auc": roc_auc, "balanced_accuracy": 0.7},
                    "intervals": {
                        "roc_auc": {
                            "point": roc_auc,
                            "lower": interval[0],
                            "upper": interval[1],
                            "level": 0.95,
                            "n_resamples": 1000,
                        }
                    },
                }
            )
        )
    return run_dir


class TestCollectRuns:
    def test_it_finds_every_complete_run(self, tmp_path: Path) -> None:
        _write_run(tmp_path, arm="B1", seed=1, roc_auc=0.70)
        _write_run(tmp_path, arm="B1", seed=2, roc_auc=0.74)
        assert len(collect_runs(tmp_path)) == 2

    def test_a_crashed_run_is_announced_not_silently_dropped(self, tmp_path: Path) -> None:
        """Provenance written at start, no results at the end: the run died. A table
        that omits it without saying so describes a different experiment."""
        _write_run(tmp_path, arm="B1", seed=1, roc_auc=0.70)
        _write_run(tmp_path, arm="B1", seed=2, roc_auc=0.0, results=False)
        with pytest.warns(UserWarning, match="without results"):
            records = collect_runs(tmp_path)
        assert len(records) == 1

    def test_an_empty_root_yields_nothing_rather_than_erroring(self, tmp_path: Path) -> None:
        assert collect_runs(tmp_path) == []

    def test_runs_are_returned_in_a_stable_order(self, tmp_path: Path) -> None:
        for seed in (3, 1, 2):
            _write_run(tmp_path, arm="B1", seed=seed, roc_auc=0.7)
        assert [r.seed for r in collect_runs(tmp_path)] == [1, 2, 3]


class TestResultsTable:
    def test_one_row_per_arm_and_metric(self, tmp_path: Path) -> None:
        _write_run(tmp_path, arm="B1", seed=1, roc_auc=0.70)
        _write_run(tmp_path, arm="PA", seed=1, roc_auc=0.76)
        table = results_table(collect_runs(tmp_path))
        assert set(table["arm"]) == {"B1", "PA"}
        assert set(table["metric"]) >= {"roc_auc", "balanced_accuracy"}

    def test_every_row_carries_what_the_ticket_requires(self, tmp_path: Path) -> None:
        """Cohort size, metric with interval, seed count and commit."""
        _write_run(tmp_path, arm="B1", seed=1, roc_auc=0.70)
        table = results_table(collect_runs(tmp_path))
        for column in (
            "arm",
            "metric",
            "point",
            "interval_low",
            "interval_high",
            "interval_kind",
            "n_seeds",
            "n_subjects",
            "n_visits",
            "commit",
            "dirty",
        ):
            assert column in table.columns, column

    def test_a_single_seed_reports_its_bootstrap_interval(self, tmp_path: Path) -> None:
        _write_run(tmp_path, arm="B1", seed=1, roc_auc=0.70, interval=(0.61, 0.79))
        row = results_table(collect_runs(tmp_path)).query("metric == 'roc_auc'").iloc[0]
        assert row["n_seeds"] == 1
        assert row["interval_kind"] == "bootstrap"
        assert (row["interval_low"], row["interval_high"]) == (0.61, 0.79)

    def test_several_seeds_report_the_across_seed_spread(self, tmp_path: Path) -> None:
        """Across-seed spread captures training variance that a bootstrap over one
        fitted model cannot, so it is the interval that calibrates a between-arm gap."""
        for seed, auc in ((1, 0.70), (2, 0.74), (3, 0.78)):
            _write_run(tmp_path, arm="B1", seed=seed, roc_auc=auc)
        row = results_table(collect_runs(tmp_path)).query("metric == 'roc_auc'").iloc[0]
        assert row["n_seeds"] == 3
        assert row["interval_kind"] == "across-seed"
        assert row["point"] == pytest.approx(0.74)
        assert (row["interval_low"], row["interval_high"]) == pytest.approx((0.70, 0.78))

    def test_the_interval_kind_is_never_left_implicit(self, tmp_path: Path) -> None:
        """The two kinds measure different things; a column that mixed them silently
        would invite exactly the comparison the protocol forbids."""
        _write_run(tmp_path, arm="B1", seed=1, roc_auc=0.70)
        for seed, auc in ((1, 0.70), (2, 0.80)):
            _write_run(tmp_path, arm="PA", seed=seed, roc_auc=auc)
        table = results_table(collect_runs(tmp_path)).query("metric == 'roc_auc'")
        kinds = dict(zip(table["arm"], table["interval_kind"], strict=True))
        assert kinds["B1"] == "bootstrap" and kinds["PA"] == "across-seed"

    def test_a_dirty_run_marks_its_whole_row(self, tmp_path: Path) -> None:
        """One unreproducible seed disqualifies the aggregate, not just itself."""
        _write_run(tmp_path, arm="B1", seed=1, roc_auc=0.70, dirty=False)
        _write_run(tmp_path, arm="B1", seed=2, roc_auc=0.74, dirty=True)
        assert bool(results_table(collect_runs(tmp_path)).iloc[0]["dirty"]) is True

    def test_seeds_run_at_different_commits_are_flagged(self, tmp_path: Path) -> None:
        """Averaging across code versions is averaging across experiments."""
        _write_run(tmp_path, arm="B1", seed=1, roc_auc=0.70, commit="a" * 40)
        _write_run(tmp_path, arm="B1", seed=2, roc_auc=0.74, commit="b" * 40)
        assert "MIXED" in results_table(collect_runs(tmp_path)).iloc[0]["commit"]

    def test_one_commit_is_reported_short_and_plain(self, tmp_path: Path) -> None:
        _write_run(tmp_path, arm="B1", seed=1, roc_auc=0.70, commit="abc1234" + "0" * 33)
        assert results_table(collect_runs(tmp_path)).iloc[0]["commit"] == "abc1234"

    def test_arms_evaluated_on_different_cohorts_are_refused(self, tmp_path: Path) -> None:
        """Comparing arms measured on different numbers of subjects is not an ablation."""
        _write_run(tmp_path, arm="B1", seed=1, roc_auc=0.70)
        odd = _write_run(tmp_path, arm="B1", seed=2, roc_auc=0.74)
        payload = json.loads((odd / "results.json").read_text())
        payload["n_subjects"] = 55
        (odd / "results.json").write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="cohort"):
            results_table(collect_runs(tmp_path))

    def test_no_records_is_an_error_not_an_empty_table(self, tmp_path: Path) -> None:
        """An empty table reads as "we measured nothing and it was fine"."""
        with pytest.raises(ValueError):
            results_table([])


class TestWriteResultsTable:
    def test_it_writes_csv_and_markdown(self, tmp_path: Path) -> None:
        _write_run(tmp_path / "runs", arm="B1", seed=1, roc_auc=0.70)
        written = write_results_table(
            results_table(collect_runs(tmp_path / "runs")), tmp_path / "out" / "results"
        )
        assert {p.suffix for p in written} == {".csv", ".md"}
        assert all(p.exists() for p in written)

    def test_the_markdown_names_the_commit_and_seed_count(self, tmp_path: Path) -> None:
        """So a table pasted into the report carries its own provenance."""
        _write_run(tmp_path / "runs", arm="B1", seed=1, roc_auc=0.70, commit="abc1234" + "0" * 33)
        written = write_results_table(
            results_table(collect_runs(tmp_path / "runs")), tmp_path / "out" / "results"
        )
        text = next(p for p in written if p.suffix == ".md").read_text()
        assert "abc1234" in text and "n_seeds" in text

    def test_a_dirty_table_says_so_where_a_reader_will_see_it(self, tmp_path: Path) -> None:
        _write_run(tmp_path / "runs", arm="B1", seed=1, roc_auc=0.70, dirty=True)
        written = write_results_table(
            results_table(collect_runs(tmp_path / "runs")), tmp_path / "out" / "results"
        )
        text = next(p for p in written if p.suffix == ".md").read_text()
        assert "dirty checkout" in text.lower()

    def test_a_clean_table_carries_no_such_warning(self, tmp_path: Path) -> None:
        """So the test above cannot pass on a note printed unconditionally."""
        _write_run(tmp_path / "runs", arm="B1", seed=1, roc_auc=0.70, dirty=False)
        written = write_results_table(
            results_table(collect_runs(tmp_path / "runs")), tmp_path / "out" / "results"
        )
        text = next(p for p in written if p.suffix == ".md").read_text()
        assert "dirty checkout" not in text.lower()


class TestWriteRunResults:
    """The writer and the reader are one pair, so a number in the report has exactly
    one origin."""

    def test_what_is_written_is_what_collect_runs_reads_back(self, tmp_path: Path) -> None:
        run_dir = _write_run(tmp_path, arm="B1", seed=1, roc_auc=0.0, results=False).parent
        run_dir = tmp_path / "B1_seed1"
        metrics = compute_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], threshold=0.5)
        write_run_results(
            run_dir,
            arm="B1",
            partition="test",
            metrics=metrics,
            n_subjects=60,
            n_visits=142,
            intervals={
                "roc_auc": ConfidenceInterval(
                    point=0.75, lower=0.61, upper=0.88, level=0.95, n_resamples=1000
                )
            },
        )
        record = collect_runs(tmp_path)[0]
        assert record.arm == "B1"
        assert record.n_subjects == 60 and record.n_visits == 142
        assert record.metrics["roc_auc"] == pytest.approx(0.75)
        assert record.intervals["roc_auc"]["lower"] == pytest.approx(0.61)

    def test_a_run_without_intervals_still_round_trips(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "B1_seed1"
        _write_run(tmp_path, arm="B1", seed=1, roc_auc=0.0, results=False)
        write_run_results(
            run_dir,
            arm="B1",
            partition="test",
            metrics=compute_metrics([0, 1], [0.2, 0.9], threshold=0.5),
            n_subjects=2,
            n_visits=2,
        )
        assert collect_runs(tmp_path)[0].intervals == {}
