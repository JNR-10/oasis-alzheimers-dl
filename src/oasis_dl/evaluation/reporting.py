"""Results tables, generated from recorded runs.

``AGENTS.md``: results tables are generated from recorded runs, never typed. A
hand-copied number is one that can silently stop matching the run that produced it,
and in a report whose whole claim is methodological care that is the worst available
failure. So this module is the only thing that produces a reportable table, and it
reads nothing but what runs wrote (ADR-0008).

It is deliberately unflattering. A crashed run is announced rather than dropped, a
dirty checkout marks its whole row, seeds run at different commits are flagged, and the
kind of interval is always named — across-seed spread and a bootstrap over one fitted
model measure different things.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import pandas as pd

from oasis_dl.evaluation.bootstrap import ConfidenceInterval
from oasis_dl.evaluation.metrics import ClassificationMetrics
from oasis_dl.utils.provenance import PROVENANCE_FILENAME

__all__ = [
    "RunRecord",
    "collect_runs",
    "results_table",
    "write_results_table",
    "write_run_results",
]

RESULTS_FILENAME = "results.json"

TABLE_COLUMNS = [
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
]


def write_run_results(
    run_dir: Path,
    *,
    arm: str,
    partition: str,
    metrics: ClassificationMetrics,
    n_subjects: int,
    n_visits: int,
    intervals: Mapping[str, ConfidenceInterval] | None = None,
) -> Path:
    """Record one run's measured result beside its provenance.

    The counterpart of :func:`collect_runs`: this is the only thing that writes what
    the table later reads, so a number in the report has exactly one origin.

    Returns:
        The path written.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "arm": arm,
        "partition": partition,
        "n_subjects": n_subjects,
        "n_visits": n_visits,
        "metrics": metrics.as_dict(),
        "intervals": {
            name: {
                "point": interval.point,
                "lower": interval.lower,
                "upper": interval.upper,
                "level": interval.level,
                "n_resamples": interval.n_resamples,
            }
            for name, interval in (intervals or {}).items()
        },
    }
    path = run_dir / RESULTS_FILENAME
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


@dataclass(frozen=True)
class RunRecord:
    """One completed run: what produced it, and what it measured.

    Attributes:
        run_dir: Where it was recorded, so a number in a table can be traced back.
        arm: The experiment name from the provenance record.
        seed: The run's seed.
        commit: Full hash, or ``"unknown"``.
        dirty: Whether the checkout had uncommitted changes.
        n_subjects, n_visits: Size of the partition the metrics were computed on.
        metrics: Point estimates, keyed by metric name.
        intervals: Bootstrap intervals, keyed by metric name. Not every metric has one.
    """

    run_dir: Path
    arm: str
    seed: int
    commit: str
    dirty: bool
    n_subjects: int
    n_visits: int
    metrics: dict[str, float]
    intervals: dict[str, dict[str, float]]


def collect_runs(root: Path) -> list[RunRecord]:
    """Load every completed run beneath ``root``.

    A directory with provenance but no results is a run that **died** — provenance is
    written at the start precisely so that is visible. Those are warned about rather
    than skipped silently, because a table that quietly omits the runs that failed
    describes a different experiment from the one that was carried out.

    Returns:
        Records sorted by arm then seed, so a regenerated table diffs cleanly.
    """
    records: list[RunRecord] = []
    incomplete: list[Path] = []

    for provenance_path in sorted(Path(root).rglob(PROVENANCE_FILENAME)):
        run_dir = provenance_path.parent
        results_path = run_dir / RESULTS_FILENAME
        if not results_path.exists():
            incomplete.append(run_dir)
            continue
        provenance = json.loads(provenance_path.read_text())
        results = json.loads(results_path.read_text())
        records.append(
            RunRecord(
                run_dir=run_dir,
                arm=str(results.get("arm", provenance["experiment"])),
                seed=int(provenance["seed"]),
                commit=str(provenance["git_commit"]),
                dirty=bool(provenance["git_dirty"]),
                n_subjects=int(results["n_subjects"]),
                n_visits=int(results["n_visits"]),
                metrics={str(k): float(v) for k, v in results["metrics"].items()},
                intervals={str(k): dict(v) for k, v in results.get("intervals", {}).items()},
            )
        )

    if incomplete:
        warnings.warn(
            f"{len(incomplete)} run(s) recorded provenance without results, so they "
            f"started and did not finish: "
            f"{', '.join(str(path.name) for path in incomplete[:5])}"
            f"{' ...' if len(incomplete) > 5 else ''}. They are absent from the table.",
            stacklevel=2,
        )
    return sorted(records, key=lambda record: (record.arm, record.seed))


def results_table(records: list[RunRecord]) -> pd.DataFrame:
    """Aggregate runs into one row per arm and metric.

    Raises:
        ValueError: If there are no records, or the arms were evaluated on different
            cohort sizes — comparing arms measured on different numbers of subjects is
            not an ablation, and averaging it would hide that.
    """
    if not records:
        raise ValueError(
            "no run records to tabulate. An empty table reads as 'we measured nothing "
            "and it was fine'."
        )

    sizes = {(record.n_subjects, record.n_visits) for record in records}
    if len(sizes) > 1:
        raise ValueError(
            f"runs report different cohort sizes {sorted(sizes)}. Arms measured on "
            f"different partitions cannot be compared, and an aggregate would conceal it."
        )

    rows: list[dict[str, Any]] = []
    for arm in sorted({record.arm for record in records}):
        group = [record for record in records if record.arm == arm]
        metrics = sorted({name for record in group for name in record.metrics})
        for metric in metrics:
            rows.append(_row(arm, metric, group))
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def _row(arm: str, metric: str, group: list[RunRecord]) -> dict[str, Any]:
    """One arm, one metric, aggregated across whatever seeds exist."""
    points = [record.metrics[metric] for record in group if metric in record.metrics]
    seeds = sorted({record.seed for record in group})

    if len(points) > 1:
        # Across-seed spread captures training variance, which a bootstrap over a
        # single fitted model cannot. It is the interval that calibrates whether a
        # between-arm gap is real.
        point, low, high, kind = fmean(points), min(points), max(points), "across-seed"
    else:
        point = points[0]
        interval = next(
            (record.intervals[metric] for record in group if metric in record.intervals), None
        )
        if interval is None:
            # No interval was recorded for this metric. Reported as absent rather than
            # as a zero-width interval, which would read as perfect precision.
            low, high, kind = float("nan"), float("nan"), "none"
        else:
            low, high, kind = float(interval["lower"]), float(interval["upper"]), "bootstrap"

    commits = sorted({record.commit for record in group})
    return {
        "arm": arm,
        "metric": metric,
        "point": point,
        "interval_low": low,
        "interval_high": high,
        "interval_kind": kind,
        "n_seeds": len(seeds),
        "n_subjects": group[0].n_subjects,
        "n_visits": group[0].n_visits,
        "commit": _describe_commits(commits),
        "dirty": any(record.dirty for record in group),
    }


def _describe_commits(commits: list[str]) -> str:
    """One short hash, or a loud marker when the seeds disagree.

    Averaging across code versions is averaging across experiments, so this is never
    quietly collapsed to the first one.
    """
    if len(commits) == 1:
        return commits[0][:7]
    return f"MIXED({', '.join(commit[:7] for commit in commits)})"


def _markdown(table: pd.DataFrame) -> str:
    """Render the table as GitHub-flavoured Markdown.

    Hand-rolled rather than ``DataFrame.to_markdown``, which needs ``tabulate``. A
    dependency added for one call is a version to keep in step for the life of the
    project, and this formatter is six lines.
    """

    def cell(value: Any) -> str:
        if isinstance(value, float):
            return "—" if pd.isna(value) else f"{value:.4f}"
        return str(value)

    header = list(table.columns)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in table.itertuples(index=False):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def write_results_table(table: pd.DataFrame, stem: Path) -> list[Path]:
    """Write the table as CSV and Markdown beside each other.

    CSV for anything downstream, Markdown so a table pasted into the report carries its
    own provenance — commit, seed count and cohort size travel with the numbers rather
    than living in a caption someone has to keep in step.

    Returns:
        The paths written, in a stable order.
    """
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = stem.with_suffix(".csv")
    markdown_path = stem.with_suffix(".md")

    table.to_csv(csv_path, index=False)

    lines = [
        "<!-- Generated by `make report` from recorded runs. Do not edit by hand: "
        "AGENTS.md forbids a typed results table. -->",
        "",
        _markdown(table),
        "",
    ]
    if bool(table["dirty"].any()):
        lines += [
            "> **At least one run came from a dirty checkout.** The code that produced "
            "those numbers does not exist in any commit, so they are not reproducible "
            "and must not be reported (ADR-0008).",
            "",
        ]
    if any(str(commit).startswith("MIXED") for commit in table["commit"]):
        lines += [
            "> **Seeds within an arm ran at different commits.** Averaging across code "
            "versions is averaging across experiments.",
            "",
        ]
    markdown_path.write_text("\n".join(lines))
    return [csv_path, markdown_path]
