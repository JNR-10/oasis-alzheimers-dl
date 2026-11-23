"""Regenerate the results tables from recorded runs.

``make report``
``python -m oasis_dl.cli.report --runs experiments --out results/tables/results``

The command half of ``AGENTS.md``'s rule that results tables are generated from
recorded runs and never typed. It reads nothing but what runs wrote — no model is
loaded, no partition is read, no metric is recomputed — so running it twice on the same
records gives the same table, and running it on different records can only give a
different one because the runs differed.

Plain ``argparse`` rather than Hydra: this stage composes no experiment config. Its
only inputs are two directories, and routing them through a config group would imply a
choice that does not exist.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from oasis_dl.evaluation.reporting import collect_runs, results_table, write_results_table

log = logging.getLogger(__name__)

DEFAULT_RUNS = Path("experiments")
DEFAULT_OUT = Path("results/tables/results")


def main(argv: list[str] | None = None) -> int:
    """Collect run records, aggregate them, write the tables.

    Returns:
        0 on success, 1 when there is nothing to tabulate. A non-zero exit is the point:
        a report build that silently produced an empty table would be worse than one
        that stopped.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--runs",
        type=Path,
        default=DEFAULT_RUNS,
        help=f"directory holding run records, searched recursively (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"output path without extension; .csv and .md are written (default: {DEFAULT_OUT})",
    )
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    records = collect_runs(arguments.runs)
    if not records:
        log.error(
            "no completed runs under %s. Train an arm first, or point --runs elsewhere.",
            arguments.runs,
        )
        return 1

    table = results_table(records)
    for path in write_results_table(table, arguments.out):
        log.info("wrote %s", path)

    if bool(table["dirty"].any()):
        log.warning(
            "at least one run came from a dirty checkout; those numbers are not "
            "reproducible and must not be reported (ADR-0008)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
