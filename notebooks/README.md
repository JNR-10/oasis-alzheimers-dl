# Notebooks

Exploration only. Nothing a result depends on lives here — a notebook is not a
reproducible entrypoint, and a number that exists only in a cell cannot be regenerated.

Notebooks import the installed package (`from oasis_dl...`) rather than reaching across
directories with `sys.path` edits. Outputs are stripped on commit by `nbstripout`,
which keeps diffs readable and keeps rendered patient images out of git.

When something here becomes load-bearing, move it into `src/oasis_dl/` and give it a
test.
