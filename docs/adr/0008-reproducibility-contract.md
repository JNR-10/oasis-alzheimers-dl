---
status: accepted
date: 2026-11-05
---

# The reproducibility contract: src layout, composed configs, nothing heavy in git

Four rules, enforced by tooling rather than by discipline:

**Code is an installed package.** `src/oasis_dl/` with a `src` layout, installed with
`pip install -e .`. Notebooks and scripts import it; nothing reaches across directories
with `sys.path` edits. Tests therefore run against the same import graph as training.

**Experiments are composed configs.** Hydra config groups under `configs/`, so the five
experiment arms differ by a named override rather than by five near-duplicate files.
An ablation that changes one variable should be one line of diff — if it is more, the
configs are hiding an uncontrolled difference.

**Data, weights, and outputs never enter git.** Enforced by `.gitignore`, a pre-commit
size hook, and a CI job that fails on tracked imaging volumes or checkpoints. The OASIS
Data Use Agreement makes this a licensing matter, not only a repository-hygiene one.

**Every run records what produced it.** Config, git commit, seed, and library versions
are written beside the metrics, under `experiments/`. A result whose configuration
cannot be recovered does not go in the report.

## Consequences

- Working on the repo requires an install step; cloning and running a script will not
  work. The cost is paid once and buys a real import boundary.
- Results are not in git, so the report's numbers are reproduced by rerunning rather
  than by reading them out of the tree. `experiments/` keeps the index that makes the
  rerun possible.
- `docs/proposal/` holds the one deliberately large tracked file — the submitted PDF —
  and the CI size check exempts that path.
