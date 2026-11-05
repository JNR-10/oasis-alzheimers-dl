# Data

**Nothing here is committed.** `.gitignore` excludes this tree's contents and CI fails
on tracked imaging volumes.

OASIS data is obtained through the official access process under its Data Use
Agreement, stored outside the repository, and reached via `OASIS_ROOT`. Access status
is tracked in [`../docs/data/access-log.md`](../docs/data/access-log.md).

| Directory | Holds |
|---|---|
| `raw/` | The download, untouched |
| `interim/` | Partly preprocessed volumes |
| `processed/` | Model-ready volumes at fixed shape and spacing |
| `manifests/` | Cohort manifests — subject identifiers, so local only |

A manifest pins which subjects, which visits, which partition, which label, under which
seed. It is the single source of truth for what a run trained on, and it is immutable:
a new cohort means a new filename (ADR-0008).
