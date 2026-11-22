# Tests

The structure mirrors `src/oasis_dl/`.

- `unit/` — pure logic, no cohort and no GPU. Partitioning, metrics, and the loss
  objective belong here; they are testable against synthetic labels and need no OASIS
  access, so they can be written before the data lands.
- `integration/` — anything touching a real cohort or a real checkpoint. Marked
  `@pytest.mark.requires_data` or `@pytest.mark.requires_gpu` so CI skips them.

The one test that matters most is the leakage check: that no subject's visits reach
more than one partition (ADR-0001). A bug there inflates every number in the report
without failing anything.

Markers are declared in `pyproject.toml`. `make test` runs what needs neither data
nor GPU.

## Building cohorts to test against

`helpers.py` builds cohort **records** in memory — subjects, visits, sequences,
labels — with no scans on disk:

```python
from tests.helpers import make_sequences, make_subject_labels

labels = make_subject_labels(n_subjects=200, impaired_fraction=0.24, seed=1)
sequences = make_sequences(n_subjects=20, seed=1)
```

Reach for these whenever the thing under test operates on records rather than pixels:
partitioning, leakage checks, metric aggregation, provenance. They are fast, they need
no OASIS access, and `scan_path` points at a path that does not exist — so a test that
accidentally opens a scan fails loudly instead of passing for the wrong reason.

`make_subject_labels` returns its keys in shuffled order on purpose. A partitioner must
be deterministic in its seed *and* independent of mapping order; sorted input hides
that bug, because sorted input is what a correct partitioner produces internally
anyway.

For anything that needs real volumes — preprocessing, datasets, collation, training —
use the generated cohort fixtures rather than these builders.

## The test that matters most

Leakage: that no subject's visits reach more than one partition (ADR-0001). It is worth
writing before the thing it checks, and worth re-running after every transformation
that could have broken it, because a leak inflates every reported number without
failing anything.
