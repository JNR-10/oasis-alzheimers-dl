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

## Break the code to check the test

A test that passes tells you nothing on its own — it may be passing for a reason that
has nothing to do with the behaviour it names. Before trusting a test that guards
something silent, delete the guard and confirm the test goes red. If it stays green,
the test is decoration.

Five guards in this suite were verified that way and the mutation is recorded beside
each one: the temporal module's `src_key_padding_mask`, the `sorted()` before the split
shuffle, subject-level resampling in the bootstrap, and on checkpoint loading both the
`matched == 0` check and the refusal of a partial match. Each fails silently in
production — the shapes stay right and a plausible number still comes out — which is
exactly the class of bug a passing test is most likely to be hiding.

### Do not reduce with `.sum()`

Several modules here end in a `LayerNorm`, whose output is normalised to zero mean and
therefore **sums to zero for any input**. So `output.sum().backward()` has an exactly
zero derivative with respect to everything upstream, and a gradient test built on it
reports "no gradient reached" no matter what the module does.

This bit once: a gradient test passed locally on a lucky random draw and failed in CI.
Use `tests/unit/test_temporal.py::_scalarize` — a seeded random projection — or any
reduction that is not annihilated by normalisation.
