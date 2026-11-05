# Tests

Empty until there is behaviour to test. The structure mirrors `src/oasis_dl/`.

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
