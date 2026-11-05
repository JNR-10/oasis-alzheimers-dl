---
status: accepted
date: 2026-11-05
---

# OASIS-3 is the primary cohort; OASIS-2 is a declared fallback

OASIS-3 is the target dataset: it is longitudinal, large, and carries the clinical
assessments the labels need. The predecessor used OASIS-1, which is cross-sectional and
therefore cannot answer this project's question at all.

Access requires an application and approval, and the timing is outside our control.
**If OASIS-3 access does not arrive in time, the project falls back to OASIS-2** — also
longitudinal, far smaller (150 subjects, 373 sessions), and already used by LongFormer,
which makes comparison easier even as it shrinks the cohort.

## Consequences

- A fallback is a **declared** event, recorded in `docs/data/` with its date and cause,
  and named in the report. Silently swapping datasets and presenting the result as the
  original plan would misrepresent the study.
- The two datasets are not equivalent and results do not transfer between them. A
  fallback resets the baselines; it does not inherit numbers computed on OASIS-3.
- Cohort construction code reads a dataset adapter rather than hard-coding OASIS-3
  paths, so the fallback is a configuration change rather than a rewrite.
