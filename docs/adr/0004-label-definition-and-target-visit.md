---
status: proposed
date: 2026-11-05
---

# Binary label definition and the target-visit rule

**This decision is proposed, not accepted.** It is fixed once the cohort is inspected,
and it is fixed *before* any model is trained. Recording it now pins the intent so the
choice cannot drift to suit results later.

The proposal: label a target visit **impaired** when the clinical assessment nearest in
time to that visit gives CDR > 0, and **cognitively normal** when it gives CDR = 0,
subject to a maximum admissible gap between the imaging session and the assessment.

Three parameters stay open until the cohort is seen: the maximum imaging-to-assessment
gap, the handling of visits with no assessment inside that window, and whether every
eligible visit of a subject becomes a target visit or only the latest.

## Considered options

**Treat a missing rating as cognitively normal.** What the predecessor did on OASIS-1,
where it relabeled 181 of 416 subjects — 43.5% of the data — on documentation alone.
Rejected here. It injects label noise of unknown size into exactly the class that
dominates the data, and OASIS-3's longitudinal design makes an unrated session much
weaker evidence of normality than a cross-sectional one.

**Use CDR at the imaging session only, with no window.** Cleanest, but OASIS-3 acquires
imaging and clinical assessment on separate days, so an exact-match rule discards most
of the cohort.

## Consequences

- Visits without a usable assessment are excluded, not relabeled, and the excluded
  count is reported.
- Every subject contributing more than one target visit multiplies correlated examples;
  that correlation is handled at the loss or sampling level, and never by moving those
  visits into different partitions (see [ADR-0001](0001-subject-level-partitioning.md)).
- The gap threshold is a cohort parameter in the manifest, not a constant in the code.
