---
status: accepted
date: 2026-11-24
---

# Binary label definition and the target-visit rule

Label a target visit **impaired** when the clinical assessment nearest in time to that
visit gives **CDR > 0** (`CDRTOT` from OASIS-3 UDSb4), and **cognitively normal** when
it gives **CDR = 0**, subject to the maximum admissible gap below.

This decision is fixed from the real OASIS-3 distributions before any model trains on
the cohort. A label rule chosen after seeing results is not a label rule.

## Parameters (fixed)

| Parameter | Value |
|---|---|
| Maximum imaging-to-assessment gap | **180 days** |
| Sessions with no assessment inside the window | **Never labeled, never a target visit**; kept only as history (a context visit); count reported |
| Which eligible visits become target visits | **Every** eligible visit |

`max_assessment_gap_days = 180` is a cohort parameter recorded in the manifest, not a
constant buried in model code.

## Evidence from OASIS-3 (2026-11-24)

Sources: `OASIS3_MR_json.csv` (T1w sessions) and `OASIS3_UDSb4_cdr.csv` (CDR). Days from
entry come from the session label (`…_dNNNN`), matching the temporal module's elapsed-
time input.

| Quantity | Value |
|---|---|
| Unique T1w MR sessions | 2,832 (1,376 subjects) |
| CDR assessments with non-null `CDRTOT` | 8,625 (1,377 subjects) |
| Median \|MRI − nearest CDR\| gap | **92 days** |
| Exact same-day match (gap = 0) | 21 sessions (0.7%) |

Coverage of T1w sessions under candidate gaps:

| Max gap | T1 sessions labeled | Subjects with ≥2 labeled T1 visits |
|---|---|---|
| 0 days | 0.7% | — |
| 90 days | 48.8% | **291** |
| 120 days | 66.3% | 431 |
| 180 days | **86.5%** | **600** |
| 365 days | 96.1% | 669 |

[ADR-0010](0010-subsampled-cohort-and-reduced-resolution.md) samples **300** subjects
with usable longitudinal structure. A 90-day window leaves only 291 multi-visit
subjects — below that sample size. **180 days** clears the sample with headroom (600
multi-visit subjects; ~22% of labeled sessions impaired at the visit level) while
staying within a clinically conventional half-year window around the scan. Widening to
365 days adds little multi-visit yield for a much looser clinical coupling.

## Considered options

**Treat a missing rating as cognitively normal.** What the predecessor did on OASIS-1,
where it relabeled 181 of 416 subjects — 43.5% of the data — on documentation alone.
Rejected. It injects label noise of unknown size into the majority class, and OASIS-3's
longitudinal design makes an unrated session weaker evidence of normality than a
cross-sectional one.

**Use CDR at the imaging session only (gap = 0).** Cleanest, but OASIS-3 acquires
imaging and clinical assessment on separate days: only 0.7% of T1 sessions match
exactly. Rejected — it discards the cohort.

**90-day maximum gap.** Closer to the median gap (92 days), but only 291 subjects have
two or more labeled T1 visits — insufficient for the ADR-0010 sample of 300.
Rejected on feasibility, not on taste.

**Only the latest eligible visit per subject is a target.** Would shrink sequences and
throw away earlier labeled visits the longitudinal arm is meant to use. Rejected —
every eligible visit is a target; correlation across a subject's sequences is handled
at sampling or loss level, never by splitting the subject across partitions
([ADR-0001](0001-subject-level-partitioning.md)).

## Consequences

- A session without a CDR inside 180 days is never filled in as cognitively normal
  and never becomes a target visit. It stays in the cohort as a context visit: earlier
  history in the sequence of a later target visit of the same subject, as the
  synthetic cohort and `build_sequences` already assume. The count is reported in
  [`../data/cohort-definition.md`](../data/cohort-definition.md).
- The ADR-0010 sampling pool counts **labeled** visits: a subject needs two or more to
  be sampled. Context visits add history but do not make a subject eligible.
- MMSE may appear in UDSb4 for cohort description; it is never an inference feature
  ([ADR-0002](0002-no-cognitive-scores-at-inference.md)).
- A subject with several eligible visits contributes several correlated sequences;
  partitions stay subject-level.
- Changing `max_assessment_gap_days` after results exist would redefine the cohort;
  do not retune it to chase metrics.
