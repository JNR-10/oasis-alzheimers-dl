# Cohort Definition

**Filled 2026-11-27 from the first real cohort run** (`python -m oasis_dl.cli.build_cohort`
against `OASIS_ROOT`, manifest at `${OASIS_ROOT}/manifests/oasis3_cohort.csv`). It is
the human-readable counterpart to the machine-readable manifest; if the two disagree,
the manifest is what ran and this file is wrong.

Counts below are **pre-preprocessing**: the manifest freezes the sampled subject list
and the eligibility audit before any scan is processed (ADR-0010). Fields that depend
on preprocessing survival (rule 4, post-process visit counts) remain `_tbd_` until #15
runs and the manifest is re-frozen.

## Inclusion rules

Each rule is checkable and reports how many sessions it removed. A rule that cannot be
stated as a check does not belong here.

| # | Rule | Sessions removed |
|---|---|---|
| 1 | Session has a usable T1-weighted scan | 0 (the source emits T1w sessions only; 4,116 T1w scan rows collapse to 2,832 sessions) |
| 2 | A clinical assessment exists within the admissible gap (ADR-0004) | 0 removed — 382 sessions have no CDR within 180 days and become **context visits** (`label=None`), not removed |
| 3 | Subject survives the ADR-0010 subject sample | 2,173 sessions (sessions of the 995 non-sampled subjects) |
| 4 | Scan passes preprocessing without failure | _tbd_ (#15) |
| 5 | Subject retains at least one eligible target visit | _tbd_ (post-preprocess) |

Rule 3 sits **before** rule 4 on purpose. Sampling precedes preprocessing, so sessions
removed by the sample are never processed at all — that ordering is where the project's
compute saving comes from. Sessions dropped by rule 3 are recorded as sampled-out, not
as preprocessing failures; conflating the two would make the pipeline look like it was
failing on hundreds of scans it never touched.

The longitudinal arm additionally requires subjects with two or more eligible visits.
That subset is reported separately, because the single-scan and longitudinal arms must
be compared on the **same subjects** or the first core ablation confounds the
architecture change with a cohort change.

## Subject sample (ADR-0010)

Drawn once, stratified by label, before partitioning. Recorded here and in the manifest.

| Field | Value |
|---|---|
| Eligible subjects before sampling (≥1 labeled T1 visit) | 1,295 |
| Subjects with ≥2 labeled T1 visits | 600 |
| `max_subjects` | 300 |
| Sample seed | 20261122 |
| Drawn on | 2026-11-27 |
| Label balance before sampling | 874 cognitively normal / 421 impaired |
| Label balance after sampling | 202 cognitively normal / 98 impaired |

The two balance rows exist so a drift between them is visible rather than assumed away.

## Resulting cohort

Subject counts per partition are fixed by the seed and will not change. Visit and
label counts at visit level are pre-preprocessing; they will be finalized after #15.

| | Subjects | Target visits (pre-preprocess) | Impaired (subject-level) | Cognitively normal (subject-level) |
|---|---|---|---|---|
| Train | 180 | 368 | 59 | 121 |
| Validation | 61 | 109 | 20 | 41 |
| Test | 59 | 118 | 19 | 40 |
| **Total** | 300 | 595 | 98 | 202 |

## Visit-count distribution

How many subjects have 1, 2, 3, … eligible visits (within the sampled 300,
pre-preprocessing). This distribution decides whether the temporal module has anything
to learn from: a cohort that is overwhelmingly single-visit cannot support a
longitudinal claim regardless of architecture.

| Eligible visits | Subjects (sampled, pre-preprocess) |
|---|---|
| 1 | 144 |
| 2 | 79 |
| 3 | 33 |
| 4 | 19 |
| 5 | 8 |
| 6 | 7 |
| 7 | 5 |
| 8 | 4 |
| 9 | 1 |

## Cohort balance

Age, sex, scanner, and education by partition and by label. Reported so that a
difference between arms can be checked against a difference in who is in them. Where
a covariate correlates with the label, it is named as a candidate shortcut rather than
presented as disease signal. _tbd_ — filled from the frozen manifest once #15 lands
and the per-session scanner/age/sex fields are joined.

## Partition seed

Partitions are drawn once with a recorded seed and never redrawn. Record it here and in
the manifest.

| Field | Value |
|---|---|
| Seed | 42 |
| Drawn on | 2026-11-27 |
| Ratio (train/val/test) | 0.60 / 0.20 / 0.20 |
