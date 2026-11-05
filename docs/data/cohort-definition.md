# Cohort Definition

**Unfilled.** This document is completed from the real data before training begins, and
is then frozen. It is the human-readable counterpart to the machine-readable manifest
at `data/manifests/`; if the two disagree, the manifest is what ran and this file is
wrong.

## Inclusion rules

Each rule is checkable and reports how many sessions it removed. A rule that cannot be
stated as a check does not belong here.

| # | Rule | Sessions removed |
|---|---|---|
| 1 | Session has a usable T1-weighted scan | _tbd_ |
| 2 | Scan passes preprocessing without failure | _tbd_ |
| 3 | A clinical assessment exists within the admissible gap (ADR-0004) | _tbd_ |
| 4 | Subject retains at least one eligible target visit | _tbd_ |

The longitudinal arm additionally requires subjects with two or more eligible visits.
That subset is reported separately, because the single-scan and longitudinal arms must
be compared on the **same subjects** or the first core ablation confounds the
architecture change with a cohort change.

## Resulting cohort

| | Subjects | Visits | Impaired | Cognitively normal |
|---|---|---|---|---|
| Train | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| Validation | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| Test | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| **Total** | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

## Visit-count distribution

How many subjects have 1, 2, 3, … eligible visits. This distribution decides whether the
temporal module has anything to learn from: a cohort that is overwhelmingly
single-visit cannot support a longitudinal claim regardless of architecture.

| Eligible visits | Subjects |
|---|---|
| 1 | _tbd_ |
| 2 | _tbd_ |
| 3 | _tbd_ |
| 4+ | _tbd_ |

## Cohort balance

Age, sex, scanner, and education by partition and by label. Reported so that a
difference between arms can be checked against a difference in who is in them. Where
a covariate correlates with the label, it is named as a candidate shortcut rather than
presented as disease signal.

## Partition seed

Partitions are drawn once with a recorded seed and never redrawn. Record it here and in
the manifest.

| Field | Value |
|---|---|
| Seed | _tbd_ |
| Drawn on | _tbd_ |
| Ratio (train/val/test) | _tbd_ |
