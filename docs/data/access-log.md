# Data Access Log

OASIS-3 access status, tracked here rather than remembered.

**Status: access confirmed.** The team reported both OASIS-2 and OASIS-3 reachable on
2026-11-22. The approval and download dates below are still unfilled — they are not
known to the repository, and a guessed date is worse than a blank.

## OASIS-3

| Field | Value |
|---|---|
| Application submitted | 2026-11-05 |
| Submitted by | Mohit Manoj Barade |
| Approval received | **granted** — _date to fill in_ |
| DUA accepted on | _date to fill in_ |
| Download completed | _in progress_ |
| Local storage path | recorded in `.env`, not in git |
| Approximate size on disk | _to record once the T1w pull completes_ |

**What is downloaded:** T1-weighted scans and FreeSurfer derivatives only. PET and CT
are the bulk of OASIS-3 by volume and are read by no arm. Downloading them would cost
storage and buy nothing.

**What is preprocessed:** the sampled subjects only
([ADR-0010](../adr/0010-subsampled-cohort-and-reduced-resolution.md)), never the whole
download. Preprocessing is the dominant one-time CPU cost in the project and the sample
is what bounds it. Sample first, then preprocess — the reverse order spends the cost
this decision exists to avoid.

## OASIS-2

| Field | Value |
|---|---|
| Access | **granted** — _date to fill in_ |
| Role | Declared fallback ([ADR-0003](../adr/0003-oasis3-primary-oasis2-fallback.md)) |
| Size on disk | ~18 GB (`OAS2_RAW_PART1.tar.gz` 10 GB + `PART2` 8 GB) |

Access being in hand does **not** make OASIS-2 a second cohort to combine with the
primary one. The XNAT Central documentation states the OASIS datasets should not be
combined, because subjects may overlap across studies. OASIS-2 replaces OASIS-3 if the
fallback fires; it never supplements it, and it is not an external validation set for a
model trained on OASIS-3.

## Fallback trigger

[ADR-0003](../adr/0003-oasis3-primary-oasis2-fallback.md) makes the fallback to OASIS-2
a declared event. If it fires, record here: the date, the cause, who decided, and what
scope changed. The report then names the fallback rather than presenting OASIS-2 results
as the original plan.

| Field | Value |
|---|---|
| Fallback triggered | no |
| Date | — |
| Cause | — |

Access is no longer the reason this might fire. With both datasets in hand the remaining
trigger is cohort viability: if too few OASIS-3 subjects have two or more usable,
labelled T1 visits, the question cannot be asked on it regardless of access.

## Why this file exists

The feasibility audit names the project's dominant failure mode as an over-scoped plan
that cannot secure data in time. That risk has now closed; the compute budget replaced
it as the binding constraint, which is what ADR-0010 answers. A date in a file is harder
to lose than a date in a conversation.
