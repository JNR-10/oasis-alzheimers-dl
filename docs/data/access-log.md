# Data Access Log

OASIS-3 access status, tracked here rather than remembered.

**Status: access confirmed.** The team reported both OASIS-2 and OASIS-3 reachable on
2026-11-22. A team member confirmed that the approval date below comes from the
approval email; the access report date is not the approval date.

## OASIS-3

| Field | Value |
|---|---|
| Application submitted | 2026-11-05 |
| Submitted by | Mohit Manoj Barade |
| Approval received | **granted** — 2026-11-09 |
| DUA accepted on | 2026-11-09 |
| Download completed | 2026-11-25 — 960-session T1w list only |
| Local storage path | recorded in `.env`, not in git |
| Approximate size on disk | T1w files: 14.77 GiB; total download: _tbd_ |

**Download scope:** The downloaded scan folders contain T1-weighted MRI only; no PET
or CT image files were found. The supplied metadata archive also contains PET and CT
metadata spreadsheets, not PET or CT imaging volumes.

On 2026-11-25, the local T1w download contained 1,435 scans in 960 sessions from
the supplied session list. The completion date is based on local download folder
creation timestamps. All 1,435 scans decompressed and passed basic NIfTI header and
payload-size checks; this is not visual quality control. A FreeSurfer measurements
spreadsheet was restored from the supplied archive; 912 of those sessions have a
non-quarantined matching row.

The selection method and creator of the 960-session list are unknown. The list is
sorted, but its order does not establish how its 300 subjects were selected. This
download is not the ADR-0010 subject sample and has not been label-balanced.

The FreeSurfer spreadsheet has left and right hippocampus, lateral ventricle and
entorhinal volume columns, plus `IntraCranialVol`. All seven columns contain positive
numeric values in all 2,681 spreadsheet rows. The OASIS imaging dictionary identifies
`IntraCranialVol` as an intracranial volume measure; whether that field is exactly
FreeSurfer's estimated total intracranial volume (eTIV) has not been independently
confirmed.

An unlabeled inventory of the available OASIS-3 MRI metadata contains 1,376 subjects,
2,832 T1w sessions and 4,116 T1w scan metadata entries. Of these, 960 sessions and
1,435 distinct scans are currently present on disk. The FreeSurfer spreadsheet has
matching entries for 2,677 T1w sessions, including 2,675 with a non-quarantined QC
status. These are metadata and file-presence counts, not a training cohort or a
source-side completeness check. The subject-identifying inventory stays outside git.

The requested sampling pool is the existing 960-session list, rather than the full
OASIS-3 metadata inventory. After the label rule is accepted in #13, #14 must assess
its eligible subjects and whether they support the planned fixed, label-stratified
sample. Existing scans will be reused; additional scans, if needed, can be downloaded
after sampling. Sessions with multiple scans remain unchanged; scan selection is a
#14 decision. No partitions have been drawn here.

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
