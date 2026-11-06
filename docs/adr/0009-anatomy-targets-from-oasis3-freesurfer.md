---
status: accepted
date: 2026-11-06
---

# Anatomy targets come from OASIS-3's shipped FreeSurfer outputs

The auxiliary anatomy targets of [ADR-0007](0007-anatomy-supervision-is-training-only.md)
are read from the FreeSurfer outputs that OASIS-3 ships with each MR session, not
extracted by re-running segmentation in this project and not derived from coordinate
bounding boxes as the predecessor did on OASIS-1.

This settles open question 1 of the [dataset card](../data/README.md) and unblocks the
sizing of [issue #17](https://github.com/JNR-10/oasis-alzheimers-dl/issues/17) before
the data arrives.

## What ships

The OASIS-3 Imaging Data Dictionary and the LaMontagne et al. 2019 paper describe the
released derivatives:

- Each T1w MR session is processed through FreeSurfer. Subcortical volumes come from
  the `aseg.stats` output; cortical surface measures and parcellated region volumes
  come from the `aparc.stats` output.
- The volumetric values are distributed as CSV spreadsheets (the "FreeSurfer output
  spreadsheet") alongside the full FreeSurfer output tree (`t1.mgz`, brainmasks,
  segmentation maps, surface maps, regional statistics).
- Outputs are anonymized: dates, timestamps, QC staff, and raw file paths are removed.
- **Not every MR session has FreeSurfer output.** Only sessions that passed FreeSurfer
  QC with a status of `pass` or `pass with edits` are shipped. Sessions that failed QC
  or were never processed have no derivatives and therefore no anatomy targets.
- FreeSurfer version varies by scanner strength: 1.5T sessions were processed with
  FreeSurfer 5.0 or 5.1; 3T sessions with FreeSurfer 5.3-HCP-patch. Re-running
  FreeSurfer on the same input produces different values than the shipped outputs, so
  this project uses the shipped outputs and does not reprocess.

Sources: OASIS-3 Imaging Data Dictionary v2.3; LaMontagne et al., 2019
([doi:10.1101/2019.12.13.19014902](https://doi.org/10.1101/2019.12.13.19014902));
OASIS-3 distribution page, <https://sites.wustl.edu/oasisbrains/home/oasis-3/>.

## What this means for the target list

The candidate targets in `src/oasis_dl/preprocessing/anatomy.py` are all supportable
from `aseg.stats` and `aparc.stats`:

- Left/right hippocampus — `aseg.stats` (`Left-Hippocampus`, `Right-Hippocampus`).
- Left/right lateral ventricle — `aseg.stats` (`Left-Lateral-Ventricle`,
  `Right-Lateral-Ventricle`).
- Total ventricular volume — derivable as left/right lateral ventricle + left/right
  inferior-lateral ventricle + 3rd + 4th + 5th ventricle, per the dictionary's stated
  formula. This is the measure that matches the predecessor's "ventricular volume rose
  34% / 52% across CDR 0 → 0.5 → 1" and is preferred over lateral ventricle alone.
- Left/right entorhinal cortex volume — `aparc.stats` (`ctx_lh_entorhinal`,
  `ctx_rh_entorhinal`).
- Total gray matter — derivable as Cortex + CorticalWhiteMatter + SubCortGray, or read
  from the shipped `BrainSeg` / `TotalGrayVol` fields.
- Estimated total intracranial volume (eTIV) — shipped by FreeSurfer and used by
  `normalize_targets` to scale regional volumes. ICV varies within a subject across
  sessions (mean subject SD ≈ 15.75 cm³ per the dictionary), so normalization is a
  per-session scaling, not a per-subject constant.

Ratios such as CSF-to-brain are derived from the shipped `Volume of CSF` and eTIV
rather than computed by an external pipeline.

## Considered options

**Re-run FreeSurfer in this project.** Rejected. FreeSurfer's output depends on its
version and on whether the input is the dicom-derived `t1.mgz` or a reconverted
nifti, and the dictionary warns that values differ. Re-running would produce a
different target distribution from the one the dataset ships, would take compute
this project does not budget for, and would make the targets a function of our
processing rather than of the dataset. Use the shipped outputs.

**Parse OASIS-1-style FSL segmentations and Talairach coordinate boxes, as the
predecessor did.** Rejected. OASIS-3 does not ship FSL segmentations; it ships
FreeSurfer outputs. The coordinate-box approach was a workaround for OASIS-1's
derivatives, not a method choice, and FreeSurfer's labeled regions are a better
source than bounding boxes intersected with a segmentation.

**Read anatomy targets from the raw T1w through a segmentation model trained in this
project.** Rejected. ADR-0007 requires the target source to be independent of the
label. A segmentation we train could be shaped by the same shortcuts the encoder
learns, and re-introduces the circularity ADR-0007 was written to prevent. The shipped
FreeSurfer outputs were produced by the data holder under their QC, independent of
this project's model.

## Consequences

- Target extraction (#17) reads CSV/`aseg.stats`/`aparc.stats` fields rather than
  running a segmentation pipeline. The work is parsing and aggregation, not image
  processing, which is why #2 noted that #17 is "an afternoon or a week" depending on
  this answer.
- Sessions without shipped FreeSurfer output have no anatomy targets. Under
  ADR-0007, anatomy targets are needed only for training subjects, so a missing
  target on a test subject costs nothing. A training subject without targets is
  excluded from the auxiliary-head loss but can still contribute to the
  classification loss; the cohort manifest (#14) records which sessions have
  derivatives so the split is honest about this.
- The FreeSurfer version varies by scanner strength, and scanner is itself a
  candidate shortcut (dataset card open question 4). The manifest records scanner
  and FreeSurfer version per session so a correlation with the label can be named
  rather than absorbed.
- eTIV is read per session and used to normalize regional volumes. The per-session
  variance in ICV is a property of the data, not noise to be removed, and is
  reported rather than suppressed.
- If a future OASIS-3 release changes the FreeSurfer version or reprocesses
  sessions, the target distribution changes with it. The manifest records the
  OASIS-3 release identifier so a re-extraction is detectable and repeatable.
- This decision is recorded before the cohort is frozen (#14) and before any target
  is extracted (#17), so the target source cannot drift to suit results.
