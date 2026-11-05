# Preprocessing Specification

One pipeline, applied identically to every arm. A difference in preprocessing between
arms would confound the ablations, so the pipeline is fixed before the first run and
its version is recorded with every result.

## Stages

| # | Stage | Purpose |
|---|---|---|
| 1 | Orientation standardization | Put every scan in a canonical axis order |
| 2 | Bias-field correction | Remove low-frequency intensity drift from the scanner |
| 3 | Brain extraction | Remove non-brain tissue |
| 4 | Registration | Bring scans into a common frame so voxel positions are comparable |
| 5 | Resampling | Common isotropic voxel size |
| 6 | Crop / pad | Fixed input dimensions |
| 7 | Intensity normalization | Comparable intensity scale across scans |

Stage order is load-bearing: bias correction before brain extraction (the field biases
the extraction), extraction before registration (non-brain tissue misleads the
alignment).

## Registration target

Open decision. Two options with different implications for the longitudinal arm:

**Each scan to a common template.** Simple and uniform. Risks absorbing real
within-subject atrophy into the registration itself — if every scan is warped to the
same template, the change the temporal module is meant to detect may be partly removed.

**Within-subject rigid alignment, then the subject to a template.** Preserves
within-subject change, which is the signal of interest. More steps, more failure modes.

The second is preferred on the reasoning above, but the choice is confirmed empirically
on a held-out handful of subjects before the cohort is processed. Settle it before
freezing the cohort and record the outcome as an ADR.

## Failure handling

Preprocessing failures silently change the cohort, which is how a pipeline quietly
turns into a different experiment. Therefore:

- Every stage reports success or failure per session, and failures are written to the
  manifest rather than dropped.
- A session that fails any stage is excluded by inclusion rule 2 and counted in
  [`../data/cohort-definition.md`](../data/cohort-definition.md).
- Visual QC on a random sample after the first full run, and on every failure. An
  automated pipeline over medical images that nobody has looked at is not validated.

## Augmentation

Conservative and training-only: small rotations, translations, scaling, and mild
intensity perturbation. Deliberately modest — aggressive spatial augmentation on
structural MRI can destroy the atrophy signal the task depends on.

For a longitudinal sample, the **same** spatial augmentation is applied across a
subject's visits. Augmenting each visit independently injects apparent anatomical change
between visits, which is exactly the signal the temporal module is trying to read.
