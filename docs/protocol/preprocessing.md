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

---

## Amendment, 2026-11-22 — the encoder fixes the resampling target

[ADR-0011](../adr/0011-pretrained-vit-encoder.md) makes the shared encoder a ViT-B/16
initialized from BrainIAC's self-supervised weights. Pretrained weights only help when
the input resembles what they were fitted on, so stages 3–5 above are no longer free
choices:

| # | Stage | Was | Now |
|---|---|---|---|
| 3 | Brain extraction | any method | **HD-BET**, as BrainIAC used |
| 4 | Registration | to a template of our choosing | **rigid, to the BrainIAC template** (1 mm, LAS, 170×206×162) |
| 5 | Resampling | to a common isotropic voxel size | **resize the registered volume to 96³** |

Stage 2 remains bias-field correction, now specifically N4.

**Stage 5 is no longer isotropic.** The template is 170×206×162, so resizing it to a
cube scales each axis differently and the effective voxel is anisotropic. That is why
`configs/data/oasis3_longitudinal.yaml` carries `voxel_size_mm: _tbd_` rather than the
`2.0` it held before: the real figure comes from the template header in #15, and `2.0`
would have been a plausible number that nothing produces.

**What does not change.** The output is still 96³, so ADR-0010's compute argument holds
and nothing downstream moves. The stage order and the failure-handling rules above are
unchanged. And the open question under *Registration target* is untouched: ADR-0011
fixes *which* template, not whether a subject's visits are aligned to one another
before the subject is aligned to it. Registering each scan independently can still
absorb the within-subject atrophy the temporal module exists to read, so that remains
#16's question to settle empirically.
