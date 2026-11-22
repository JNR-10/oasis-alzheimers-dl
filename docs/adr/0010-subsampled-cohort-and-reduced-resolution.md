---
status: accepted
date: 2026-11-22
---

# Train on a fixed subject subsample at reduced resolution

OASIS-3 access has arrived. The full cohort is affordable to *hold* but not to *train
on* repeatedly inside a five-week semester on shared GPUs. Rather than move to a smaller
dataset, the cohort is subsampled and the volumes are coarsened.

| Parameter | Was | Now |
|---|---|---|
| Subjects | all with >= 2 eligible visits | **300**, sampled at subject level |
| Voxel size | 1.5 mm isotropic | **2.0 mm** isotropic |
| Volume shape | 128 x 128 x 128 | **96 x 96 x 96** |
| Visits per sequence | 5 | **3**, counting back from the target visit |

Cost per training run scales with voxel count, so the resolution change alone is ~2.4x.
Together these cut a run by roughly an order of magnitude.

## Considered options

**Switch to OASIS-2.** 150 subjects, 18 GB, no sampling needed. Rejected because a
subsample of OASIS-3 dominates it: the same compute buys twice the subjects, sample size
becomes a parameter rather than a fixed property of the dataset, and scaling up later is
a rerun rather than a new dataset.

**Cut arms instead, keep the full cohort.** Insufficient alone. Arm count is already the
first thing cut under pressure, and cutting it leaves per-run cost untouched.

**Keep 1.5 mm.** Rejected. Resolution is the largest single lever, and the signal this
project reads is gross structural atrophy — at 2 mm the hippocampus still spans several
hundred voxels. Fine detail is not what separates CDR 0 from CDR 1.

## Consequences

- **The sample is drawn once, with a recorded seed, before partitioning, and never
  redrawn.** Sampling after seeing results is the same error as redrawing a split
  ([ADR-0001](0001-subject-level-partitioning.md)) and harder to spot, because the
  subjects that disappear leave no trace in the partition.
- **Raising `max_subjects` produces a different cohort.** Its results do not merge with
  the smaller cohort's, and a comparison between the two is not an ablation.
- **The study is powered for large effects only.** 300 subjects at 60/20/20 leaves ~60
  test subjects, so subject-level bootstrap intervals will be wide. Intervals are
  reported with every point estimate, and a null is reported as *underpowered* rather
  than as evidence of no effect.
- **Sampling runs before preprocessing**, so only sampled subjects are processed. This
  is where most of the one-time CPU cost is avoided.
- Only T1-weighted scans are downloaded. PET and CT are the bulk of OASIS-3 by volume
  and go unused here.
- Preprocessing at 2 mm is a cohort-wide parameter, so changing it later invalidates
  every result computed before the change, exactly as a split change would.
