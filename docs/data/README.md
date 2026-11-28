# Dataset Card — OASIS-3

Status: access granted, cohort not yet built. Figures below describe the **released
dataset**, not our cohort. Our cohort numbers stay unknown until
[`cohort-definition.md`](cohort-definition.md) is filled in from the real data, and
nothing in this project quotes a cohort size until then.

## Source

**OASIS-3: Longitudinal Neuroimaging, Clinical, and Cognitive Dataset for Normal Aging
and Alzheimer Disease.** LaMontagne et al., 2019. DOI
[10.1101/2019.12.13.19014902](https://doi.org/10.1101/2019.12.13.19014902).
Distribution: <https://sites.wustl.edu/oasisbrains/home/oasis-3/>, served through XNAT
Central.

Longitudinal by design — subjects are imaged and assessed repeatedly over years, which
is why this project can ask its question at all and why the predecessor's OASIS-1
cohort could not.

The distribution has grown since publication: the paper reports 1,098 participants and
2,168 MR sessions, the live site 1,378 participants and 2,842 MR sessions. The site is
also internally inconsistent — 1,378 in prose, 1,379 in its statistics block, and a
755 + 622 normal/impaired breakdown that sums to 1,377. **Cite none of them.** The
number that goes in the report is the one computed from our own download.

## Terms of use

OASIS data is released under a Data Use Agreement accepted at application time. Three
consequences bind this repository:

- No imaging data, derivatives, or subject-identifying manifests are committed. Enforced
  by `.gitignore`, a pre-commit size hook, and a CI job.
- The DUA is cited in any publication or report that uses the data.
- Redistribution is not permitted. Anyone reproducing this work applies for their own
  access; this repository ships the code and the cohort *definition*, never the data.

## What we use

| Element | Use |
|---|---|
| T1-weighted structural MRI | The model's only imaging input |
| Session timing (days from entry) | Elapsed time for the temporal module |
| Clinical assessment (CDR) | Label derivation — a training target, never an input |
| MMSE | Cohort description only — excluded from inference by [ADR-0002](../adr/0002-no-cognitive-scores-at-inference.md) |
| Imaging derivatives | Source of anatomy targets — FreeSurfer outputs shipped with OASIS-3, per [ADR-0009](../adr/0009-anatomy-targets-from-oasis3-freesurfer.md) |

**Not downloaded:** PET and CT. They are most of OASIS-3 by volume and are read by no
arm.

## Scale and sampling

The full cohort is affordable to store and not to train on repeatedly inside the weeks
remaining. Under
[ADR-0010](../adr/0010-subsampled-cohort-and-reduced-resolution.md) the project trains
on **300 subjects sampled at subject level**, at 2 mm isotropic, 96³, with 3 visits per
sequence.

The sample is drawn once with a recorded seed, **before** partitioning and **before**
preprocessing. Drawing it before preprocessing is the point: that is where the one-time
CPU cost lives. Raising the sample size later produces a different cohort rather than a
larger one.

## Open questions, to settle before the cohort is fixed

1. **Which derivatives ship with OASIS-3, and in what form?** **Answered — see
   [ADR-0009](../adr/0009-anatomy-targets-from-oasis3-freesurfer.md).** OASIS-3 ships
   FreeSurfer outputs per MR session: subcortical volumes from `aseg.stats` and
   cortical surface/region measures from `aparc.stats`, distributed as CSV
   spreadsheets alongside the full FreeSurfer output tree (`t1.mgz`, brainmasks,
   segmentation maps, surface maps). Not every session has derivatives — only those
   that passed FreeSurfer QC (`pass` or `pass with edits`). FreeSurfer version varies
   by scanner strength (1.5T → 5.0/5.1; 3T → 5.3-HCP-patch), and re-running FreeSurfer
   produces different values, so this project uses the shipped outputs. Sources: OASIS-3
   Imaging Data Dictionary v2.3; LaMontagne et al. 2019
   ([doi:10.1101/2019.12.13.19014902](https://doi.org/10.1101/2019.12.13.19014902)).
2. **How far apart can an imaging session and its clinical assessment be** and still
   yield a usable label? **Answered — [ADR-0004](../adr/0004-label-definition-and-target-visit.md):**
   maximum gap **180 days** to the nearest CDR (`CDRTOT`); sessions outside the window
   are never labeled or used as targets, only as history; every labeled visit is a
   target visit. Chosen from the
   real gap distribution (median 92 days); 90 days left too few multi-visit subjects for
   the ADR-0010 sample of 300.
3. **How many subjects have two or more usable, labeled T1 visits?** Under the 180-day
   rule the metadata census finds **600** such subjects — enough headroom to draw the
   ADR-0010 sample of 300. Final cohort counts still wait on the frozen manifest after
   preprocessing.
4. **Scanner and protocol heterogeneity across sessions.** OASIS-3 spans multiple
   scanners; if scanner correlates with label, it is a shortcut and must be reported.
   FreeSurfer version varies with scanner strength (see ADR-0010), so scanner and
   derivative version are confounded and both are recorded in the manifest.

## Related datasets

The OASIS releases are siblings, not versions. XNAT Central's documentation states they
**should not be combined**, because subjects may overlap across studies. That rules out
using one as an external validation set for a model trained on another.

**OASIS-2** — the declared fallback under [ADR-0003](../adr/0003-oasis3-primary-oasis2-fallback.md).
Longitudinal, 150 subjects and 373 sessions, ~18 GB. Used by LongFormer, which makes
comparison easier even as the small size limits what can be concluded. Access is in
hand; that makes the fallback cheap to execute, not a second cohort to merge.

**OASIS-1** — the predecessor project's dataset. Cross-sectional, so it cannot support
a longitudinal question. Relevant here only as the source of prior findings.

**OASIS-4** — a clinical cohort of 663 participants and 676 MR sessions, drawn from
memory-clinic referrals rather than community volunteers. Effectively single-visit at
~1.02 sessions per subject, so it cannot train the longitudinal arms. Named here
because a reader who knows the OASIS family will ask why it is absent.
