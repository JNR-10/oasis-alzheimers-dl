# Dataset Card — OASIS-3

Status of this document: written before access was confirmed. Figures quoted from the
OASIS-3 publication describe the released dataset, **not** our cohort. Our cohort
numbers are unknown until [`cohort-definition.md`](cohort-definition.md) is filled in
from the real data, and nothing in this project should quote a cohort size until then.

## Source

**OASIS-3: Longitudinal Neuroimaging, Clinical, and Cognitive Dataset for Normal Aging
and Alzheimer Disease.** LaMontagne et al., 2019. DOI
[10.1101/2019.12.13.19014902](https://doi.org/10.1101/2019.12.13.19014902).
Distribution: <https://sites.wustl.edu/oasisbrains/home/oasis-3/>

Longitudinal by design — subjects are imaged and assessed repeatedly over years, which
is why this project can ask its question at all and why the predecessor's OASIS-1
cohort could not.

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
   yield a usable label? Open parameter in [ADR-0004](../adr/0004-label-definition-and-target-visit.md).
3. **How many subjects have two or more usable, labeled T1 visits?** This number, not
   the headline dataset size, determines whether the longitudinal arm is trainable.
4. **Scanner and protocol heterogeneity across sessions.** OASIS-3 spans multiple
   scanners; if scanner correlates with label, it is a shortcut and must be reported.
   FreeSurfer version varies with scanner strength (see ADR-0009), so scanner and
   derivative version are confounded and both are recorded in the manifest.

## Related datasets

**OASIS-2** — the declared fallback under [ADR-0003](../adr/0003-oasis3-primary-oasis2-fallback.md).
Longitudinal, roughly 150 subjects and 373 sessions. Used by LongFormer, which makes
comparison easier even as the small size limits what can be concluded.

**OASIS-1** — the predecessor project's dataset. Cross-sectional, so it cannot support
a longitudinal question. Relevant here only as the source of prior findings.
