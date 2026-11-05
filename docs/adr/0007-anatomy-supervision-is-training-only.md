---
status: accepted
date: 2026-11-05
---

# Anatomy targets supervise training and are never read at inference

An auxiliary head predicts structural quantities — hippocampal and ventricular volumes
and related measures — during training, under a combined objective
`L_total = L_classification + λ · L_anatomy`. At inference the head is discarded and the
prediction comes from the imaging pathway alone.

The aim is to shape the encoder's representation toward anatomy already known to track
Alzheimer's pathology, where the predecessor measured the effect directly: hippocampal
volume fell 17% / 27% and ventricular volume rose 34% / 52% across CDR 0 → 0.5 → 1.

## Considered options

**Feed measured anatomy in as input features.** Almost certainly more accurate, and the
predecessor's imaging-only model already reached 84.5% this way. Rejected because it
rebuilds the hand-engineered pipeline this project exists to replace, and makes the
system depend at inference on a segmentation step that can fail silently.

**Skip auxiliary supervision entirely.** The honest null. It remains the control arm of
the second core ablation — the project tests whether this head helps rather than
assuming it does.

## Consequences

- Anatomy targets are needed only for training subjects, so a target extraction that
  fails on a test subject costs nothing.
- `λ` is a hyperparameter tuned on validation subjects only. Tuning it on test subjects
  would turn the ablation into a search for a favorable number.
- The targets must come from a source independent of the label. A target derived from a
  clinician's reading of the same scan that produced the label would reintroduce
  circularity through the back door.
- Target extraction depends on the dataset's derivatives. OASIS-1's FSL segmentations
  are not OASIS-3's FreeSurfer outputs, so the predecessor's extraction code informs
  this work but does not port to it unchanged.
