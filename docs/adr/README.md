# Architecture Decision Records

One decision per file, numbered in order. A record is written when a choice is hard
to reverse, would surprise a reader who only saw the code, and came out of a real
trade-off. Everything else is just the obvious thing and needs no record.

| # | Decision | Status |
|---|---|---|
| [0001](0001-subject-level-partitioning.md) | Subject-level partitioning is the evaluation contract | accepted |
| [0002](0002-no-cognitive-scores-at-inference.md) | Cognitive test scores are excluded from the inference pathway | accepted |
| [0003](0003-oasis3-primary-oasis2-fallback.md) | OASIS-3 is primary; OASIS-2 is a declared fallback | accepted |
| [0004](0004-label-definition-and-target-visit.md) | Binary label definition and the target-visit rule | accepted |
| [0005](0005-shared-encoder-over-deformation-fusion.md) | Shared visit encoder with a temporal module | accepted |
| [0006](0006-elapsed-time-as-explicit-input.md) | Elapsed time is an explicit model input | accepted |
| [0007](0007-anatomy-supervision-is-training-only.md) | Anatomy targets supervise training only | accepted |
| [0008](0008-reproducibility-contract.md) | The reproducibility contract | accepted |
| [0009](0009-anatomy-targets-from-oasis3-freesurfer.md) | Anatomy targets come from OASIS-3's shipped FreeSurfer outputs | accepted |
| [0010](0010-subsampled-cohort-and-reduced-resolution.md) | Subsampled cohort at reduced resolution | accepted |
| [0011](0011-pretrained-vit-encoder.md) | Pretrained 3D ViT-B/16 shared encoder, and its OASIS overlap | accepted |

ADR-0004 reached `accepted` on 2026-11-24 from the real OASIS-3 MRI↔CDR gap
distribution (180-day window; unassessed visits excluded; every eligible visit is a
target). It must stay fixed before any real-cohort training run.

Superseding a record keeps the original file and marks its status; it is not deleted,
because the reasoning that was once persuasive is the useful part of the history.
