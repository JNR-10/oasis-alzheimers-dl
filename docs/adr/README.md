# Architecture Decision Records

One decision per file, numbered in order. A record is written when a choice is hard
to reverse, would surprise a reader who only saw the code, and came out of a real
trade-off. Everything else is just the obvious thing and needs no record.

| # | Decision | Status |
|---|---|---|
| [0001](0001-subject-level-partitioning.md) | Subject-level partitioning is the evaluation contract | accepted |
| [0002](0002-no-cognitive-scores-at-inference.md) | Cognitive test scores are excluded from the inference pathway | accepted |
| [0003](0003-oasis3-primary-oasis2-fallback.md) | OASIS-3 is primary; OASIS-2 is a declared fallback | accepted |
| [0004](0004-label-definition-and-target-visit.md) | Binary label definition and the target-visit rule | **proposed** |
| [0005](0005-shared-encoder-over-deformation-fusion.md) | Shared visit encoder with a temporal module | accepted |
| [0006](0006-elapsed-time-as-explicit-input.md) | Elapsed time is an explicit model input | accepted |
| [0007](0007-anatomy-supervision-is-training-only.md) | Anatomy targets supervise training only | accepted |
| [0008](0008-reproducibility-contract.md) | The reproducibility contract | accepted |

ADR-0004 is the one open decision. It must reach `accepted` before any model trains,
because a label rule chosen after seeing results is not a label rule.

Superseding a record keeps the original file and marks its status; it is not deleted,
because the reasoning that was once persuasive is the useful part of the history.
