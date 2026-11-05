---
status: accepted
date: 2026-11-05
---

# A shared visit encoder with a temporal module, not pairwise deformation fusion

Each visit is encoded by the **same** 3D network, and the resulting embeddings are
combined by a temporal module. The alternative — LongFormer's design, which pairs a
current scan with a deformation field or optical-flow representation computed against a
prior scan — is treated as a baseline to compare against, not as the design to copy.

## Considered options

**Pairwise deformation fusion (LongFormer).** Models change directly and is the closest
published precedent. Rejected as our design for three reasons: it is defined over scan
*pairs*, so a subject with five visits has no natural single representation; it requires
a registration quality we cannot guarantee across the whole cohort, and a bad
deformation field fails silently; and reusing it would leave us with no comparison arm.

**One network over the stacked sequence.** Concatenating visits as extra channels.
Rejected — it fixes the visit count, and OASIS-3 subjects have wildly varying counts.

## Consequences

- Variable-length sequences are handled naturally: the encoder is per-visit and the
  temporal module attends over however many exist.
- The single-scan control arm is exactly this model with the temporal module removed,
  which is what makes the first core ablation a clean one-variable comparison.
- Weight sharing means the encoder sees every visit of every subject, which is more
  gradient signal per parameter than a per-position encoder would get.
- Change is modeled in embedding space rather than voxel space, so the model can miss
  fine-grained local deformation that LongFormer captures. Whether that costs accuracy
  is an empirical question the comparison answers.
