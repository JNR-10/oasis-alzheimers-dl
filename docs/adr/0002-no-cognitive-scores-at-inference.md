---
status: accepted
date: 2026-11-05
---

# Cognitive test scores are excluded from the inference pathway

MMSE and similar cognitive assessments are never inference features. They may inform
how labels are defined, and they may be reported in cohort tables, but no model in
this project reads one when producing a prediction.

This is inherited directly from the predecessor project, which found that a model using
**MMSE alone — one feature — reached 86.9% accuracy, matching its full eight-feature
baseline**. MMSE carried 37.8–61.7% of importance across tree models. Those models were
MMSE proxies: they predicted a dementia diagnosis from a test that measures dementia.
High accuracy, no diagnostic value.

## Consequences

- Reported accuracy will be lower than an MMSE-inclusive model. That is the intended
  trade and must not be read as a weaker result.
- A label derived from CDR is still partly informed by cognitive assessment, so
  circularity is reduced rather than eliminated. The report states this explicitly
  instead of claiming the inference pathway is fully independent of cognition.
- Any feature that is a near-deterministic function of a cognitive score is treated as
  the score itself and excluded on the same grounds.
