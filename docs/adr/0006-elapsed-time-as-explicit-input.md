---
status: accepted
date: 2026-11-05
---

# Elapsed time is an explicit model input, not an assumed grid

Visit spacing in OASIS-3 is irregular — months for some subjects, years for others. The
temporal module receives the **actual elapsed time** between visits as a continuous
input, rather than treating position in the sequence as a proxy for time.

## Considered options

**Positional encoding over visit index.** The standard Transformer treatment. Rejected:
it tells the model a visit is "the third one," not that it came four years later. Two
subjects with identical sequences of embeddings and wildly different spacing would be
read identically, and atrophy *rate* is the entire clinical signal of interest.

**Resample every subject onto a uniform grid.** Makes batching trivial. Rejected: it
invents observations the scanner never made. The feasibility audit names this directly —
the model must handle missing visits "without inventing uniform time points."

## Consequences

- Batching is harder: sequences vary in both length and spacing, so the collate step
  pads and masks rather than stacking.
- Elapsed time is the one non-imaging inference input the project permits. It is
  admissible because it is an acquisition fact, not a measurement of the patient — it
  carries no information about cognitive status on its own.
- Age at visit is *not* given to the model on the same reasoning, because it correlates
  strongly with the label through normal aging and would act as a shortcut.
