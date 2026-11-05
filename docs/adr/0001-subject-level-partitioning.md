---
status: accepted
date: 2026-11-05
---

# Subject-level partitioning is the evaluation contract

A subject contributes several visits, and visits from one person are far more similar
to each other than to anyone else's. We assign train, validation, and test partitions
**by subject**, so that no person's visits, derivatives, or normalization statistics
cross a partition boundary. Every headline comparison reports on the same held-out
test subjects.

## Considered options

**Visit-level random split.** Standard in much of the published literature and the
reason several reported accuracies are not comparable to ours. A subject's visit at
month 0 lands in train and month 24 in test; the model recognizes the person, not the
disease. Rejected — it measures re-identification, not detection.

**k-fold cross-validation over subjects.** Statistically stronger and still leak-free.
Rejected as the primary protocol because 3D longitudinal training is too expensive to
run k times per arm within one semester. Retained as an option for the final model if
compute allows.

## Consequences

- The usable sample is counted in subjects, not scans, and is correspondingly smaller.
- Partition assignment is drawn once, written to the manifest, and never redrawn.
  Re-drawing after seeing results is a form of test-set fitting.
- Our numbers will likely look *worse* than papers using visit-level splits. That gap
  is the point and the report says so rather than quietly competing on it.
- Class balance cannot be perfectly stratified, because stratification happens at
  subject level while the label lives at visit level.
