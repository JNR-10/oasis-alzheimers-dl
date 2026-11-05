# Evaluation Specification

How a number in the report is produced. One implementation, shared by every arm — if
two arms compute their metrics with different code, their comparison measures the code.

## Unit of evaluation

Metrics are computed over **target visits** in the test partition, with the subject
recorded for each. Two aggregations are reported where they differ:

- **Per-visit**: every target visit counts once. Subjects with more visits weigh more.
- **Per-subject**: each subject contributes one score, averaged over their visits.

They diverge when visit count correlates with label — which it plausibly does, since
subjects who progress tend to be followed longer. When they diverge, both are reported
and the divergence is discussed rather than resolved by picking the better one.

## Uncertainty

**Bootstrap over test subjects**, not over visits — resampling visits would break the
within-subject correlation the subject-level protocol exists to respect. 95% intervals
from 1000 resamples, with the seed recorded.

**Across seeds**, where compute allows: each arm trained with several seeds and reported
as mean ± spread. This is the one that calibrates whether a between-arm gap is real,
because it captures training variance that a bootstrap over a single fitted model cannot.

## Threshold selection

ROC-AUC is threshold-free and is the headline. Metrics that need a threshold —
balanced accuracy, macro-F1, sensitivity, specificity — use a threshold chosen on
**validation** subjects and then applied unchanged to test. Choosing a threshold on test
data converts the test set into a tuning set.

## Reporting

Every results table carries: cohort size by partition, the metric with its interval, the
seed count, and the git commit. A number without that context is not reportable.

Results tables are **generated** from recorded runs, not typed. A hand-copied number is
a number that can silently stop matching the run that produced it.
