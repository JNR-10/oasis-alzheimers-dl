"""Uncertainty via bootstrap over subjects.

Resampling **subjects**, not visits — and taking all of a drawn subject's visits with
them. Resampling visits would break the within-subject correlation the whole
subject-level protocol exists to respect, and would report intervals narrower than the
data supports (docs/protocol/evaluation.md).

A difference between arms is reported with its uncertainty or not at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

DEFAULT_RESAMPLES = 1000


@dataclass(frozen=True)
class ConfidenceInterval:
    """A point estimate with a percentile interval.

    Attributes:
        point: The metric computed on the real sample.
        lower, upper: Percentile bounds across resamples.
        level: Nominal coverage, e.g. 0.95.
        n_resamples: How many resamples produced the bounds. Lower than requested when
            some draws left the metric undefined.
    """

    point: float
    lower: float
    upper: float
    level: float
    n_resamples: int

    def __str__(self) -> str:
        raise NotImplementedError


def bootstrap_metrics(
    y_true: Sequence[int] | NDArray[np.int_],
    y_score: Sequence[float] | NDArray[np.float64],
    subject_ids: Sequence[str],
    *,
    threshold: float,
    metrics: Sequence[str] = ("roc_auc", "balanced_accuracy", "macro_f1", "sensitivity"),
    n_resamples: int = DEFAULT_RESAMPLES,
    level: float = 0.95,
    seed: int,
) -> dict[str, ConfidenceInterval]:
    """Bootstrap the metric set by resampling subjects with replacement.

    Each resample draws subjects, then takes **all** of a drawn subject's visits. A
    subject drawn twice contributes their visits twice — that is what preserves the
    clustering per-visit resampling would destroy.

    Resamples where a metric is undefined — a draw containing one class only — are
    dropped from that metric's distribution rather than imputed, and the surviving count
    is reported in ``n_resamples``.

    Args:
        y_true: Per-visit binary labels.
        y_score: Per-visit scores.
        subject_ids: Per-visit subject id, same length as the other two.
        threshold: Operating point, chosen on validation data and held fixed.
        metrics: Which fields of :class:`~oasis_dl.evaluation.metrics.ClassificationMetrics`
            to bootstrap.
        n_resamples: Resample count.
        level: Nominal coverage.
        seed: Recorded alongside the interval so it can be reproduced exactly.

    Raises:
        ValueError: If inputs are empty or of different lengths.
    """
    raise NotImplementedError
