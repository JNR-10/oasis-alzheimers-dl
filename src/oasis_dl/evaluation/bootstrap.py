"""Uncertainty via bootstrap over subjects.

Resampling **subjects**, not visits — and taking all of a drawn subject's visits with
them. Resampling visits would break the within-subject correlation the whole
subject-level protocol exists to respect, and would report intervals narrower than the
data supports (docs/protocol/evaluation.md).

A difference between arms is reported with its uncertainty or not at all.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from oasis_dl.evaluation.metrics import (
    ClassificationMetrics,
    compute_metrics,
    rows_by_subject,
)

__all__ = ["DEFAULT_RESAMPLES", "ConfidenceInterval", "bootstrap_metrics"]

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
        """``0.812 [0.701, 0.903]`` — the form a results table carries.

        The interval is never dropped. A point estimate alone invites a reader to
        compare two arms that overlap completely.
        """
        return f"{self.point:.3f} [{self.lower:.3f}, {self.upper:.3f}]"


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
        ValueError: If inputs are empty, of different lengths, or name a metric that
            :class:`~oasis_dl.evaluation.metrics.ClassificationMetrics` does not carry.
    """
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    if labels.size == 0:
        raise ValueError("no predictions to bootstrap")
    if not labels.shape == scores.shape == (len(subject_ids),):
        raise ValueError(
            f"y_true {labels.shape}, y_score {scores.shape} and subject_ids "
            f"({len(subject_ids)},) must all be the same length"
        )
    unknown = sorted(set(metrics) - set(ClassificationMetrics.__dataclass_fields__))
    if unknown:
        raise ValueError(
            f"not a metric: {', '.join(unknown)}. Available: "
            f"{', '.join(sorted(ClassificationMetrics.__dataclass_fields__))}"
        )
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")
    if not 0 < level < 1:
        raise ValueError(f"level must lie in (0, 1), got {level}")

    # Rows grouped by subject, so a drawn subject brings all of their visits. Sorted,
    # so the draw depends on the seed and not on the order rows arrived in.
    clusters = [np.array(rows, dtype=int) for rows in rows_by_subject(subject_ids).values()]

    point = compute_metrics(labels, scores, threshold=threshold).as_dict()
    draws: dict[str, list[float]] = {name: [] for name in metrics}

    generator = np.random.default_rng(seed)
    for _ in range(n_resamples):
        drawn = generator.integers(0, len(clusters), size=len(clusters))
        selected = np.concatenate([clusters[index] for index in drawn])
        resampled = compute_metrics(
            labels[selected], scores[selected], threshold=threshold
        ).as_dict()
        for name in metrics:
            value = float(resampled[name])
            # A draw that contains one class only leaves ROC-AUC undefined. Dropping it
            # keeps the distribution honest; imputing 0.5 would pull every interval
            # toward chance and understate the spread.
            if not math.isnan(value):
                draws[name].append(value)

    tail = (1 - level) / 2 * 100
    intervals: dict[str, ConfidenceInterval] = {}
    for name in metrics:
        values = draws[name]
        if not values:
            # Every resample was undefined. Reported as NaN bounds rather than omitted,
            # so the metric still appears in the table as unmeasured.
            intervals[name] = ConfidenceInterval(
                point=float(point[name]),
                lower=float("nan"),
                upper=float("nan"),
                level=level,
                n_resamples=0,
            )
            continue
        lower, upper = np.percentile(values, [tail, 100 - tail])
        intervals[name] = ConfidenceInterval(
            point=float(point[name]),
            lower=float(lower),
            upper=float(upper),
            level=level,
            n_resamples=len(values),
        )
    return intervals
