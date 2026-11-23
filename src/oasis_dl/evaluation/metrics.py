"""The metric set from docs/protocol/evaluation.md.

One implementation, used by every arm. Two arms computing their metrics with different
code would be comparing the code rather than the models, so this is the only place a
reportable number is produced.

Plain accuracy is deliberately absent. Under this project's class balance a model that
predicts one class scores well while learning nothing, so offering the number at all
invites it into a table where it would mislead.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ClassificationMetrics",
    "aggregate_by_subject",
    "compute_metrics",
    "select_threshold",
]


def _as_arrays(
    y_true: Sequence[int] | NDArray[np.int_],
    y_score: Sequence[float] | NDArray[np.float64],
) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    """Validate the pair once, so every entry point refuses the same bad input."""
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    if labels.size == 0:
        raise ValueError("no predictions to score")
    if labels.shape != scores.shape:
        raise ValueError(
            f"y_true has {labels.shape} entries and y_score has {scores.shape}; "
            f"they must align element-wise"
        )
    if not np.isin(labels, (0, 1)).all():
        raise ValueError(f"labels must be 0 or 1, found {sorted(set(labels.tolist()))}")
    return labels, scores


def _ratio(numerator: int, denominator: int) -> float:
    """NaN when the denominator is empty, never a stand-in value.

    Specificity with no negatives is not 0, not 1, and not 0.5 — it is unmeasured, and
    a number here would be reported as though it had been measured.
    """
    return float(numerator) / denominator if denominator else float("nan")


@dataclass(frozen=True)
class ClassificationMetrics:
    """Every metric the protocol requires, for one arm on one partition.

    Attributes:
        roc_auc: Threshold-free ranking quality. The headline number. NaN when only one
            class is present — NaN rather than 0.5, so that "not measurable" cannot be
            read as "no better than chance".
        balanced_accuracy: Mean of sensitivity and specificity.
        macro_f1: F1 averaged over classes, weighting the minority class equally.
        sensitivity: True positive rate on the impaired class. The costly error is a
            missed impairment, so this is read before specificity.
        specificity: True negative rate.
        true_negatives, false_positives, false_negatives, true_positives: The confusion
            matrix, kept flat so it survives a round trip through CSV.
        threshold: The operating point the threshold-dependent metrics used. Recorded so
            that a test number can never be traced back to a threshold picked on test
            data.
        n: Number of evaluated units.
        n_impaired: How many were impaired. Reported beside every metric, because a
            metric without its class balance is not interpretable.
    """

    roc_auc: float
    balanced_accuracy: float
    macro_f1: float
    sensitivity: float
    specificity: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    threshold: float
    n: int
    n_impaired: int

    def as_dict(self) -> dict[str, float | int]:
        """Flat mapping, for writing into a results table."""
        return dict(asdict(self))


def compute_metrics(
    y_true: Sequence[int] | NDArray[np.int_],
    y_score: Sequence[float] | NDArray[np.float64],
    *,
    threshold: float,
) -> ClassificationMetrics:
    """Compute the full metric set at a fixed threshold.

    Takes the threshold rather than choosing one, so that a test-set threshold cannot
    be selected by accident. Callers get theirs from :func:`select_threshold` run on
    validation data.

    Args:
        y_true: Binary labels, 1 for impaired.
        y_score: Predicted probability or score for the impaired class.
        threshold: Operating point.

    Raises:
        ValueError: If inputs are empty or of different lengths.
    """
    labels, scores = _as_arrays(y_true, y_score)
    predicted = scores >= threshold

    true_positives = int(np.sum(predicted & (labels == 1)))
    false_negatives = int(np.sum(~predicted & (labels == 1)))
    true_negatives = int(np.sum(~predicted & (labels == 0)))
    false_positives = int(np.sum(predicted & (labels == 0)))

    sensitivity = _ratio(true_positives, true_positives + false_negatives)
    specificity = _ratio(true_negatives, true_negatives + false_positives)

    return ClassificationMetrics(
        roc_auc=_roc_auc(labels, scores),
        # NaN propagates: with one class absent this is unmeasured, and the mean of a
        # measured half and an unmeasured half is not a balanced accuracy.
        balanced_accuracy=(sensitivity + specificity) / 2,
        macro_f1=_macro_f1(labels, predicted),
        sensitivity=sensitivity,
        specificity=specificity,
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_positives=true_positives,
        threshold=float(threshold),
        n=int(labels.size),
        n_impaired=int(np.sum(labels == 1)),
    )


def _roc_auc(labels: NDArray[np.int_], scores: NDArray[np.float64]) -> float:
    """Threshold-free ranking quality, or NaN when only one class is present.

    NaN rather than 0.5 is the whole point: a single-class partition has no ranking to
    be right or wrong about, and 0.5 would be read as "no better than chance".
    """
    if np.unique(labels).size < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def _macro_f1(labels: NDArray[np.int_], predicted: NDArray[np.bool_]) -> float:
    """F1 averaged over classes, so the minority class weighs as much as the majority."""
    from sklearn.metrics import f1_score

    return float(f1_score(labels, predicted.astype(int), average="macro", zero_division=0))


def select_threshold(
    y_true: Sequence[int] | NDArray[np.int_],
    y_score: Sequence[float] | NDArray[np.float64],
    *,
    objective: str = "youden",
) -> float:
    """Choose an operating point on **validation** data.

    Never call this on the test partition. Choosing a threshold on test data turns the
    test set into a tuning set, and the number it produces is no longer held out.

    Args:
        y_true: Binary validation labels.
        y_score: Validation scores.
        objective: ``"youden"`` maximizes sensitivity + specificity - 1, treating the
            two errors as equally costly. ``"sensitivity_first"`` picks the
            highest-specificity point still reaching 0.90 sensitivity, for when a missed
            impairment is the worse error.

    Raises:
        ValueError: If both classes are not present, or the objective is unknown.
    """
    labels, scores = _as_arrays(y_true, y_score)
    if np.unique(labels).size < 2:
        raise ValueError(
            "cannot choose a threshold from a single-class validation set: there is no "
            "trade-off to balance."
        )
    if objective not in _OBJECTIVES:
        raise ValueError(
            f"unknown objective {objective!r}; supported: {', '.join(sorted(_OBJECTIVES))}"
        )

    from sklearn.metrics import roc_curve

    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    if objective == "youden":
        best = int(np.argmax(true_positive_rate - false_positive_rate))
    else:
        reaching = np.flatnonzero(true_positive_rate >= _SENSITIVITY_FLOOR)
        # The lowest threshold always reaches sensitivity 1.0, so `reaching` is never
        # empty and the floor is always attainable.
        best = int(reaching[np.argmin(false_positive_rate[reaching])])

    chosen = float(thresholds[best])
    # roc_curve prepends an infinite threshold representing "predict nothing". Returning
    # it would classify every subject as normal while looking like a valid choice.
    return chosen if math.isfinite(chosen) else float(np.max(scores))


_OBJECTIVES = frozenset({"youden", "sensitivity_first"})
_SENSITIVITY_FLOOR = 0.90


def aggregate_by_subject(
    y_true: Sequence[int] | NDArray[np.int_],
    y_score: Sequence[float] | NDArray[np.float64],
    subject_ids: Sequence[str],
) -> tuple[NDArray[np.int_], NDArray[np.float64], list[str]]:
    """Collapse per-visit predictions to one per subject.

    Per-visit and per-subject metrics diverge when visit count correlates with label —
    which it plausibly does, since subjects who progress tend to be followed longer. The
    protocol reports both when they differ rather than choosing the flattering one.

    A subject's score is the mean over their target visits; their label is impaired if
    any target visit is impaired, matching :func:`~oasis_dl.data.splits.subject_label`.

    Returns:
        Labels, scores, and subject ids, ordered by subject id.

    Raises:
        ValueError: If the three inputs are not the same length.
    """
    labels, scores = _as_arrays(y_true, y_score)
    if len(subject_ids) != labels.size:
        raise ValueError(
            f"{len(subject_ids)} subject ids for {labels.size} predictions; every "
            f"prediction must name the subject it came from"
        )

    rows: dict[str, list[int]] = {}
    for index, subject in enumerate(subject_ids):
        rows.setdefault(str(subject), []).append(index)

    ordered = sorted(rows)
    subject_labels = np.array([int(labels[rows[subject]].max()) for subject in ordered], dtype=int)
    subject_scores = np.array(
        [float(scores[rows[subject]].mean()) for subject in ordered], dtype=float
    )
    return subject_labels, subject_scores, ordered
