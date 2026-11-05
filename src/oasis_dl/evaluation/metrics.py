"""The metric set from docs/protocol/evaluation.md.

One implementation, used by every arm. Two arms computing their metrics with different
code would be comparing the code rather than the models, so this is the only place a
reportable number is produced.

Plain accuracy is deliberately absent. Under this project's class balance a model that
predicts one class scores well while learning nothing, so offering the number at all
invites it into a table where it would mislead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


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
        raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    """
    raise NotImplementedError
