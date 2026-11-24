"""Metrics and the subject-level bootstrap — docs/protocol/evaluation.md.

Expected values here are worked by hand, not recomputed with the same library the
implementation uses. A test that derives its expectation the way the code does passes by
construction and can never disagree with it.

The worked example throughout:

    y_true  = [0,   0,   1,    1  ]
    y_score = [0.1, 0.4, 0.35, 0.8]   threshold 0.5  ->  predictions [0, 0, 0, 1]

    confusion: TN=2  FP=0  FN=1  TP=1
    sensitivity 1/2 = 0.5      specificity 2/2 = 1.0     balanced accuracy 0.75
    ROC-AUC: of the 4 (neg, pos) pairs, 3 are correctly ordered -> 0.75
    F1(impaired) = 2(1)(0.5)/1.5 = 2/3   F1(normal) = 2(2/3)(1)/(5/3) = 0.8
    macro-F1 = (2/3 + 0.8)/2 = 0.7333...
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from oasis_dl.evaluation.bootstrap import ConfidenceInterval, bootstrap_metrics
from oasis_dl.evaluation.metrics import (
    aggregate_by_subject,
    compute_metrics,
    select_threshold,
)

Y_TRUE = [0, 0, 1, 1]
Y_SCORE = [0.1, 0.4, 0.35, 0.8]


class TestComputeMetrics:
    def test_the_worked_example(self) -> None:
        m = compute_metrics(Y_TRUE, Y_SCORE, threshold=0.5)
        assert (m.true_negatives, m.false_positives, m.false_negatives, m.true_positives) == (
            2,
            0,
            1,
            1,
        )
        assert m.sensitivity == pytest.approx(0.5)
        assert m.specificity == pytest.approx(1.0)
        assert m.balanced_accuracy == pytest.approx(0.75)
        assert m.roc_auc == pytest.approx(0.75)
        assert m.macro_f1 == pytest.approx((2 / 3 + 0.8) / 2)

    def test_the_threshold_is_recorded_on_the_result(self) -> None:
        """So a reported number can always be traced to the threshold that produced it."""
        assert compute_metrics(Y_TRUE, Y_SCORE, threshold=0.31).threshold == pytest.approx(0.31)

    def test_class_balance_travels_with_the_metrics(self) -> None:
        """A metric without its class balance is not interpretable."""
        m = compute_metrics(Y_TRUE, Y_SCORE, threshold=0.5)
        assert (m.n, m.n_impaired) == (4, 2)

    def test_moving_the_threshold_moves_the_confusion_matrix(self) -> None:
        strict = compute_metrics(Y_TRUE, Y_SCORE, threshold=0.9)
        assert (strict.true_positives, strict.false_negatives) == (0, 2)

    def test_roc_auc_is_nan_on_a_single_class_partition(self) -> None:
        """NaN, never 0.5. "Not measurable" must not read as "no better than chance"."""
        m = compute_metrics([1, 1, 1], [0.2, 0.7, 0.9], threshold=0.5)
        assert math.isnan(m.roc_auc)

    def test_the_threshold_metrics_survive_a_single_class_partition(self) -> None:
        """Sensitivity is still defined with no negatives; specificity is not."""
        m = compute_metrics([1, 1, 1], [0.2, 0.7, 0.9], threshold=0.5)
        assert m.sensitivity == pytest.approx(2 / 3)
        assert math.isnan(m.specificity)

    def test_roc_auc_is_threshold_free(self) -> None:
        a = compute_metrics(Y_TRUE, Y_SCORE, threshold=0.1)
        b = compute_metrics(Y_TRUE, Y_SCORE, threshold=0.9)
        assert a.roc_auc == pytest.approx(b.roc_auc)

    def test_as_dict_is_flat_and_round_trips_through_csv(self) -> None:
        flat = compute_metrics(Y_TRUE, Y_SCORE, threshold=0.5).as_dict()
        assert all(isinstance(v, (int, float)) for v in flat.values())
        assert "roc_auc" in flat and "threshold" in flat

    def test_plain_accuracy_is_not_offered(self) -> None:
        """Deliberate: on a 76/24 cohort it rewards predicting one class."""
        assert "accuracy" not in compute_metrics(Y_TRUE, Y_SCORE, threshold=0.5).as_dict()

    @pytest.mark.parametrize(("y_true", "y_score"), [([], []), ([0, 1], [0.5]), ([0], [0.1, 0.2])])
    def test_empty_or_mismatched_input_is_refused(
        self, y_true: list[int], y_score: list[float]
    ) -> None:
        with pytest.raises(ValueError):
            compute_metrics(y_true, y_score, threshold=0.5)


class TestSelectThreshold:
    def test_youden_picks_the_separating_point(self) -> None:
        """Perfectly separable at 0.5: every negative below, every positive above."""
        chosen = select_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        m = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold=chosen)
        assert m.sensitivity == pytest.approx(1.0)
        assert m.specificity == pytest.approx(1.0)

    def test_sensitivity_first_reaches_its_floor(self) -> None:
        """For when a missed impairment is the worse error."""
        rng = np.random.default_rng(0)
        y_true = np.repeat([0, 1], 50)
        y_score = np.concatenate([rng.normal(0.4, 0.2, 50), rng.normal(0.6, 0.2, 50)])
        chosen = select_threshold(y_true, y_score, objective="sensitivity_first")
        assert compute_metrics(y_true, y_score, threshold=chosen).sensitivity >= 0.90

    def test_a_single_class_validation_set_is_refused(self) -> None:
        with pytest.raises(ValueError):
            select_threshold([1, 1, 1], [0.2, 0.5, 0.9])

    def test_an_unknown_objective_is_refused(self) -> None:
        with pytest.raises(ValueError, match="youden"):
            select_threshold(Y_TRUE, Y_SCORE, objective="whatever-wins")


class TestAggregateBySubject:
    def test_a_subjects_visits_collapse_to_one_row(self) -> None:
        y, s, ids = aggregate_by_subject([0, 0, 1], [0.2, 0.4, 0.9], ["A", "A", "B"])
        assert ids == ["A", "B"]
        assert s[0] == pytest.approx(0.3)
        assert list(y) == [0, 1]

    def test_a_subject_who_ever_converts_is_impaired(self) -> None:
        """Matches splits.subject_label, so the two cannot disagree about a subject."""
        y, _, ids = aggregate_by_subject([0, 1], [0.2, 0.9], ["A", "A"])
        assert ids == ["A"] and list(y) == [1]

    def test_ordering_is_by_subject_id_not_by_arrival(self) -> None:
        _, _, ids = aggregate_by_subject([0, 0], [0.1, 0.2], ["B", "A"])
        assert ids == ["A", "B"]

    def test_per_visit_and_per_subject_can_differ(self) -> None:
        """The reason both are reported: visit count plausibly correlates with label,
        since subjects who progress tend to be followed longer."""
        y_true = [1, 1, 1, 0]
        y_score = [0.9, 0.9, 0.9, 0.8]
        ids = ["A", "A", "A", "B"]
        per_visit = compute_metrics(y_true, y_score, threshold=0.85)
        y, s, _ = aggregate_by_subject(y_true, y_score, ids)
        per_subject = compute_metrics(y, s, threshold=0.85)
        assert per_visit.n == 4 and per_subject.n == 2

    def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(ValueError):
            aggregate_by_subject([0, 1], [0.2], ["A", "B"])


class TestBootstrapResamplesSubjects:
    @staticmethod
    def _clustered(
        n_subjects: int, visits: int, seed: int
    ) -> tuple[list[int], list[float], list[str]]:
        """Each subject repeated `visits` times with an identical row.

        Maximal within-subject correlation, so subject-level and visit-level resampling
        must disagree by as much as they ever will.
        """
        rng = np.random.default_rng(seed)
        y_true: list[int] = []
        y_score: list[float] = []
        ids: list[str] = []
        for i in range(n_subjects):
            label = int(i % 2)
            score = float(rng.uniform(0.2, 0.8))
            y_true += [label] * visits
            y_score += [score] * visits
            ids += [f"S{i:03d}"] * visits
        return y_true, y_score, ids

    def test_clustered_data_yields_a_wider_interval_than_the_same_rows_independent(self) -> None:
        """The acceptance criterion, and the whole reason this module exists.

        Identical rows. Told they belong to 30 subjects, the interval must be wider than
        when each row is called its own subject — resampling visits discards the
        clustering and reports confidence the data does not support.
        """
        y_true, y_score, ids = self._clustered(n_subjects=30, visits=6, seed=1)
        independent = [f"V{i:04d}" for i in range(len(ids))]

        by_subject = bootstrap_metrics(
            y_true, y_score, ids, threshold=0.5, metrics=("roc_auc",), n_resamples=400, seed=7
        )["roc_auc"]
        by_visit = bootstrap_metrics(
            y_true,
            y_score,
            independent,
            threshold=0.5,
            metrics=("roc_auc",),
            n_resamples=400,
            seed=7,
        )["roc_auc"]

        assert by_subject.upper - by_subject.lower > by_visit.upper - by_visit.lower

    def test_the_point_estimate_is_the_real_sample_not_a_resample_mean(self) -> None:
        y_true, y_score, ids = self._clustered(n_subjects=20, visits=3, seed=2)
        interval = bootstrap_metrics(
            y_true, y_score, ids, threshold=0.5, metrics=("roc_auc",), n_resamples=200, seed=7
        )["roc_auc"]
        assert interval.point == pytest.approx(
            compute_metrics(y_true, y_score, threshold=0.5).roc_auc
        )

    def test_the_interval_contains_the_point_estimate(self) -> None:
        y_true, y_score, ids = self._clustered(n_subjects=25, visits=2, seed=3)
        for interval in bootstrap_metrics(
            y_true, y_score, ids, threshold=0.5, n_resamples=300, seed=7
        ).values():
            assert interval.lower <= interval.point <= interval.upper

    def test_the_same_seed_reproduces_the_interval(self) -> None:
        y_true, y_score, ids = self._clustered(n_subjects=20, visits=3, seed=4)

        def run() -> ConfidenceInterval:
            return bootstrap_metrics(
                y_true,
                y_score,
                ids,
                threshold=0.5,
                metrics=("roc_auc",),
                n_resamples=200,
                seed=11,
            )["roc_auc"]

        a, b = run(), run()
        assert (a.lower, a.upper) == (b.lower, b.upper)

    def test_undefined_resamples_are_dropped_and_counted_not_imputed(self) -> None:
        """A single-class draw has no ROC-AUC. Imputing 0.5 would pull every interval
        toward chance and understate the spread."""
        y_true = [0, 0, 0, 0, 1, 1]
        y_score = [0.1, 0.2, 0.3, 0.4, 0.8, 0.9]
        ids = ["A", "B", "C", "D", "E", "F"]
        interval = bootstrap_metrics(
            y_true, y_score, ids, threshold=0.5, metrics=("roc_auc",), n_resamples=300, seed=5
        )["roc_auc"]
        assert interval.n_resamples < 300
        assert interval.n_resamples > 0

    def test_a_requested_metric_that_does_not_exist_is_refused(self) -> None:
        y_true, y_score, ids = self._clustered(n_subjects=10, visits=2, seed=6)
        with pytest.raises(ValueError, match="vibes"):
            bootstrap_metrics(
                y_true, y_score, ids, threshold=0.5, metrics=("vibes",), n_resamples=10, seed=1
            )

    def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(ValueError):
            bootstrap_metrics([0, 1], [0.2, 0.8], ["A"], threshold=0.5, n_resamples=10, seed=1)

    def test_the_interval_is_printable_for_a_results_table(self) -> None:
        y_true, y_score, ids = self._clustered(n_subjects=15, visits=2, seed=8)
        text = str(
            bootstrap_metrics(
                y_true, y_score, ids, threshold=0.5, metrics=("roc_auc",), n_resamples=100, seed=1
            )["roc_auc"]
        )
        assert "[" in text and "]" in text
