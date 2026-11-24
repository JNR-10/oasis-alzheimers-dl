"""Tests for :func:`oasis_dl.data.cohort.sample_subjects` (ADR-0010 / issue #14).

Sampling is drawn once, stratified by label, deterministic in the seed, and independent
of dict insertion order. A bug here silently changes the cohort every result rests on.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest
from tests.helpers import make_subject_labels

from oasis_dl.data.cohort import sample_subjects
from oasis_dl.types import Label, SubjectId


def _labels_from_ids(
    subject_labels: dict[SubjectId, Label],
    sampled: list[SubjectId],
) -> Counter[Label]:
    return Counter(subject_labels[sid] for sid in sampled)


class TestSampleSubjects:
    def test_none_keeps_every_subject_sorted(self) -> None:
        labels = make_subject_labels(40, seed=1)
        sampled = sample_subjects(labels, max_subjects=None, seed=0)
        assert sampled == sorted(labels)

    def test_requesting_more_than_exist_keeps_all(self) -> None:
        labels = make_subject_labels(20, seed=2)
        sampled = sample_subjects(labels, max_subjects=100, seed=0)
        assert sampled == sorted(labels)

    def test_zero_returns_empty(self) -> None:
        labels = make_subject_labels(20, seed=3)
        assert sample_subjects(labels, max_subjects=0, seed=0) == []

    def test_size_matches_max_subjects(self) -> None:
        labels = make_subject_labels(100, impaired_fraction=0.24, seed=4)
        sampled = sample_subjects(labels, max_subjects=30, seed=0)
        assert len(sampled) == 30

    def test_sampled_ids_are_sorted(self) -> None:
        labels = make_subject_labels(80, seed=5)
        sampled = sample_subjects(labels, max_subjects=25, seed=11)
        assert sampled == sorted(sampled)

    def test_sampled_ids_are_a_subset(self) -> None:
        labels = make_subject_labels(60, seed=6)
        sampled = sample_subjects(labels, max_subjects=20, seed=2)
        assert set(sampled) <= set(labels)

    def test_deterministic_in_the_seed(self) -> None:
        labels = make_subject_labels(100, seed=7)
        left = sample_subjects(labels, max_subjects=40, seed=99)
        right = sample_subjects(labels, max_subjects=40, seed=99)
        assert left == right

    def test_different_seeds_usually_differ(self) -> None:
        labels = make_subject_labels(100, seed=8)
        a = sample_subjects(labels, max_subjects=40, seed=1)
        b = sample_subjects(labels, max_subjects=40, seed=2)
        assert a != b

    def test_independent_of_mapping_insertion_order(self) -> None:
        """Same trap as partition_subjects: dict order must not change the draw."""
        base = make_subject_labels(100, impaired_fraction=0.3, seed=9)
        reversed_order = dict(reversed(list(base.items())))
        assert list(base) != list(reversed_order)

        seed = 42
        assert sample_subjects(base, max_subjects=35, seed=seed) == sample_subjects(
            reversed_order, max_subjects=35, seed=seed
        )

    def test_stratification_preserves_label_ratio(self) -> None:
        labels = make_subject_labels(200, impaired_fraction=0.25, seed=10)
        sampled = sample_subjects(labels, max_subjects=80, seed=3)
        counts = _labels_from_ids(labels, sampled)
        # 25% of 80 = 20; largest-remainder should land on 20.
        assert counts[Label.IMPAIRED] == 20
        assert counts[Label.COGNITIVELY_NORMAL] == 60

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            sample_subjects({}, max_subjects=10, seed=0)

    def test_negative_max_subjects_raises(self) -> None:
        labels = make_subject_labels(10, seed=11)
        with pytest.raises(ValueError, match="non-negative"):
            sample_subjects(labels, max_subjects=-1, seed=0)


def _population(n_impaired: int, n_normal: int) -> dict[SubjectId, Label]:
    """A cohort with an exact class composition, so the quotas are predictable."""
    labels: dict[SubjectId, Label] = {}
    for i in range(n_impaired):
        labels[SubjectId(f"IMP{i:04d}")] = Label.IMPAIRED
    for i in range(n_normal):
        labels[SubjectId(f"NRM{i:04d}")] = Label.COGNITIVELY_NORMAL
    return labels


class TestTheMinorityClassSurvivesTheDraw:
    """Largest-remainder allocation can round a rare class away entirely.

    Nothing downstream catches that. Partitioning succeeds, training succeeds, and the
    first symptom is an undefined ROC-AUC at evaluation — long after the sample was
    frozen and the scans preprocessed. So it is refused at the draw.
    """

    def test_a_sample_that_would_drop_a_class_is_refused(self) -> None:
        with pytest.raises(ValueError, match="IMPAIRED"):
            sample_subjects(_population(2, 98), max_subjects=10, seed=0)

    def test_the_error_reports_the_population_it_could_not_represent(self) -> None:
        """So the reader can tell "sample too small" from "cohort too imbalanced"."""
        with pytest.raises(ValueError, match=re.escape("2.0%")):
            sample_subjects(_population(2, 98), max_subjects=20, seed=0)

    def test_the_adr_0010_configuration_is_unaffected(self) -> None:
        """The realistic case must keep working: 300 subjects from ~1100 at 24%.

        A guard that fired on the configuration the project actually uses would be a
        bug, not a safeguard.
        """
        sampled = sample_subjects(_population(264, 836), max_subjects=300, seed=20261122)
        counts = _labels_from_ids(_population(264, 836), sampled)
        assert len(sampled) == 300
        assert counts[Label.IMPAIRED] == 72

    def test_a_single_class_population_is_passed_through(self) -> None:
        """Nothing to stratify and nothing to drop. Refusing here would be refusing a
        cohort that is merely unlabelled-for-one-class, which is a different problem."""
        assert len(sample_subjects(_population(0, 50), max_subjects=10, seed=0)) == 10

    def test_a_sample_large_enough_to_carry_the_minority_is_allowed(self) -> None:
        """The boundary: 2 impaired in 100, drawing 50, is exactly one impaired."""
        sampled = sample_subjects(_population(2, 98), max_subjects=50, seed=0)
        assert _labels_from_ids(_population(2, 98), sampled)[Label.IMPAIRED] == 1
