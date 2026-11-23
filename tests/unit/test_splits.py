"""Subject-level partitioning — ADR-0001, and the control the whole report rests on.

A bug here inflates every number without failing anything, so these tests are written
against the properties that matter rather than against an expected assignment. An
expected-assignment test passes on a split that is deterministic and wrong.

The detector gets the same treatment: `check_no_leakage` is tested by planting a leak,
because a leak check that cannot fail is worse than none — it reads as a guarantee.
"""

from __future__ import annotations

from collections import Counter

import pytest
from tests.helpers import make_sequence, make_sequences, make_subject_labels

from oasis_dl.data.splits import (
    DEFAULT_RATIOS,
    LeakageError,
    assign_sequences,
    check_no_leakage,
    partition_subjects,
    subject_label,
)
from oasis_dl.types import Label, Partition, SubjectId

RATIOS = (0.60, 0.20, 0.20)
"""The ADR-0010 ratios. DEFAULT_RATIOS differs; both must work."""


class TestSubjectLabel:
    def test_a_subject_who_ever_converts_counts_as_impaired(self) -> None:
        """Stratification only. A subject who progresses is an impaired subject for
        balancing, even though their early visits carry the normal label."""
        sequences = [
            make_sequence("OAS1", label=Label.COGNITIVELY_NORMAL),
            make_sequence("OAS1", label=Label.IMPAIRED),
        ]
        assert subject_label(sequences) is Label.IMPAIRED

    def test_a_consistently_normal_subject_stays_normal(self) -> None:
        sequences = [make_sequence("OAS1", label=Label.COGNITIVELY_NORMAL) for _ in range(3)]
        assert subject_label(sequences) is Label.COGNITIVELY_NORMAL

    def test_no_sequences_is_an_error_not_a_default_label(self) -> None:
        """Defaulting would quietly stratify an empty subject as normal."""
        with pytest.raises(ValueError):
            subject_label([])


class TestDeterminism:
    def test_the_same_seed_gives_the_same_assignment(self) -> None:
        labels = make_subject_labels(n_subjects=60, seed=1)
        first = partition_subjects(labels, ratios=RATIOS, seed=42)
        second = partition_subjects(labels, ratios=RATIOS, seed=42)
        assert first == second

    def test_a_different_seed_gives_a_different_assignment(self) -> None:
        labels = make_subject_labels(n_subjects=60, seed=1)
        a = partition_subjects(labels, ratios=RATIOS, seed=42)
        b = partition_subjects(labels, ratios=RATIOS, seed=43)
        assert a != b

    def test_insertion_order_cannot_change_the_split(self) -> None:
        """The quiet one. `make_subject_labels` deliberately shuffles key order, so a
        partitioner that iterates the mapping instead of sorting fails here and nowhere
        else — and the split would then depend on how a caller happened to build a dict."""
        labels = make_subject_labels(n_subjects=80, seed=7)
        reordered = dict(reversed(list(labels.items())))
        # Dicts compare equal regardless of key order, so compare the key *sequences*.
        assert list(reordered) != list(labels), "the fixture did not actually reorder"
        assert partition_subjects(labels, ratios=RATIOS, seed=42) == partition_subjects(
            reordered, ratios=RATIOS, seed=42
        )


class TestCoverageAndRatios:
    def test_every_subject_is_assigned_exactly_once(self) -> None:
        labels = make_subject_labels(n_subjects=97, seed=3)
        assignment = partition_subjects(labels, ratios=RATIOS, seed=42)
        assert set(assignment) == set(labels)

    @pytest.mark.parametrize("ratios", [RATIOS, DEFAULT_RATIOS, (0.8, 0.1, 0.1)])
    def test_partition_sizes_track_the_requested_ratios(
        self, ratios: tuple[float, float, float]
    ) -> None:
        labels = make_subject_labels(n_subjects=300, seed=3)
        counts = Counter(partition_subjects(labels, ratios=ratios, seed=42).values())
        for partition, ratio in zip(Partition, ratios, strict=True):
            assert abs(counts[partition] / 300 - ratio) < 0.03

    def test_a_cohort_too_small_to_fill_every_partition_is_refused(self) -> None:
        """Silently returning an empty test partition would be discovered at evaluation."""
        labels = make_subject_labels(n_subjects=2, seed=0)
        with pytest.raises(ValueError):
            partition_subjects(labels, ratios=RATIOS, seed=42)

    def test_an_empty_cohort_is_refused(self) -> None:
        with pytest.raises(ValueError):
            partition_subjects({}, ratios=RATIOS, seed=42)

    @pytest.mark.parametrize("ratios", [(0.5, 0.2, 0.2), (0.6, 0.2, 0.3), (-0.1, 0.5, 0.6)])
    def test_ratios_that_do_not_form_a_partition_are_refused(
        self, ratios: tuple[float, float, float]
    ) -> None:
        labels = make_subject_labels(n_subjects=50, seed=0)
        with pytest.raises(ValueError):
            partition_subjects(labels, ratios=ratios, seed=42)


class TestStratification:
    def test_each_partition_carries_roughly_the_cohort_class_balance(self) -> None:
        """Approximate by construction: balanced at subject level while training uses
        the visit-level label. The difference is reported, not corrected."""
        labels = make_subject_labels(n_subjects=300, impaired_fraction=0.24, seed=5)
        assignment = partition_subjects(labels, ratios=RATIOS, seed=42)
        for partition in Partition:
            members = [s for s, p in assignment.items() if p is partition]
            impaired = sum(labels[s] is Label.IMPAIRED for s in members)
            assert abs(impaired / len(members) - 0.24) < 0.08

    def test_the_minority_class_reaches_every_partition(self) -> None:
        """An unstratified draw can empty the minority class out of `test`, and ROC-AUC
        on a single-class partition is undefined — a late, confusing failure."""
        labels = make_subject_labels(n_subjects=50, impaired_fraction=0.2, seed=11)
        assignment = partition_subjects(labels, ratios=RATIOS, seed=42)
        for partition in Partition:
            members = [s for s, p in assignment.items() if p is partition]
            assert any(labels[s] is Label.IMPAIRED for s in members)


class TestAssignSequences:
    def test_sequences_are_routed_to_their_subjects_partition(self) -> None:
        sequences = make_sequences(n_subjects=40, seed=2)
        labels = {
            s: subject_label([q for q in sequences if q.subject_id == s])
            for s in {q.subject_id for q in sequences}
        }
        assignment = partition_subjects(labels, ratios=RATIOS, seed=42)
        grouped = assign_sequences(sequences, assignment)
        for partition, members in grouped.items():
            for sequence in members:
                assert assignment[sequence.subject_id] is partition

    def test_every_partition_key_is_present_even_when_empty(self) -> None:
        """A missing key is an AttributeError at the dataloader, far from the cause."""
        sequences = [make_sequence("OAS1")]
        grouped = assign_sequences(sequences, {SubjectId("OAS1"): Partition.TRAIN})
        assert set(grouped) == set(Partition)

    def test_no_sequence_is_lost(self) -> None:
        sequences = make_sequences(n_subjects=30, seed=4)
        labels = {
            s: subject_label([q for q in sequences if q.subject_id == s])
            for s in {q.subject_id for q in sequences}
        }
        grouped = assign_sequences(sequences, partition_subjects(labels, ratios=RATIOS, seed=42))
        assert sum(len(v) for v in grouped.values()) == len(sequences)

    def test_an_unassigned_subject_is_an_error_not_a_dropped_row(self) -> None:
        """Dropping it would shrink the cohort silently, and the count would still
        look plausible."""
        sequences = [make_sequence("OAS1"), make_sequence("OAS2")]
        with pytest.raises(LeakageError, match="OAS2"):
            assign_sequences(sequences, {SubjectId("OAS1"): Partition.TRAIN})


class TestLeakageDetector:
    def test_a_clean_split_passes(self) -> None:
        sequences = make_sequences(n_subjects=30, seed=6)
        labels = {
            s: subject_label([q for q in sequences if q.subject_id == s])
            for s in {q.subject_id for q in sequences}
        }
        grouped = assign_sequences(sequences, partition_subjects(labels, ratios=RATIOS, seed=42))
        check_no_leakage(grouped)

    def test_a_planted_leak_is_caught_and_the_subject_is_named(self) -> None:
        """The detector is tested, not assumed. A check that cannot fail reads as a
        guarantee and provides none."""
        leaked = make_sequence("OAS_LEAK")
        grouped = {
            Partition.TRAIN: [leaked, make_sequence("OAS1")],
            Partition.VAL: [make_sequence("OAS2")],
            Partition.TEST: [leaked],
        }
        with pytest.raises(LeakageError, match="OAS_LEAK"):
            check_no_leakage(grouped)

    def test_the_error_names_the_partitions_involved(self) -> None:
        leaked = make_sequence("OAS_LEAK")
        grouped = {
            Partition.TRAIN: [leaked],
            Partition.VAL: [],
            Partition.TEST: [leaked],
        }
        with pytest.raises(LeakageError, match=r"train.*test|test.*train"):
            check_no_leakage(grouped)

    def test_a_subject_appearing_twice_in_one_partition_is_not_a_leak(self) -> None:
        """Normal: a subject contributes several sequences to its own partition."""
        check_no_leakage(
            {
                Partition.TRAIN: [make_sequence("OAS1"), make_sequence("OAS1")],
                Partition.VAL: [make_sequence("OAS2")],
                Partition.TEST: [make_sequence("OAS3")],
            }
        )
