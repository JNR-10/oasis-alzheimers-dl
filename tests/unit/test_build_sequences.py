"""Tests for :func:`oasis_dl.data.cohort.build_sequences` (issue #14).

Grouping is dataset-independent: given a Visit stream, every labeled visit becomes a
target and its history cannot contain the future.
"""

from __future__ import annotations

import pytest
from tests.helpers import make_visit

from oasis_dl.data.cohort import build_sequences
from oasis_dl.types import Label, SubjectId


class TestBuildSequences:
    def test_one_sequence_per_labeled_visit(self) -> None:
        visits = [
            make_visit("S1", 0, None),
            make_visit("S1", 400, Label.COGNITIVELY_NORMAL),
            make_visit("S1", 900, Label.IMPAIRED),
        ]
        sequences = build_sequences(visits)
        assert len(sequences) == 2
        assert [len(s.visits) for s in sequences] == [2, 3]
        assert sequences[0].visits[-1].days_from_entry == 400
        assert sequences[1].visits[-1].days_from_entry == 900

    def test_later_visits_are_excluded_from_earlier_targets(self) -> None:
        visits = [
            make_visit("S1", 0, Label.COGNITIVELY_NORMAL),
            make_visit("S1", 400, Label.IMPAIRED),
        ]
        early = build_sequences(visits)[0]
        assert [v.days_from_entry for v in early.visits] == [0]
        assert early.visits[-1].label is Label.COGNITIVELY_NORMAL

    def test_unlabeled_only_subject_yields_nothing(self) -> None:
        visits = [make_visit("S1", 0, None), make_visit("S1", 100, None)]
        assert build_sequences(visits) == []

    def test_max_length_keeps_the_most_recent_visits(self) -> None:
        visits = [
            make_visit("S1", 0, None),
            make_visit("S1", 200, None),
            make_visit("S1", 400, None),
            make_visit("S1", 900, Label.IMPAIRED),
        ]
        sequences = build_sequences(visits, max_length=2)
        assert len(sequences) == 1
        assert [v.days_from_entry for v in sequences[0].visits] == [400, 900]

    def test_max_length_none_keeps_full_history(self) -> None:
        visits = [
            make_visit("S1", 0, None),
            make_visit("S1", 400, None),
            make_visit("S1", 900, Label.IMPAIRED),
        ]
        sequences = build_sequences(visits, max_length=None)
        assert [v.days_from_entry for v in sequences[0].visits] == [0, 400, 900]

    def test_input_order_does_not_change_output(self) -> None:
        visits = [
            make_visit("S2", 50, Label.IMPAIRED),
            make_visit("S1", 900, Label.IMPAIRED),
            make_visit("S1", 0, None),
            make_visit("S2", 0, None),
        ]
        reversed_visits = list(reversed(visits))
        left = build_sequences(visits)
        right = build_sequences(reversed_visits)
        assert [s.subject_id for s in left] == [s.subject_id for s in right]
        assert [[v.days_from_entry for v in s.visits] for s in left] == [
            [v.days_from_entry for v in s.visits] for s in right
        ]

    def test_sequences_are_ordered_by_subject_then_target_day(self) -> None:
        visits = [
            make_visit("S2", 10, Label.IMPAIRED),
            make_visit("S1", 500, Label.COGNITIVELY_NORMAL),
            make_visit("S1", 100, Label.COGNITIVELY_NORMAL),
        ]
        sequences = build_sequences(visits)
        keys = [(s.subject_id, s.visits[-1].days_from_entry) for s in sequences]
        assert keys == sorted(keys)

    def test_multiple_subjects_stay_separated(self) -> None:
        visits = [
            make_visit("S1", 0, Label.COGNITIVELY_NORMAL),
            make_visit("S2", 0, Label.IMPAIRED),
        ]
        sequences = build_sequences(visits)
        assert {s.subject_id for s in sequences} == {
            SubjectId("S1"),
            SubjectId("S2"),
        }
        assert all(v.subject_id == s.subject_id for s in sequences for v in s.visits)

    def test_max_length_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            build_sequences([make_visit("S1", 0, Label.IMPAIRED)], max_length=0)

    def test_empty_input_returns_empty(self) -> None:
        assert build_sequences([]) == []
