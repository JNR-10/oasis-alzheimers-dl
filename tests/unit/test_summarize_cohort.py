"""Tests for :func:`oasis_dl.data.cohort.summarize_cohort` (issue #14)."""

from __future__ import annotations

import pytest
from tests.helpers import make_sequence, make_visit

from oasis_dl.data.cohort import build_sequences, summarize_cohort
from oasis_dl.types import Label, Partition, SubjectId, Visit


def _lookup(frame, section: str, key: str) -> int:
    rows = frame[(frame["section"] == section) & (frame["key"] == key)]
    assert len(rows) == 1, f"expected one row for {section}/{key}, got {len(rows)}"
    return int(rows.iloc[0]["value"])


class TestSummarizeCohort:
    def test_totals_and_visit_count_distribution(self) -> None:
        sequences = [
            make_sequence("S1", (0, 400), Label.COGNITIVELY_NORMAL),
            make_sequence("S1", (0, 400, 900), Label.IMPAIRED),
            make_sequence("S2", (0,), Label.COGNITIVELY_NORMAL),
        ]
        summary = summarize_cohort(sequences)
        assert _lookup(summary, "totals", "subjects") == 2
        assert _lookup(summary, "totals", "sequences") == 3
        # S1 has visits at 0,400,900; S2 has visit at 0.
        assert _lookup(summary, "totals", "visits") == 4
        assert _lookup(summary, "visit_count", "1") == 1
        assert _lookup(summary, "visit_count", "3") == 1

    def test_subject_label_is_impaired_if_any_target_is(self) -> None:
        sequences = [
            make_sequence("S1", (0,), Label.COGNITIVELY_NORMAL),
            make_sequence("S1", (0, 500), Label.IMPAIRED),
            make_sequence("S2", (0,), Label.COGNITIVELY_NORMAL),
        ]
        summary = summarize_cohort(sequences)
        assert _lookup(summary, "label", Label.IMPAIRED.name) == 1
        assert _lookup(summary, "label", Label.COGNITIVELY_NORMAL.name) == 1

    def test_partition_rows_require_a_complete_assignment(self) -> None:
        sequences = [
            make_sequence("S1", (0,), Label.IMPAIRED),
            make_sequence("S2", (0,), Label.COGNITIVELY_NORMAL),
        ]
        assignment = {
            SubjectId("S1"): Partition.TRAIN,
            SubjectId("S2"): Partition.TEST,
        }
        summary = summarize_cohort(sequences, assignment=assignment)
        assert _lookup(summary, "partition_train", "subjects") == 1
        assert _lookup(summary, "partition_train", Label.IMPAIRED.name) == 1
        assert _lookup(summary, "partition_test", "subjects") == 1
        assert _lookup(summary, "partition_val", "subjects") == 0

    def test_assignment_mismatch_raises(self) -> None:
        sequences = [make_sequence("S1", (0,), Label.IMPAIRED)]
        with pytest.raises(ValueError, match="unassigned"):
            summarize_cohort(
                sequences,
                assignment={SubjectId("S2"): Partition.TRAIN},
            )

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            summarize_cohort([])

    def test_works_on_builder_output(self) -> None:
        visits: list[Visit] = [
            make_visit("S1", 0, None),
            make_visit("S1", 400, Label.COGNITIVELY_NORMAL),
            make_visit("S2", 0, Label.IMPAIRED),
        ]
        summary = summarize_cohort(build_sequences(visits))
        assert _lookup(summary, "totals", "subjects") == 2
        assert _lookup(summary, "totals", "sequences") == 2
