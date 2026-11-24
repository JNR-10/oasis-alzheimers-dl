"""The domain types' invariants and derived properties.

These are the cheapest place to catch a leak. A :class:`VisitSequence` that admits a
visit later than its target visit is a future leak that no split can see, so the
invariant is enforced at construction and tested here rather than trusted downstream.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from oasis_dl.types import Label, SessionId, SubjectId, Visit, VisitSequence


def make_visit(
    subject: str,
    day: int,
    label: Label | None = Label.COGNITIVELY_NORMAL,
) -> Visit:
    """A visit with the fields these tests care about and defaults for the rest."""
    return Visit(
        subject_id=SubjectId(subject),
        session_id=SessionId(f"{subject}_d{day:04d}"),
        days_from_entry=day,
        scan_path=Path(f"/nowhere/{subject}_d{day:04d}.nii.gz"),
        label=label,
    )


class TestVisit:
    def test_a_labeled_visit_can_be_a_target_visit(self) -> None:
        assert make_visit("OAS1", 0, Label.IMPAIRED).is_target_visit

    def test_an_unlabeled_visit_cannot_be_a_target_visit(self) -> None:
        assert not make_visit("OAS1", 0, None).is_target_visit


class TestVisitSequenceInvariants:
    def test_rejects_an_empty_sequence(self) -> None:
        with pytest.raises(ValueError, match="at least one visit"):
            VisitSequence(subject_id=SubjectId("OAS1"), visits=())

    def test_rejects_visits_from_another_subject(self) -> None:
        with pytest.raises(ValueError, match="subject"):
            VisitSequence(
                subject_id=SubjectId("OAS1"),
                visits=(make_visit("OAS1", 0), make_visit("OAS2", 100)),
            )

    def test_rejects_visits_out_of_chronological_order(self) -> None:
        with pytest.raises(ValueError, match="chronological"):
            VisitSequence(
                subject_id=SubjectId("OAS1"),
                visits=(make_visit("OAS1", 500), make_visit("OAS1", 100)),
            )

    def test_rejects_an_unlabeled_target_visit(self) -> None:
        with pytest.raises(ValueError, match="label"):
            VisitSequence(
                subject_id=SubjectId("OAS1"),
                visits=(make_visit("OAS1", 0), make_visit("OAS1", 100, None)),
            )

    def test_allows_unlabeled_context_visits_before_the_target(self) -> None:
        """A visit with no assessment is still usable as history."""
        sequence = VisitSequence(
            subject_id=SubjectId("OAS1"),
            visits=(
                make_visit("OAS1", 0, None),
                make_visit("OAS1", 400, Label.IMPAIRED),
            ),
        )
        assert len(sequence.visits) == 2


class TestVisitSequenceProperties:
    @pytest.fixture
    def sequence(self) -> VisitSequence:
        """Three visits at irregular spacing, target impaired at day 1200."""
        return VisitSequence(
            subject_id=SubjectId("OAS1"),
            visits=(
                make_visit("OAS1", 100, Label.COGNITIVELY_NORMAL),
                make_visit("OAS1", 400, Label.COGNITIVELY_NORMAL),
                make_visit("OAS1", 1200, Label.IMPAIRED),
            ),
        )

    def test_target_visit_is_the_last_one(self, sequence: VisitSequence) -> None:
        assert sequence.target_visit.days_from_entry == 1200

    def test_label_comes_from_the_target_visit(self, sequence: VisitSequence) -> None:
        assert sequence.label is Label.IMPAIRED

    def test_elapsed_days_are_anchored_on_the_target_visit(self, sequence: VisitSequence) -> None:
        """Target reads 0, earlier visits are negative (ADR-0006).

        Worked by hand from the fixture: 100 - 1200, 400 - 1200, 1200 - 1200.
        """
        assert sequence.elapsed_days == (-1100, -800, 0)

    def test_a_multi_visit_sequence_is_longitudinal(self, sequence: VisitSequence) -> None:
        assert sequence.is_longitudinal

    def test_a_single_visit_sequence_is_not_longitudinal(self) -> None:
        single = VisitSequence(
            subject_id=SubjectId("OAS1"),
            visits=(make_visit("OAS1", 100, Label.IMPAIRED),),
        )
        assert not single.is_longitudinal
        assert single.elapsed_days == (0,)
