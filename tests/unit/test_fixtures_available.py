"""Proves the shared fixtures work for a teammate who imports nothing of mine."""

from oasis_dl.data.synthetic import SyntheticCohort
from oasis_dl.types import VisitSequence


def test_a_teammate_can_use_the_cohort_fixture(synthetic_cohort: SyntheticCohort) -> None:
    assert len(synthetic_cohort.sessions()) > 0


def test_a_teammate_can_use_the_sequences_fixture(
    synthetic_sequences: list[VisitSequence],
) -> None:
    assert all(len(s.visits) <= 3 for s in synthetic_sequences)
    assert any(s.is_longitudinal for s in synthetic_sequences)
