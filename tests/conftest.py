"""Fixtures shared across the test suite.

The synthetic cohort lives here rather than in one test module because every ticket
downstream of #3 needs a cohort to test against, and generating one per test is slow
enough to discourage writing tests at all.

The cohort fixtures are **session-scoped**: generated once per pytest run and reused.
Treat them as read-only. A test that needs its own cohort — a different label balance,
no signal, a specific seed — builds one from :class:`SyntheticCohortSpec` directly
instead of mutating these.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from oasis_dl.data.synthetic import SyntheticCohort, SyntheticCohortSpec
from oasis_dl.types import VisitSequence


@pytest.fixture(scope="session")
def synthetic_spec() -> SyntheticCohortSpec:
    """The default cohort shape: small, irregular, 24% impaired, tiny volumes."""
    return SyntheticCohortSpec(n_subjects=32, seed=20261122)


@pytest.fixture(scope="session")
def synthetic_cohort(
    tmp_path_factory: pytest.TempPathFactory,
    synthetic_spec: SyntheticCohortSpec,
) -> SyntheticCohort:
    """A generated cohort with its volumes on disk. Read-only; shared by every test."""
    root: Path = tmp_path_factory.mktemp("synthetic_cohort")
    return SyntheticCohort(root=root, spec=synthetic_spec).generate()


@pytest.fixture(scope="session")
def synthetic_sequences(synthetic_cohort: SyntheticCohort) -> list[VisitSequence]:
    """Every sequence the default cohort yields, capped at the ADR-0010 length of 3."""
    return synthetic_cohort.sequences(max_length=3)


@pytest.fixture(scope="session")
def longitudinal_sequences(
    synthetic_sequences: list[VisitSequence],
) -> list[VisitSequence]:
    """Only the sequences with more than one visit.

    The subset the temporal module has anything to read. Both arms of ablation 1 are
    compared on the *same* subjects, so this is for testing the module in isolation,
    never for building a cohort an arm is evaluated on.
    """
    return [s for s in synthetic_sequences if s.is_longitudinal]
