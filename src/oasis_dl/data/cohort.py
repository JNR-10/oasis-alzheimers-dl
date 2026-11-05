"""Cohort construction: dataset sessions in, visit sequences out.

Unimplemented. Needs a real OASIS download to write against, and the label rule it
depends on is still open (ADR-0004). The interface is fixed here so that the modules
downstream of it can be built and tested in the meantime.

The dataset-specific parts sit behind :class:`CohortSource` so that the OASIS-2
fallback of ADR-0003 is a different implementation rather than a rewrite.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from oasis_dl.types import SubjectId, Visit, VisitSequence


@dataclass(frozen=True)
class InclusionRules:
    """The checkable conditions a session must meet to become a visit.

    Every field is a cohort parameter rather than a constant in the code, so that the
    cohort can be rebuilt under different rules and the difference attributed.

    Attributes:
        max_assessment_gap_days: How far a clinical assessment may be from an imaging
            session and still label it. Open parameter of ADR-0004.
        min_visits_for_longitudinal: Visits a subject needs to enter the longitudinal
            arm. The single-scan and longitudinal arms are compared on this same
            subset, so that the first core ablation varies architecture alone.
        require_preprocessing_success: Whether to drop sessions whose preprocessing
            failed. Always true in practice; present so the count of failures can be
            obtained by building the cohort both ways.
    """

    max_assessment_gap_days: int
    min_visits_for_longitudinal: int = 2
    require_preprocessing_success: bool = True


class CohortSource(Protocol):
    """A dataset that can yield sessions. Implemented once per dataset.

    The OASIS-2 fallback (ADR-0003) is a second implementation of this protocol, which
    is what makes the fallback a configuration change rather than a rewrite.
    """

    def sessions(self) -> pd.DataFrame:
        """Every session the dataset offers, before any inclusion rule is applied."""
        ...

    def scan_path(self, subject_id: SubjectId, session_id: str) -> Path:
        """Where the preprocessed T1-weighted scan for a session lives."""
        ...


def build_visits(
    source: CohortSource,
    rules: InclusionRules,
) -> tuple[list[Visit], pd.DataFrame]:
    """Apply the inclusion rules and return the surviving visits plus an audit trail.

    The audit trail is returned rather than logged because the count removed by each
    rule is reported in ``docs/data/cohort-definition.md``, and a number that has to be
    reconstructed from logs is a number that will be reconstructed wrong.

    Returns:
        The visits that passed, and one row per *input* session recording which rule
        removed it, if any.
    """
    raise NotImplementedError(
        "Blocked on OASIS-3 access and on ADR-0004 reaching accepted status. "
        "See docs/data/access-log.md."
    )


def build_sequences(
    visits: Iterable[Visit],
    *,
    max_length: int | None = None,
) -> list[VisitSequence]:
    """Group visits into one sequence per target visit.

    Every labeled visit becomes a target visit, and its sequence is that visit plus the
    subject's earlier visits. Later visits are excluded at construction — a sequence
    that cannot physically contain the future cannot leak it (see
    :class:`~oasis_dl.types.VisitSequence`).

    Args:
        visits: All visits in the cohort, in any order.
        max_length: Keep at most this many visits, counting back from the target.
            None keeps all of them. Bounds memory for subjects with long histories.

    Returns:
        One sequence per labeled visit, so a subject with three labeled visits
        contributes three correlated sequences. That correlation is handled by sampling
        or loss weighting, never by splitting the subject across partitions (ADR-0001).
    """
    raise NotImplementedError(
        "Blocked on build_visits. The grouping logic itself is dataset-independent and "
        "can be written as soon as the Visit stream is real."
    )


def summarize_cohort(sequences: Sequence[VisitSequence]) -> pd.DataFrame:
    """Produce the counts that fill docs/data/cohort-definition.md.

    Subjects, visits, and label balance per partition, plus the visit-count
    distribution that decides whether the temporal module has anything to learn from.
    """
    raise NotImplementedError("Blocked on build_sequences.")
