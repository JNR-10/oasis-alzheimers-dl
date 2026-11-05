"""Core domain types.

These mirror ``GLOSSARY.md`` one-for-one. A concept named there has a type here, and a
type here appears there — so that a reader can move between the vocabulary and the code
without translating.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import NewType

SubjectId = NewType("SubjectId", str)
"""Identifies a person. The unit of partitioning (ADR-0001)."""

SessionId = NewType("SessionId", str)
"""Identifies one acquisition occasion for a subject."""


class Partition(str, Enum):
    """Train, validation, or test. A subject belongs to exactly one (ADR-0001)."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Label(int, Enum):
    """The binary supervision target at a target visit."""

    COGNITIVELY_NORMAL = 0
    IMPAIRED = 1


@dataclass(frozen=True)
class Visit:
    """An imaging session that passed inclusion and carries a resolvable label.

    Frozen because a visit is a record of something that happened. Code that wants a
    modified visit is almost always code that should be building a new cohort instead.

    Attributes:
        subject_id: The person this visit belongs to.
        session_id: The acquisition occasion.
        days_from_entry: Days between the subject's study entry and this session. The
            raw quantity from which elapsed time is derived; absolute dates are not
            used, and OASIS does not release them.
        scan_path: Location of the preprocessed T1-weighted scan. Local and
            environment-specific; never committed (ADR-0008).
        label: The binary label, or None when this visit is usable as context in a
            sequence but not as a target visit.
        anatomy_targets: Structural quantities for auxiliary supervision, keyed by
            name. Training-only (ADR-0007); empty for test subjects without cost.
    """

    subject_id: SubjectId
    session_id: SessionId
    days_from_entry: int
    scan_path: Path
    label: Label | None = None
    anatomy_targets: dict[str, float] | None = None

    @property
    def is_target_visit(self) -> bool:
        """Whether this visit can serve as a target visit, i.e. whether it has a label."""
        raise NotImplementedError


@dataclass(frozen=True)
class VisitSequence:
    """A subject's visits in chronological order, ending at the target visit.

    The target visit is always last. Visits after it are excluded at construction
    rather than masked later — a sequence that physically cannot contain the future
    cannot leak it.

    Invariants to enforce in ``__post_init__`` when this is implemented: at least one
    visit, every visit from ``subject_id``, visits in chronological order, and the last
    visit labeled. Each one, violated, is a silent bug rather than a crash.

    Attributes:
        subject_id: The person. All visits share it.
        visits: Chronologically ordered, length >= 1, last element is the target visit.
    """

    subject_id: SubjectId
    visits: tuple[Visit, ...]

    @property
    def target_visit(self) -> Visit:
        """The visit whose label is predicted."""
        raise NotImplementedError

    @property
    def label(self) -> Label:
        """The sequence's label, taken from the target visit."""
        raise NotImplementedError

    @property
    def elapsed_days(self) -> tuple[int, ...]:
        """Days from each visit to the target visit, so the target is always 0.

        Negative values are earlier visits. Anchoring on the target rather than the
        baseline keeps the quantity meaningful when sequences are truncated, and keeps
        the model's reference point the thing it is predicting (ADR-0006).
        """
        raise NotImplementedError

    @property
    def is_longitudinal(self) -> bool:
        """Whether this sequence carries more than one visit."""
        raise NotImplementedError
