"""In-memory builders for cohort shapes, so a test needs no scans on disk.

Most of what this project gets wrong is wrong before a single voxel is read. Subject
partitioning, leakage checks, metric aggregation and provenance all operate on
*records* — who, when, which partition, which label — and none of them needs pixels.
Building those records here keeps those tests fast and keeps them honest, because a
test that cannot read a scan cannot accidentally pass by reading one.

``scan_path`` therefore points at a path that does not exist. If your code opens it,
your code is doing something these helpers were not built for, and the failure is the
useful kind.

Use these for:

- **#4** metrics and the subject-level bootstrap — needs labels and subject ids
- **#6** subject-level splits and leakage tests — needs subjects, labels, sequences
- **#12** provenance and run records — needs a cohort to describe

For tests that genuinely need scans on disk — preprocessing, datasets, collation,
anything that trains — use the generated cohort fixtures instead. They write real
NIfTI volumes and are the right tool once pixels are involved.
"""

from __future__ import annotations

import random
from pathlib import Path

from oasis_dl.types import Label, SessionId, SubjectId, Visit, VisitSequence

__all__ = [
    "make_sequence",
    "make_sequences",
    "make_subject_labels",
    "make_visit",
]

NOWHERE = Path("/nonexistent/synthetic")
"""Where these records claim their scans live. Nothing is there, on purpose."""


def make_visit(
    subject: str,
    day: int,
    label: Label | None = Label.COGNITIVELY_NORMAL,
) -> Visit:
    """One visit, with the fields record-level tests care about.

    Args:
        subject: Subject id. Plain ``str`` for convenience at call sites.
        day: Days from the subject's study entry.
        label: ``None`` makes this a context visit — usable inside a sequence but never
            as a target visit.
    """
    return Visit(
        subject_id=SubjectId(subject),
        session_id=SessionId(f"{subject}_d{day:04d}"),
        days_from_entry=day,
        scan_path=NOWHERE / f"{subject}_d{day:04d}.nii.gz",
        label=label,
    )


def make_sequence(
    subject: str,
    days: tuple[int, ...] = (0, 400, 900),
    label: Label = Label.COGNITIVELY_NORMAL,
) -> VisitSequence:
    """A subject's visits ending at a labeled target visit.

    ``days`` must be ascending — the last visit is the target visit, and a sequence
    whose last element is not its latest visit is a future leak rather than a sorting
    detail.

    Only the final visit carries ``label``; earlier ones are context. That mirrors the
    real cohort, where a session without an assessment inside the admissible gap is
    still usable history.
    """
    if not days:
        raise ValueError("a sequence needs at least one visit")
    if list(days) != sorted(days):
        raise ValueError(f"days must be ascending, got {days}")

    visits = [make_visit(subject, day, None) for day in days[:-1]]
    visits.append(make_visit(subject, days[-1], label))
    return VisitSequence(subject_id=SubjectId(subject), visits=tuple(visits))


def make_subject_labels(
    n_subjects: int = 100,
    impaired_fraction: float = 0.24,
    seed: int = 0,
) -> dict[SubjectId, Label]:
    """A subject-to-label mapping for stratification and partitioning tests.

    Deterministic in ``seed``. The default balance is near the predecessor's 76/24
    split, so a partitioner that silently ignores stratification shows up as a skew
    rather than passing on a cohort that was balanced anyway.

    **Insertion order is deliberately shuffled.** A partitioner must be deterministic
    in its seed *and independent of mapping order*; one that sorts its keys passes
    either way, one that does not will disagree with itself when the caller builds the
    dict differently.
    """
    if n_subjects < 1:
        raise ValueError("n_subjects must be at least 1")
    if not 0.0 <= impaired_fraction <= 1.0:
        raise ValueError("impaired_fraction must be a proportion")

    rng = random.Random(seed)
    n_impaired = round(n_subjects * impaired_fraction)
    labels = [Label.IMPAIRED] * n_impaired + [Label.COGNITIVELY_NORMAL] * (n_subjects - n_impaired)
    rng.shuffle(labels)

    ids = [SubjectId(f"SUB{i:04d}") for i in range(n_subjects)]
    rng.shuffle(ids)
    return dict(zip(ids, labels, strict=True))


def make_sequences(
    n_subjects: int = 20,
    impaired_fraction: float = 0.24,
    max_visits: int = 3,
    seed: int = 0,
) -> list[VisitSequence]:
    """A small cohort of sequences, one per subject, at irregular spacing.

    Visit counts and gaps vary, because a cohort on a uniform grid lets code that
    ignores elapsed time look correct (ADR-0006).

    One sequence per subject keeps the subject-to-sequence mapping one-to-one, which is
    what you want while testing a partitioner. The real cohort yields several
    correlated sequences per subject; reach for the generated cohort fixtures when that
    correlation is the thing under test.
    """
    rng = random.Random(seed)
    labels = make_subject_labels(n_subjects, impaired_fraction, seed)

    sequences: list[VisitSequence] = []
    for subject_id, label in labels.items():
        n_visits = rng.randint(1, max_visits)
        day = rng.randint(0, 120)
        days = [day]
        for _ in range(n_visits - 1):
            day += rng.randint(150, 900)
            days.append(day)
        sequences.append(make_sequence(str(subject_id), tuple(days), label))
    return sequences
