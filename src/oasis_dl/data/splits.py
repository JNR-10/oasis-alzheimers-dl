"""Subject-level partitioning — the evaluation contract of ADR-0001.

The project's load-bearing methodological control. A bug here inflates every reported
number without failing anything, so this is the first module to implement and the first
to test.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from oasis_dl.types import Label, Partition, SubjectId, VisitSequence

DEFAULT_RATIOS: tuple[float, float, float] = (0.70, 0.15, 0.15)


class LeakageError(AssertionError):
    """A subject's data was found in more than one partition.

    Raised rather than warned. A leak invalidates every downstream number, so the only
    safe response is to stop.
    """


def subject_label(sequences: Iterable[VisitSequence]) -> Label:
    """Collapse a subject's sequences to one label, for stratification only.

    A subject counts as impaired if **any** of their target visits is impaired, since a
    subject who progresses is an impaired subject for balancing purposes even if their
    early visits are normal.

    This affects only how subjects are spread across partitions. Training and evaluation
    always use the per-visit label from the target visit.
    """
    raise NotImplementedError


def partition_subjects(
    subject_labels: Mapping[SubjectId, Label],
    *,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int,
) -> dict[SubjectId, Partition]:
    """Assign each subject to exactly one partition, stratified by label.

    Must be deterministic in the seed **and independent of mapping insertion order** —
    sort subject ids before shuffling, or the split silently depends on how the caller
    happened to build its dict.

    Partitions are drawn once and recorded in the manifest (ADR-0001). This is not
    re-run to "improve" a split after seeing results.

    Stratification is approximate: it balances labels at *subject* level while the label
    used in training lives at *visit* level, so per-visit class balance will differ
    between partitions. That difference is reported, not corrected by moving subjects.

    Args:
        subject_labels: Every subject in the cohort, with its stratification label.
        ratios: Train, validation, test proportions. Must sum to 1.
        seed: Recorded in the manifest alongside the assignment.

    Returns:
        One partition per subject, covering every key of ``subject_labels``.

    Raises:
        ValueError: If ratios are invalid or the cohort is empty.
    """
    raise NotImplementedError


def assign_sequences(
    sequences: Iterable[VisitSequence],
    assignment: Mapping[SubjectId, Partition],
) -> dict[Partition, list[VisitSequence]]:
    """Route each sequence to its subject's partition.

    A sequence whose subject has no assignment is a :class:`LeakageError`, not a
    silently dropped row and not a guessed partition.

    Returns:
        Sequences grouped by partition. Every partition key present, possibly empty.
    """
    raise NotImplementedError


def check_no_leakage(grouped: Mapping[Partition, Iterable[VisitSequence]]) -> None:
    """Verify that no subject appears in more than one partition.

    Called after the split is built and again after the dataloaders are constructed, on
    the principle that the guarantee is worth re-checking at every point an intervening
    transformation could have broken it.

    Raises:
        LeakageError: Naming every offending subject and the partitions involved.
    """
    raise NotImplementedError
