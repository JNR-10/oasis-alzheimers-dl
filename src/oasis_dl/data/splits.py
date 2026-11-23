"""Subject-level partitioning — the evaluation contract of ADR-0001.

The project's load-bearing methodological control. A bug here inflates every reported
number without failing anything, so this is the first module to implement and the first
to test.
"""

from __future__ import annotations

import random
from collections import defaultdict
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

    Raises:
        ValueError: If there are no sequences. Defaulting an empty subject to
            "cognitively normal" would stratify a subject that does not exist.
    """
    labels = [sequence.label for sequence in sequences]
    if not labels:
        raise ValueError("cannot derive a subject label from no sequences")
    return (
        Label.IMPAIRED
        if any(label is Label.IMPAIRED for label in labels)
        else (Label.COGNITIVELY_NORMAL)
    )


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
        ValueError: If ratios are invalid, the cohort is empty, or the cohort is too
            small to fill every partition.
    """
    _validate_ratios(ratios)
    if not subject_labels:
        raise ValueError("cannot partition an empty cohort")

    # Stratify by drawing each class separately, so the minority class lands in every
    # partition in roughly its cohort proportion. An unstratified draw can empty it out
    # of `test`, where it surfaces much later as an undefined ROC-AUC.
    strata: dict[Label, list[SubjectId]] = defaultdict(list)
    for subject in sorted(subject_labels):
        strata[subject_labels[subject]].append(subject)

    assignment: dict[SubjectId, Partition] = {}
    for label in sorted(strata, key=lambda value: value.value):
        members = strata[label]
        # Sorted above, then shuffled from the seed. Iterating the mapping directly
        # would make the split depend on how the caller happened to build its dict —
        # reproducible on one machine and not on another.
        random.Random(f"{seed}-{label.value}").shuffle(members)
        start = 0
        for partition, size in zip(Partition, _allocate(len(members), ratios), strict=True):
            for subject in members[start : start + size]:
                assignment[subject] = partition
            start += size

    empty = [p.value for p in Partition if not any(v is p for v in assignment.values())]
    if empty:
        raise ValueError(
            f"a cohort of {len(subject_labels)} subjects split {ratios} leaves "
            f"{', '.join(empty)} empty. An empty partition is discovered at evaluation, "
            f"long after the run that produced it."
        )
    return assignment


def _validate_ratios(ratios: tuple[float, float, float]) -> None:
    """Refuse anything that is not a partition of the cohort."""
    if len(ratios) != len(Partition):
        raise ValueError(f"expected {len(Partition)} ratios, got {len(ratios)}")
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError(f"every ratio must be positive, got {ratios}")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1, got {ratios} summing to {sum(ratios)}")


def _allocate(n: int, ratios: tuple[float, float, float]) -> list[int]:
    """Split ``n`` across the partitions by largest remainder.

    Truncating each share would lose up to two subjects per stratum to rounding, and
    they would vanish from the cohort without appearing in any count.
    """
    exact = [n * ratio for ratio in ratios]
    sizes = [int(value) for value in exact]
    order = sorted(range(len(ratios)), key=lambda i: exact[i] - sizes[i], reverse=True)
    for index in order[: n - sum(sizes)]:
        sizes[index] += 1
    return sizes


def assign_sequences(
    sequences: Iterable[VisitSequence],
    assignment: Mapping[SubjectId, Partition],
) -> dict[Partition, list[VisitSequence]]:
    """Route each sequence to its subject's partition.

    A sequence whose subject has no assignment is a :class:`LeakageError`, not a
    silently dropped row and not a guessed partition.

    Returns:
        Sequences grouped by partition. Every partition key present, possibly empty.

    Raises:
        LeakageError: If a sequence's subject has no assignment. Dropping it would
            shrink the cohort silently and every downstream count would still look
            plausible.
    """
    grouped: dict[Partition, list[VisitSequence]] = {partition: [] for partition in Partition}
    for sequence in sequences:
        partition = assignment.get(sequence.subject_id)
        if partition is None:
            raise LeakageError(
                f"subject {sequence.subject_id} has a sequence but no partition "
                f"assignment. It was either added after the split was drawn or lost "
                f"from it; either way the split no longer covers the cohort."
            )
        grouped[partition].append(sequence)
    return grouped


def check_no_leakage(grouped: Mapping[Partition, Iterable[VisitSequence]]) -> None:
    """Verify that no subject appears in more than one partition.

    Called after the split is built and again after the dataloaders are constructed, on
    the principle that the guarantee is worth re-checking at every point an intervening
    transformation could have broken it.

    Raises:
        LeakageError: Naming every offending subject and the partitions involved.
    """
    homes: dict[SubjectId, set[str]] = defaultdict(set)
    for partition, sequences in grouped.items():
        for sequence in sequences:
            homes[sequence.subject_id].add(partition.value)

    leaked = {subject: sorted(where) for subject, where in homes.items() if len(where) > 1}
    if leaked:
        detail = "; ".join(
            f"{subject} in {', '.join(where)}" for subject, where in sorted(leaked.items())
        )
        raise LeakageError(
            f"{len(leaked)} subject(s) span more than one partition: {detail}. "
            f"Every number computed from this split is invalid (ADR-0001)."
        )
