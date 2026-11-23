"""Cohort construction: dataset sessions in, visit sequences out.

Unimplemented. The label rule it depends on is still open (ADR-0004); the download
itself has arrived. The interface is fixed here so that the modules downstream of it
can be built and tested against the synthetic cohort in the meantime.

The dataset-specific parts sit behind :class:`CohortSource` so that the OASIS-2
fallback of ADR-0003 is a different implementation rather than a rewrite.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from oasis_dl.types import Label, SubjectId, Visit, VisitSequence


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

    OASIS-3 is the primary implementation; the OASIS-2 fallback (ADR-0003) is a second
    one, which is what makes a fallback a configuration change rather than a rewrite.

    **An implementation must not expose a cognitive score as a model-readable column.**
    MMSE may label and may describe the cohort, but a source that hands it downstream
    alongside the imaging columns is one careless feature list away from the
    circularity this project exists to avoid (ADR-0002). Drop it at the adapter.
    """

    def sessions(self) -> pd.DataFrame:
        """Every session the dataset offers, before any inclusion rule is applied.

        Metadata only — no imaging is read. This is what makes it affordable to decide
        which subjects to sample before anything is preprocessed.
        """
        ...

    def scan_path(self, subject_id: SubjectId, session_id: str) -> Path:
        """Where the preprocessed T1-weighted scan for a session lives."""
        ...


def sample_subjects(
    subject_labels: Mapping[SubjectId, Label],
    *,
    max_subjects: int | None,
    seed: int,
) -> list[SubjectId]:
    """Draw the fixed subject subsample of ADR-0010, stratified by label.

    **Call this before preprocessing, not after.** The point of sampling is to avoid
    processing scans that will never be trained on, and most of the one-time CPU cost
    of the project sits in the stages this call skips. The ordering is:

        sessions -> eligible subjects -> *sample* -> preprocess -> visits -> partition

    Stratified so that a small draw cannot wander away from the cohort's class balance
    by luck. This is a sample of the *population*, not a balancing intervention: the
    label ratio is preserved, not corrected.

    Must be deterministic in the seed **and independent of mapping insertion order** —
    sort subject ids before shuffling, or the sample silently depends on how the caller
    happened to build its dict. Same failure as in
    :func:`~oasis_dl.data.splits.partition_subjects`, and the same fix.

    The seed is distinct from the partition seed and recorded separately in the
    manifest. They answer different questions, and conflating them makes it impossible
    to say later which draw a cohort came from.

    Args:
        subject_labels: Every subject eligible before sampling, with its label.
        max_subjects: How many to keep. ``None`` keeps all of them, which is the
            OASIS-2 case and the behaviour to restore if compute stops binding.
        seed: Recorded in the manifest alongside the sampled ids.

    Returns:
        The sampled subject ids, sorted, so the caller cannot accidentally depend on
        draw order. Returns every candidate when ``max_subjects`` is None or exceeds
        the candidate count — asking for more subjects than exist is not an error.

    Raises:
        ValueError: If ``max_subjects`` is negative or ``subject_labels`` is empty.
    """
    if not subject_labels:
        raise ValueError("subject_labels is empty — nothing to sample")
    if max_subjects is not None and max_subjects < 0:
        raise ValueError(f"max_subjects must be non-negative, got {max_subjects}")

    # Sort first so dict insertion order cannot change the draw (ADR-0010 / ADR-0001).
    candidates = sorted(subject_labels)
    n = len(candidates)
    if max_subjects is None or max_subjects >= n:
        return candidates

    if max_subjects == 0:
        return []

    by_label: dict[Label, list[SubjectId]] = defaultdict(list)
    for subject_id in candidates:
        by_label[subject_labels[subject_id]].append(subject_id)

    # Largest-remainder allocation keeps the sample's label ratio near the population's
    # without overshooting a class that has few subjects.
    quotas: dict[Label, int] = {}
    remainders: list[tuple[float, Label]] = []
    allocated = 0
    for label in sorted(by_label, key=lambda lab: lab.value):
        exact = max_subjects * len(by_label[label]) / n
        floor = int(exact)
        quotas[label] = floor
        allocated += floor
        remainders.append((exact - floor, label))
    remainders.sort(key=lambda item: (-item[0], item[1].value))
    for i in range(max_subjects - allocated):
        quotas[remainders[i][1]] += 1

    # A class the population has but the sample does not is not a smaller cohort, it is
    # a different study. Nothing downstream catches it: partitioning succeeds, training
    # succeeds, and the first symptom is an undefined ROC-AUC at evaluation, long after
    # the sample was frozen and the scans preprocessed (ADR-0010).
    dropped = sorted(
        (label for label, pool in by_label.items() if pool and quotas[label] == 0),
        key=lambda lab: lab.value,
    )
    # No `len(by_label) > 1` guard: a population of one class takes the whole quota, so
    # `dropped` is empty there by construction rather than by a condition no test could
    # tell apart from its absence.
    if dropped:
        shares = ", ".join(
            f"{label.name} {len(by_label[label])}/{n} ({len(by_label[label]) / n:.1%})"
            for label in sorted(by_label, key=lambda lab: lab.value)
        )
        raise ValueError(
            f"sampling {max_subjects} of {n} subjects leaves no "
            f"{', '.join(label.name for label in dropped)} subjects at all. "
            f"Population: {shares}. Raise max_subjects, or accept that this cohort is "
            f"too imbalanced at this size to support a binary comparison — but do not "
            f"discover that at evaluation."
        )

    rng = random.Random(seed)
    sampled: list[SubjectId] = []
    for label in sorted(quotas, key=lambda lab: lab.value):
        pool = by_label[label]
        k = quotas[label]
        if k >= len(pool):
            sampled.extend(pool)
            continue
        draw = pool.copy()
        rng.shuffle(draw)
        sampled.extend(draw[:k])

    return sorted(sampled)


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
        "Blocked on ADR-0004 reaching accepted status — the label rule must be fixed "
        "before a cohort is built from it. See docs/adr/0004-label-definition-and-"
        "target-visit.md."
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
