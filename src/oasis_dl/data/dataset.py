"""Torch datasets and batching over visit sequences.

Unimplemented below the interface — needs real scans. The collate function is the
interesting part and the one most likely to be got wrong, so its contract is written
out here before anyone implements it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from oasis_dl.types import VisitSequence


@dataclass
class SequenceBatch:
    """A padded batch of visit sequences.

    Sequences vary in both length and spacing (ADR-0006), so a batch is padded and
    masked rather than stacked.

    Attributes:
        scans: ``(batch, max_visits, 1, D, H, W)``. Padded positions are zero and are
            excluded by ``mask``, never by relying on their value.
        elapsed_days: ``(batch, max_visits)``. Days from each visit to its target
            visit, so the target is 0 and earlier visits are negative.
        mask: ``(batch, max_visits)`` boolean. True marks a real visit. The temporal
            module must consume this — attention over padded positions silently
            fabricates history.
        labels: ``(batch,)``. One label per sequence, from the target visit.
        anatomy_targets: ``(batch, n_targets)`` or None. Training only (ADR-0007).
        subject_ids: One per sequence, carried through so that metrics can be
            aggregated per subject as well as per visit.
    """

    scans: torch.Tensor
    elapsed_days: torch.Tensor
    mask: torch.Tensor
    labels: torch.Tensor
    anatomy_targets: torch.Tensor | None
    subject_ids: tuple[str, ...]


class VisitSequenceDataset(Dataset[VisitSequence]):
    """Visit sequences, loading scans on access.

    Args:
        sequences: Belonging to exactly one partition. Passing sequences from more
            than one partition is the mistake ADR-0001 exists to prevent, so callers
            build one dataset per partition.
        transform: Applied per scan. For a longitudinal sample the *same* spatial
            transform is applied across a subject's visits — independent augmentation
            injects apparent anatomical change between visits, which is precisely the
            signal the temporal module reads (see docs/protocol/preprocessing.md).
        load_anatomy_targets: Training only. Test sequences need no targets, so a
            target extraction that failed on a test subject costs nothing.
    """

    def __init__(
        self,
        sequences: Sequence[VisitSequence],
        *,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        load_anatomy_targets: bool = False,
    ) -> None:
        self.sequences = list(sequences)
        self.transform = transform
        self.load_anatomy_targets = load_anatomy_targets

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> VisitSequence:
        raise NotImplementedError("Blocked on preprocessed scans being available on disk.")


def collate_sequences(batch: Sequence[VisitSequence]) -> SequenceBatch:
    """Pad a list of sequences into a :class:`SequenceBatch`.

    Right-padding would put the target visit at a different index for every sequence.
    **Left-pad instead**, so the target visit is always last and a model that reads the
    final position reads the right thing for every sample in the batch.
    """
    raise NotImplementedError("Blocked on VisitSequenceDataset.__getitem__.")


def single_scan_collate(batch: Sequence[VisitSequence]) -> SequenceBatch:
    """Collate using only each sequence's target visit.

    The control arm of the first core ablation. Implemented as a collate function
    rather than a separate dataset so that both arms of the ablation draw from exactly
    the same sequences, and the only difference between them is how much of each
    sequence the model is allowed to see.
    """
    raise NotImplementedError("Blocked on collate_sequences.")
