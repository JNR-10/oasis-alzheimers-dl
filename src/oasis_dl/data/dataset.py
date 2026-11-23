"""Torch datasets and batching over visit sequences.

The collate function is the interesting part and the one most likely to be got wrong,
so its contract is written out before the implementation.

**Loading happens in the dataset, not in collate.** The shared-augmentation rule below
is the reason: the same spatial transform must apply across one subject's visits, and
that is only expressible where a subject's visits are together — inside
``__getitem__``. Collate then sees tensors that are already correct, and its only job
is padding. Consequently ``__getitem__`` yields a :class:`LoadedSequence` (the record
plus its scans) rather than a bare :class:`~oasis_dl.types.VisitSequence`, which stays
a pure record of what happened.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, NamedTuple, cast

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from oasis_dl.types import VisitSequence

__all__ = [
    "LoadedSequence",
    "SequenceBatch",
    "VisitSequenceDataset",
    "collate_sequences",
    "single_scan_collate",
]


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
            Rows for sequences without targets are zero and are excluded by
            ``anatomy_mask``, never by trusting their value.
        anatomy_mask: ``(batch,)`` boolean, or None when no sequence carries
            targets. False marks a sequence the auxiliary loss must skip. Not
            every session ships FreeSurfer output (ADR-0009), so a training
            subject without targets still contributes classification loss and is
            excluded from the auxiliary head alone.
        subject_ids: One per sequence, carried through so that metrics can be
            aggregated per subject as well as per visit.
    """

    scans: torch.Tensor
    elapsed_days: torch.Tensor
    mask: torch.Tensor
    labels: torch.Tensor
    anatomy_targets: torch.Tensor | None
    anatomy_mask: torch.Tensor | None
    subject_ids: tuple[str, ...]

    def to(self, device: torch.device) -> SequenceBatch:
        """Move every tensor field, leaving the subject ids where they are.

        Written over ``fields()`` rather than field by field so that a tensor added to
        this batch later cannot be silently left on the CPU — which surfaces as a device
        mismatch deep inside a model, naming a layer nobody wrote.
        """
        moved: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            moved[field.name] = value.to(device) if isinstance(value, torch.Tensor) else value
        return SequenceBatch(**moved)


@dataclass(frozen=True)
class LoadedSequence:
    """One sequence with its scans read from disk and transformed.

    Attributes:
        sequence: The record. Its ``elapsed_days`` and ``label`` stay authoritative.
        scans: ``(visits, 1, D, H, W)`` in the sequence's own order, so the target
            visit is the last element here as it is everywhere else.
        anatomy_targets: The target visit's structural quantities, or None when not
            requested.
    """

    sequence: VisitSequence
    scans: torch.Tensor
    anatomy_targets: dict[str, float] | None = None


def load_scan(path: Path) -> torch.Tensor:
    """Read one preprocessed scan as ``(1, D, H, W)`` float32.

    The channel axis is added here rather than at collate so that a transform sees the
    same shape a model will.
    """
    # nibabel.load is typed as a union of every image class it can return; this
    # pipeline only ever writes NIfTI, so narrow rather than widen the caller.
    image = cast("nib.Nifti1Image", nib.load(path))
    array = np.asarray(image.dataobj, dtype=np.float32)
    return torch.from_numpy(array).unsqueeze(0)


class VisitSequenceDataset(Dataset[LoadedSequence]):
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
        seed: Base for the per-sample transform draw. Recorded so an augmented epoch
            is reproducible (ADR-0008). Call :meth:`set_epoch` each epoch, or every
            epoch draws the identical augmentation and it degenerates into fixed
            preprocessing.
    """

    def __init__(
        self,
        sequences: Sequence[VisitSequence],
        *,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        load_anatomy_targets: bool = False,
        seed: int = 0,
    ) -> None:
        self.sequences = list(sequences)
        self.transform = transform
        self.load_anatomy_targets = load_anatomy_targets
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Advance the augmentation draw.

        Without this the per-sample seed is fixed, so a sample receives the same
        augmentation at every epoch and the augmentation stops being augmentation.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> LoadedSequence:
        sequence = self.sequences[index]

        # One draw per sample, reused for every visit of that subject. Drawing
        # independently per visit would manufacture anatomical change between visits —
        # the exact signal the temporal module is asked to detect.
        sample_seed = self.seed + self.epoch * 1_000_003 + index

        scans = []
        for visit in sequence.visits:
            scan = load_scan(visit.scan_path)
            if self.transform is not None:
                # Save and restore around the draw. Seeding torch globally from inside
                # a dataset makes model init, dropout and shuffling depend on the order
                # batches happen to be fetched in, which breaks ADR-0008 reproducibility
                # in a way nothing would report.
                state = torch.get_rng_state()
                try:
                    torch.manual_seed(sample_seed)
                    scan = self.transform(scan)
                finally:
                    torch.set_rng_state(state)
            scans.append(scan)

        targets = sequence.target_visit.anatomy_targets if self.load_anatomy_targets else None
        return LoadedSequence(sequence=sequence, scans=torch.stack(scans), anatomy_targets=targets)


def _empty_batch_guard(batch: Sequence[LoadedSequence]) -> None:
    """Refuse an empty batch rather than return a zero-row one.

    A zero-row SequenceBatch flows through training without erroring and contributes
    nothing, which looks like a quiet epoch rather than a bug.
    """
    if not batch:
        raise ValueError("cannot collate an empty batch")


def _stack_anatomy_targets(
    batch: Sequence[LoadedSequence],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Stack anatomy targets in a stable key order, with a per-sequence validity mask.

    Masked per sequence rather than dropped per batch. Not every session ships
    FreeSurfer output (ADR-0009), so a batch containing one target-less training
    subject must still train the auxiliary head on the rest — otherwise whether the
    auxiliary objective runs at all depends on how the sampler happened to fill the
    batch, which is a silent, non-reproducible change to the loss.

    Returns:
        ``(values, mask)``, or ``(None, None)`` when no sequence carries targets.
        Rows without targets are zero **and** masked False; the mask is the contract,
        never the zero.

    Raises:
        ValueError: If sequences disagree on *which* targets they carry. A missing
            target is omitted, never zero-filled — a zero volume is a
            plausible-looking number that is never true.
    """
    present = [item.anatomy_targets for item in batch]
    if all(targets is None for targets in present):
        return None, None

    key_sets = {tuple(sorted(targets)) for targets in present if targets is not None}
    if len(key_sets) > 1:
        raise ValueError(
            f"sequences in one batch carry different anatomy targets: {key_sets}. "
            f"The auxiliary head predicts a fixed vector, so a batch cannot hold two "
            f"different target sets."
        )

    keys = sorted(next(iter(key_sets)))
    values = torch.zeros((len(batch), len(keys)), dtype=torch.float32)
    mask = torch.zeros(len(batch), dtype=torch.bool)
    for row, targets in enumerate(present):
        if targets is None:
            continue
        values[row] = torch.tensor([float(targets[k]) for k in keys])
        mask[row] = True
    return values, mask


class _BatchIdentity(NamedTuple):
    """The fields a batch carries over unchanged from its sequences."""

    labels: torch.Tensor
    anatomy_targets: torch.Tensor | None
    anatomy_mask: torch.Tensor | None
    subject_ids: tuple[str, ...]


def _batch_identity(batch: Sequence[LoadedSequence]) -> _BatchIdentity:
    """The fields both collate functions carry unchanged from the sequences.

    Shared so that the longitudinal and single-scan arms cannot drift apart in how
    they label or identify a batch. Ablation 1 varies architecture alone, and that
    guarantee is easier to keep when there is one place the labels come from.
    """
    targets, mask = _stack_anatomy_targets(batch)
    return _BatchIdentity(
        labels=torch.tensor([int(item.sequence.label) for item in batch]),
        anatomy_targets=targets,
        anatomy_mask=mask,
        subject_ids=tuple(str(item.sequence.subject_id) for item in batch),
    )


def collate_sequences(batch: Sequence[LoadedSequence]) -> SequenceBatch:
    """Pad a list of sequences into a :class:`SequenceBatch`.

    Right-padding would put the target visit at a different index for every sequence.
    **Left-pad instead**, so the target visit is always last and a model that reads the
    final position reads the right thing for every sample in the batch.
    """
    _empty_batch_guard(batch)

    max_visits = max(item.scans.shape[0] for item in batch)
    reference = batch[0].scans
    scans = torch.zeros((len(batch), max_visits, *reference.shape[1:]), dtype=reference.dtype)
    elapsed = torch.zeros((len(batch), max_visits), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_visits), dtype=torch.bool)

    for row, item in enumerate(batch):
        n_visits = item.scans.shape[0]
        start = max_visits - n_visits  # left pad: real visits occupy the tail
        scans[row, start:] = item.scans
        # Padded positions keep elapsed=0, which is also what the target visit
        # reads. The two are only distinguishable through `mask`, so a consumer
        # that ignores the mask sees a batch of target visits. The temporal module
        # must take the mask, not infer history from this tensor.
        elapsed[row, start:] = torch.tensor(item.sequence.elapsed_days, dtype=torch.float32)
        mask[row, start:] = True

    identity = _batch_identity(batch)
    return SequenceBatch(
        scans=scans,
        elapsed_days=elapsed,
        mask=mask,
        labels=identity.labels,
        anatomy_targets=identity.anatomy_targets,
        anatomy_mask=identity.anatomy_mask,
        subject_ids=identity.subject_ids,
    )


def single_scan_collate(batch: Sequence[LoadedSequence]) -> SequenceBatch:
    """Collate using only each sequence's target visit.

    The control arm of the first core ablation. Implemented as a collate function
    rather than a separate dataset so that both arms of the ablation draw from exactly
    the same sequences, and the only difference between them is how much of each
    sequence the model is allowed to see.
    """
    _empty_batch_guard(batch)

    scans = torch.stack([item.scans[-1] for item in batch]).unsqueeze(1)
    n = len(batch)
    identity = _batch_identity(batch)
    return SequenceBatch(
        scans=scans,
        # The target visit is its own anchor, so elapsed time is zero by construction
        # rather than by convention.
        elapsed_days=torch.zeros((n, 1), dtype=torch.float32),
        mask=torch.ones((n, 1), dtype=torch.bool),
        labels=identity.labels,
        anatomy_targets=identity.anatomy_targets,
        anatomy_mask=identity.anatomy_mask,
        subject_ids=identity.subject_ids,
    )
