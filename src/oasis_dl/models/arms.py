"""The five experimental arms, assembled from shared parts.

Assembly matters as much as the parts. B1 and PA differ by exactly one component — the
temporal module — because they are the two sides of the first core ablation and any
other difference would confound it (docs/protocol/README.md).
"""

from __future__ import annotations

import torch
from torch import nn

from oasis_dl.data.dataset import SequenceBatch
from oasis_dl.models.heads import AnatomyHead, ClassificationHead
from oasis_dl.models.temporal import TemporalAttention
from oasis_dl.models.encoder import VisitEncoder


class ModelOutput(nn.Module):
    """Marker base so every arm returns the same shape of result."""


class SingleScanModel(nn.Module):
    """**B1** — encoder plus classification head, target visit only.

    The transparent baseline. Receives the same ``SequenceBatch`` as the longitudinal
    arms and reads only the final position, so that the two arms of ablation 1 consume
    identical data and differ solely in how much of it the architecture can use.
    """

    def __init__(self, encoder: VisitEncoder, head: ClassificationHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, batch: SequenceBatch) -> dict[str, torch.Tensor]:
        """Returns ``{"logits": (batch,)}``."""
        raise NotImplementedError


class LongitudinalModel(nn.Module):
    """**PA** and **PB** — shared encoder, temporal module, classification head.

    PB is PA with ``anatomy_head`` supplied. The forward pass is otherwise identical,
    and at inference the anatomy head is not called at all, which is what makes the two
    arms comparable (ADR-0007).
    """

    def __init__(
        self,
        encoder: VisitEncoder,
        temporal: TemporalAttention,
        head: ClassificationHead,
        anatomy_head: AnatomyHead | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.temporal = temporal
        self.head = head
        self.anatomy_head = anatomy_head

    def forward(self, batch: SequenceBatch) -> dict[str, torch.Tensor]:
        """Returns ``{"logits": ...}``, plus ``"anatomy"`` when training arm PB.

        The encoder is applied per visit by flattening the batch and visit axes
        together, which is what gives the shared encoder its full gradient signal.
        """
        raise NotImplementedError
