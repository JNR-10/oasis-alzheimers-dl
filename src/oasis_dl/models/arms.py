"""The five experimental arms, assembled from shared parts.

Assembly matters as much as the parts. B1 and PA differ by exactly one component — the
temporal module — because they are the two sides of the first core ablation and any
other difference would confound it (docs/protocol/README.md).
"""

from __future__ import annotations

import torch
from torch import nn

from oasis_dl.data.dataset import SequenceBatch
from oasis_dl.models.encoder import VisitEncoder
from oasis_dl.models.heads import AnatomyHead, ClassificationHead
from oasis_dl.models.temporal import TemporalAttention

__all__ = ["LongitudinalModel", "ModelOutput", "SingleScanModel"]


def _check_scans(scans: torch.Tensor) -> None:
    """Refuse a batch of the wrong rank before it reaches a convolution."""
    if scans.dim() != 6:
        raise ValueError(
            f"expected (batch, visits, channels, D, H, W) scans, got "
            f"{tuple(scans.shape)}. Collation produces this rank even for a "
            f"single-visit arm, so a 5-dimensional tensor means the batch was built "
            f"somewhere else."
        )


def _encode_real_visits(
    encoder: VisitEncoder, scans: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Encode every real visit once, and no padded ones.

    Padded positions are **excluded from the encoder**, not merely masked afterwards.
    The ResNet backbones carry BatchNorm, which takes statistics over whatever is in
    the batch — so encoding zero-filled padding would let a sequence's prediction
    depend on how long the longest *other* sequence in its batch happened to be. That
    is reproducible-looking, batch-order dependent, and invisible in every shape.

    Flattening batch and visit axes together is also what gives the shared encoder its
    full gradient signal: every visit of every subject is one more training example for
    it, not just the ones at the final position (ADR-0005).

    Returns:
        ``(batch, visits, embedding_dim)``, zero at padded positions.
    """
    batch, visits = scans.shape[:2]
    flat = scans.reshape(batch * visits, *scans.shape[2:])
    flat_mask = mask.reshape(-1)

    encoded = encoder(flat[flat_mask])
    embeddings = torch.zeros(
        batch * visits, encoded.shape[1], dtype=encoded.dtype, device=encoded.device
    )
    # Out-of-place, so autograd routes gradients back through `encoded` cleanly.
    embeddings = embeddings.index_copy(0, torch.nonzero(flat_mask).squeeze(1), encoded)
    return embeddings.reshape(batch, visits, -1)


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
        """Returns ``{"logits": (batch,)}``.

        Reads ``scans[:, -1]`` and nothing else. Collation left-pads precisely so the
        final position is the target visit for every sequence, so this arm sees the same
        visit for every subject while receiving the identical batch arm PA receives.
        """
        _check_scans(batch.scans)
        representation = self.encoder(batch.scans[:, -1])
        return {"logits": self.head(representation)}


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
        _check_scans(batch.scans)
        embeddings = _encode_real_visits(self.encoder, batch.scans, batch.mask)
        representation = self.temporal(embeddings, batch.elapsed_days, batch.mask)

        outputs = {"logits": self.head(representation)}
        # Training only (ADR-0007). Gated on `self.training` rather than on a flag the
        # caller passes, so arms PA and PB cannot differ at inference even by mistake:
        # in eval mode the head is not called and the key is simply absent.
        if self.anatomy_head is not None and self.training:
            outputs["anatomy"] = self.anatomy_head(representation)
        return outputs
