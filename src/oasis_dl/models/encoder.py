"""The shared visit encoder: one scan in, one embedding out.

Shared means the same weights encode every visit of every subject (ADR-0005). Two
consequences follow and both are load-bearing: the temporal module compares embeddings
that live in one space, and the encoder sees every visit as training signal rather than
only the visits at its own sequence position.
"""

from __future__ import annotations

import torch
from torch import nn


class VisitEncoder(nn.Module):
    """Maps a 3D T1-weighted scan to a fixed-length embedding.

    Wraps a MONAI 3D backbone rather than defining one, so that swapping ResNet for
    DenseNet is a config change. The project's contribution is the controlled
    comparison, not the backbone (see docs/audits/).

    Args:
        backbone: MONAI network name, e.g. ``"resnet18"``.
        embedding_dim: Width of the output embedding.
        in_channels: 1 for T1-weighted structural MRI.
        pretrained: Whether to initialize from pretrained weights. A stretch goal —
            the available 3D checkpoints are largely CT-trained, and CT-to-T1 transfer
            is an open question rather than a free win (docs/survey/).
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        embedding_dim: int = 512,
        in_channels: int = 1,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels
        self.pretrained = pretrained
        raise NotImplementedError("Backbone construction not written yet.")

    def forward(self, scans: torch.Tensor) -> torch.Tensor:
        """Encode a batch of scans.

        Args:
            scans: ``(batch, in_channels, D, H, W)``.

        Returns:
            ``(batch, embedding_dim)``.
        """
        raise NotImplementedError
