"""The shared visit encoder: one scan in, one embedding out.

Shared means the same weights encode every visit of every subject (ADR-0005). Two
consequences follow and both are load-bearing: the temporal module compares embeddings
that live in one space, and the encoder sees every visit as training signal rather than
only the visits at its own sequence position.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

__all__ = ["VisitEncoder"]


def _build_resnet(depth: int, in_channels: int, _embedding_dim: int) -> tuple[nn.Module, int]:
    """A MONAI 3D ResNet with its classifier removed, so it emits pooled features.

    Ignores the requested width: a ResNet emits its native feature size and
    :class:`VisitEncoder` adds the projection when it differs. DenseNet projects
    internally instead, which is why both builders share one signature.
    """
    from monai.networks.nets import resnet10, resnet18, resnet34, resnet50

    builders = {10: resnet10, 18: resnet18, 34: resnet34, 50: resnet50}
    module = builders[depth](
        spatial_dims=3,
        n_input_channels=in_channels,
        feed_forward=False,
        shortcut_type="B",
        bias_downsample=False,
    )
    # BasicBlock stacks end at 512; Bottleneck (depth 50) expands by 4.
    width = 2048 if depth >= 50 else 512
    return module, width


def _build_densenet(variant: int, in_channels: int, embedding_dim: int) -> tuple[nn.Module, int]:
    """A MONAI 3D DenseNet.

    Its classifier is the projection: MONAI's DenseNet always ends in a linear layer,
    so asking it for ``out_channels=embedding_dim`` produces the configured width
    directly and this encoder adds no second projection on top.
    """
    from monai.networks.nets import DenseNet121, DenseNet169, DenseNet201

    builders = {121: DenseNet121, 169: DenseNet169, 201: DenseNet201}
    module = builders[variant](spatial_dims=3, in_channels=in_channels, out_channels=embedding_dim)
    return module, embedding_dim


@dataclass(frozen=True)
class _BackboneSpec:
    """How to build a backbone, and the smallest volume it can accept.

    ``min_spatial_size`` is measured, not derived: each backbone downsamples a fixed
    number of times, and below its minimum an interior pooling layer is handed a 1-voxel
    input and raises from somewhere the caller never wrote. Recording it here turns that
    into an error naming the backbone and the shape.
    """

    build: Callable[[int, int], tuple[nn.Module, int]]
    min_spatial_size: int


_BACKBONES: dict[str, _BackboneSpec] = {
    "resnet10": _BackboneSpec(lambda c, d: _build_resnet(10, c, d), 8),
    "resnet18": _BackboneSpec(lambda c, d: _build_resnet(18, c, d), 8),
    "resnet34": _BackboneSpec(lambda c, d: _build_resnet(34, c, d), 8),
    "resnet50": _BackboneSpec(lambda c, d: _build_resnet(50, c, d), 8),
    "densenet121": _BackboneSpec(lambda c, d: _build_densenet(121, c, d), 32),
    "densenet169": _BackboneSpec(lambda c, d: _build_densenet(169, c, d), 32),
    "densenet201": _BackboneSpec(lambda c, d: _build_densenet(201, c, d), 32),
}
"""Backbone name to spec. Adding one is a registry entry, not a new encoder."""


class VisitEncoder(nn.Module):
    """Maps a 3D T1-weighted scan to a fixed-length embedding.

    Wraps a MONAI 3D backbone rather than defining one, so that swapping ResNet for
    DenseNet is a config change. The project's contribution is the controlled
    comparison, not the backbone (see docs/audits/).

    The module holds no per-visit or per-position parameters. That is what makes it
    *shared*: a visit's embedding depends on the scan and nothing else, so the same
    weights can encode position 1 of one subject and position 3 of another, and the
    temporal module compares them in one space.

    Args:
        backbone: MONAI network name, e.g. ``"resnet18"``. See
            :meth:`supported_backbones`.
        embedding_dim: Width of the output embedding. A linear projection is added
            only when the backbone's native width differs.
        in_channels: 1 for T1-weighted structural MRI.
        pretrained: Whether to initialize from pretrained weights. A stretch goal —
            the available 3D checkpoints are largely CT-trained, and CT-to-T1 transfer
            is an open question rather than a free win (docs/survey/).

    Attributes:
        min_spatial_size: Smallest volume edge this backbone accepts. ResNets take 8,
            DenseNets 32 — which matters because the synthetic cohort defaults to 16.

    Raises:
        ValueError: If ``backbone`` is unknown or the dimensions are not positive.
        NotImplementedError: If ``pretrained`` is requested.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        embedding_dim: int = 512,
        in_channels: int = 1,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(
                f"unknown backbone {backbone!r}; supported: {', '.join(self.supported_backbones())}"
            )
        if embedding_dim < 1 or in_channels < 1:
            raise ValueError("embedding_dim and in_channels must be positive")
        if pretrained:
            raise NotImplementedError(
                "Pretrained 3D weights are a stretch goal, not wired up. The available "
                "checkpoints are largely CT-trained and CT-to-T1 transfer is an open "
                "question (docs/survey/). Enabling this silently would make every arm "
                "depend on an untested transfer."
            )

        self.backbone = backbone
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels
        self.pretrained = pretrained

        spec = _BACKBONES[backbone]
        self.min_spatial_size = spec.min_spatial_size
        self.body, native_width = spec.build(in_channels, embedding_dim)
        self.project: nn.Module = (
            nn.Identity()
            if native_width == embedding_dim
            else nn.Linear(native_width, embedding_dim)
        )

    @staticmethod
    def supported_backbones() -> tuple[str, ...]:
        """Every backbone name this encoder accepts."""
        return tuple(_BACKBONES)

    def forward(self, scans: torch.Tensor) -> torch.Tensor:
        """Encode a batch of scans.

        Args:
            scans: ``(batch, in_channels, D, H, W)``.

        Returns:
            ``(batch, embedding_dim)``.

        Raises:
            ValueError: If ``scans`` is not 5-dimensional. Caught here rather than
                deep inside the backbone, where the error names a convolution the
                caller never wrote.
        """
        if scans.dim() != 5:
            raise ValueError(
                f"expected a 5-dimensional (batch, channels, D, H, W) tensor, got "
                f"shape {tuple(scans.shape)}. A sequence batch must be flattened to "
                f"(batch * visits, channels, D, H, W) before encoding."
            )
        spatial = tuple(scans.shape[2:])
        if min(spatial) < self.min_spatial_size:
            raise ValueError(
                f"{self.backbone} needs at least {self.min_spatial_size} voxels in "
                f"every spatial dimension, got {spatial}. Below that an interior "
                f"pooling layer is handed a single voxel. Either raise the volume "
                f"shape or pick a backbone with a smaller minimum "
                f"(resnet* accept 8)."
            )
        features = self.body(scans)
        projected: torch.Tensor = self.project(features)
        return projected
