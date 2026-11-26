"""The shared visit encoder: one scan in, one embedding out.

Shared means the same weights encode every visit of every subject (ADR-0005). Two
consequences follow and both are load-bearing: the temporal module compares embeddings
that live in one space, and the encoder sees every visit as training signal rather than
only the visits at its own sequence position.

Two families of backbone live here and they differ in one way that matters to callers.
Convolutional backbones (ResNet, DenseNet) accept any volume above a minimum. A Vision
Transformer does not: its positional embeddings are built for one input shape, so the
shape is a constructor argument and an input of any other size is rejected rather than
silently interpolated (ADR-0011).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

__all__ = ["PretrainedCheckpoint", "VisitEncoder"]

Shape3D = tuple[int, int, int]


def _build_resnet(
    depth: int, in_channels: int, _embedding_dim: int, _volume_shape: Shape3D
) -> tuple[nn.Module, int]:
    """A MONAI 3D ResNet with its classifier removed, so it emits pooled features.

    Ignores the requested width and the volume shape: a ResNet is fully convolutional,
    so it emits its native feature size for any input above its minimum, and
    :class:`VisitEncoder` adds the projection when that differs.
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


def _build_densenet(
    variant: int, in_channels: int, embedding_dim: int, _volume_shape: Shape3D
) -> tuple[nn.Module, int]:
    """A MONAI 3D DenseNet.

    Its classifier is the projection: MONAI's DenseNet always ends in a linear layer,
    so asking it for ``out_channels=embedding_dim`` produces the configured width
    directly and this encoder adds no second projection on top.
    """
    from monai.networks.nets import DenseNet121, DenseNet169, DenseNet201

    builders = {121: DenseNet121, 169: DenseNet169, 201: DenseNet201}
    module = builders[variant](spatial_dims=3, in_channels=in_channels, out_channels=embedding_dim)
    return module, embedding_dim


class _PooledViT(nn.Module):
    """A MONAI 3D ViT that emits one vector per volume instead of one per patch.

    MONAI's ``ViT`` returns ``(batch, tokens, hidden)`` plus its intermediate states.
    Mean-pooling over tokens is what makes this encoder *shared* in the sense ADR-0005
    means: the result does not depend on where a patch sits, so two identical volumes
    encode identically wherever they appear.

    There is no class token to take instead. MONAI adds one only with
    ``classification=True``, and BrainIAC's ViT is built without it, as this one is.
    BrainIAC's fine-tuning code reads ``features[0][:, 0]`` and calls it the CLS token,
    but in that model token 0 is the first patch, one corner of the volume. How its
    SimCLR pretraining pooled is not in its public code, so whether mean pooling matches
    what the weights were fitted for is unconfirmed.
    """

    def __init__(self, vit: nn.Module) -> None:
        super().__init__()
        self.vit = vit

    def forward(self, scans: torch.Tensor) -> torch.Tensor:
        tokens = self.vit(scans)
        if isinstance(tokens, tuple):
            tokens = tokens[0]
        pooled: torch.Tensor = tokens.mean(dim=1)
        return pooled


def _build_vit(
    patch: int,
    hidden: int,
    layers: int,
    heads: int,
    in_channels: int,
    _embedding_dim: int,
    volume_shape: Shape3D,
) -> tuple[nn.Module, int]:
    """A MONAI 3D Vision Transformer, pooled to one vector per volume.

    ``volume_shape`` is required rather than inferred because a ViT's positional
    embeddings are sized at construction. Building one for the wrong shape fails at the
    first forward pass with an error about a reshape, naming nothing the caller wrote.
    """
    from monai.networks.nets import ViT

    bad = [n for n in volume_shape if n % patch]
    if bad:
        raise ValueError(
            f"volume_shape {volume_shape} is not divisible by the patch size {patch}. "
            f"A ViT tiles the volume into whole patches, so every dimension must be a "
            f"multiple of {patch}."
        )
    module = ViT(
        in_channels=in_channels,
        img_size=volume_shape,
        patch_size=(patch, patch, patch),
        hidden_size=hidden,
        mlp_dim=hidden * 4,
        num_layers=layers,
        num_heads=heads,
        classification=False,
        spatial_dims=3,
    )
    _drop_unused_cross_attention(module)
    return _PooledViT(module), hidden


def _drop_unused_cross_attention(vit: nn.Module) -> None:
    """Remove the cross-attention layers MONAI allocates and never runs.

    MONAI 1.4's ``TransformerBlock`` builds ``cross_attn`` and ``norm_cross_attn``
    unconditionally, and only *uses* them when ``with_cross_attention`` is set — which
    ``ViT`` neither sets nor exposes. In ViT-B that is 28.3M of 116.7M parameters that
    never receive a gradient, inflate the parameter count every document quotes, and
    take optimizer state on the GPU for nothing.

    Removed after construction rather than avoided during it, so the random draws for
    every other weight are unchanged and the forward pass is bit-identical — which a
    test pins. Guarded per block, so a future MONAI that does enable cross-attention
    keeps it.
    """
    for block in getattr(vit, "blocks", ()):
        if getattr(block, "with_cross_attention", True):
            continue
        for name in ("cross_attn", "norm_cross_attn"):
            if hasattr(block, name):
                delattr(block, name)


@dataclass(frozen=True)
class _BackboneSpec:
    """How to build a backbone, and what input it will accept.

    ``min_spatial_size`` is measured, not derived: each convolutional backbone
    downsamples a fixed number of times, and below its minimum an interior pooling layer
    is handed a 1-voxel input and raises from somewhere the caller never wrote.
    Recording it here turns that into an error naming the backbone and the shape.

    ``fixed_input`` marks a backbone whose input shape is fixed at construction. For
    those the check is equality, not a minimum.
    """

    build: Callable[[int, int, Shape3D], tuple[nn.Module, int]]
    min_spatial_size: int
    fixed_input: bool = False


_BACKBONES: dict[str, _BackboneSpec] = {
    "resnet10": _BackboneSpec(lambda c, d, s: _build_resnet(10, c, d, s), 8),
    "resnet18": _BackboneSpec(lambda c, d, s: _build_resnet(18, c, d, s), 8),
    "resnet34": _BackboneSpec(lambda c, d, s: _build_resnet(34, c, d, s), 8),
    "resnet50": _BackboneSpec(lambda c, d, s: _build_resnet(50, c, d, s), 8),
    "densenet121": _BackboneSpec(lambda c, d, s: _build_densenet(121, c, d, s), 32),
    "densenet169": _BackboneSpec(lambda c, d, s: _build_densenet(169, c, d, s), 32),
    "densenet201": _BackboneSpec(lambda c, d, s: _build_densenet(201, c, d, s), 32),
    # ViT-B/16 is the BrainIAC architecture, so a BrainIAC checkpoint loads into it
    # without reshaping anything (ADR-0011).
    "vit_base_16": _BackboneSpec(
        lambda c, d, s: _build_vit(16, 768, 12, 12, c, d, s), 16, fixed_input=True
    ),
    # Same code path at about 1/40th the size at 96^3, for tests and for any machine
    # without a GPU. The ratio is shape-dependent — positional embeddings grow with the
    # token count — which is how an earlier figure measured at 32^3 came out wrong.
    # Production numbers never come from this one.
    "vit_tiny_8": _BackboneSpec(
        lambda c, d, s: _build_vit(8, 192, 4, 3, c, d, s), 8, fixed_input=True
    ),
}
"""Backbone name to spec. Adding one is a registry entry, not a new encoder."""


@dataclass(frozen=True)
class PretrainedCheckpoint:
    """Provenance for a set of pretrained weights.

    The fields that matter for this project are not the architecture ones. They are
    ``pretraining_corpus`` and ``overlaps_oasis``: a brain-MRI foundation model trained
    on an open corpus has almost certainly seen OASIS, because OASIS is one of the few
    open longitudinal brain MRI datasets. That is not a reason to avoid pretraining, but
    it is a reason the fact must travel with the weights rather than live in a reviewer's
    memory, so loading such a checkpoint warns and the warning names the corpus.

    Attributes:
        backbone: The only backbone these weights fit.
        source: Where the weights come from.
        license_terms: Short form. All current entries are research-only.
        pretraining_corpus: Named datasets used for self-supervised pretraining.
        overlaps_oasis: Whether that corpus includes any OASIS release.
        key_prefix: What the file puts before each backbone weight's name. Weights
            outside it, such as a pretraining projection head, are not loaded.
    """

    backbone: str
    source: str
    license_terms: str
    pretraining_corpus: tuple[str, ...]
    overlaps_oasis: bool
    key_prefix: str = ""


_CHECKPOINTS: dict[str, PretrainedCheckpoint] = {
    "brainiac_vit_b": PretrainedCheckpoint(
        backbone="vit_base_16",
        source="https://github.com/AIM-KannLab/BrainIAC",
        license_terms="BrainIAC Research-Only License (non-commercial academic)",
        pretraining_corpus=("ABCD", "ADNI", "OASIS-2", "OASIS-3", "ABIDE", "PPMI", "IXI"),
        overlaps_oasis=True,
        # The released file is a Lightning checkpoint holding the ViT under `backbone.`
        # beside the SimCLR projection head; BrainIAC's own loader strips the prefix the
        # same way (src/model.py).
        key_prefix="backbone.",
    ),
}
"""Known checkpoints. Weights are never committed — see :meth:`VisitEncoder.pretrained_checkpoints`."""


class VisitEncoder(nn.Module):
    """Maps a 3D T1-weighted scan to a fixed-length embedding.

    Wraps a MONAI 3D backbone rather than defining one, so that swapping ResNet for a
    Vision Transformer is a config change. The project's contribution is the controlled
    comparison, not the backbone (see docs/audits/).

    The module holds no per-visit or per-position parameters. That is what makes it
    *shared*: a visit's embedding depends on the scan and nothing else, so the same
    weights can encode position 1 of one subject and position 3 of another, and the
    temporal module compares them in one space.

    Args:
        backbone: MONAI network name, e.g. ``"vit_base_16"``. See
            :meth:`supported_backbones`.
        embedding_dim: Width of the output embedding. A linear projection is added
            only when the backbone's native width differs.
        in_channels: 1 for T1-weighted structural MRI.
        volume_shape: The input shape this encoder is built for. Convolutional
            backbones ignore it; a ViT is constructed around it and accepts nothing
            else. Defaults to the ADR-0010 shape.
        pretrained: Name of a checkpoint in :meth:`pretrained_checkpoints`, or None for
            random initialization. Whichever is chosen must be identical across every
            arm, or ablation 1 confounds the temporal module with the initialization.
        checkpoint_path: Where the weights file actually is. Required with
            ``pretrained`` because weights never enter git, and nothing here downloads
            them — an encoder that fetched a checkpoint over the network would make a
            recorded run depend on a URL that can change under it.

    Attributes:
        min_spatial_size: Smallest volume edge this backbone accepts.

    Raises:
        ValueError: If ``backbone`` or ``pretrained`` is unknown, the dimensions are
            not positive, or a checkpoint is named without a path to it.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        embedding_dim: int = 512,
        in_channels: int = 1,
        volume_shape: Shape3D = (96, 96, 96),
        pretrained: str | None = None,
        checkpoint_path: Path | None = None,
    ) -> None:
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(
                f"unknown backbone {backbone!r}; supported: {', '.join(self.supported_backbones())}"
            )
        if embedding_dim < 1 or in_channels < 1:
            raise ValueError("embedding_dim and in_channels must be positive")
        if any(n < 1 for n in volume_shape):
            raise ValueError(f"volume_shape must be positive, got {volume_shape}")

        self.backbone = backbone
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels
        self.volume_shape = volume_shape
        self.pretrained = pretrained

        # Checked before the backbone is built. A ViT-B is 116M parameters, and a
        # misspelled checkpoint name should not cost the time to allocate them.
        if pretrained is not None:
            self._check_pretrained(pretrained, checkpoint_path)

        spec = _BACKBONES[backbone]
        self.min_spatial_size = spec.min_spatial_size
        self.fixed_input = spec.fixed_input
        self.body, native_width = spec.build(in_channels, embedding_dim, volume_shape)
        self.project: nn.Module = (
            nn.Identity()
            if native_width == embedding_dim
            else nn.Linear(native_width, embedding_dim)
        )

        if pretrained is not None:
            assert checkpoint_path is not None  # established by _check_pretrained
            self._load_pretrained(_CHECKPOINTS[pretrained], checkpoint_path)

    def _check_pretrained(self, name: str, checkpoint_path: Path | None) -> None:
        """Refuse an unusable checkpoint, and announce a contaminated one.

        Separate from the load so that every reason a checkpoint cannot work is raised
        before any weights are allocated.
        """
        if name not in _CHECKPOINTS:
            raise ValueError(
                f"unknown checkpoint {name!r}; known: {', '.join(self.pretrained_checkpoints())}"
            )
        spec = _CHECKPOINTS[name]
        if spec.backbone != self.backbone:
            raise ValueError(
                f"checkpoint {name!r} holds {spec.backbone} weights but this encoder is "
                f"{self.backbone}. Loading it would silently initialize almost nothing."
            )
        if checkpoint_path is None:
            raise ValueError(
                f"checkpoint {name!r} needs checkpoint_path. Weights never enter git, so "
                f"download them from {spec.source} and point the config at the local file."
            )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"no checkpoint at {checkpoint_path} (for {name!r})")

        if spec.overlaps_oasis:
            warnings.warn(
                f"{name!r} was pretrained on a corpus including "
                f"{', '.join(c for c in spec.pretraining_corpus if 'OASIS' in c)}, so the "
                f"encoder has seen test-partition scans without their labels. Absolute "
                f"numbers are therefore not comparable with from-scratch literature. The "
                f"arm-to-arm contrasts are unaffected because every arm shares this "
                f"encoder — see ADR-0011.",
                stacklevel=3,
            )

    def _load_pretrained(self, spec: PretrainedCheckpoint, checkpoint_path: Path) -> None:
        """Copy every backbone weight in the checkpoint into the network, or refuse.

        Only the weights under ``spec.key_prefix`` are read, with the prefix removed,
        and they go into the MONAI network itself rather than this encoder's wrapper
        around it, which is how the checkpoint's authors load it too.

        A key mismatch is silent in ``load_state_dict(strict=False)``: training then
        proceeds from random weights, wholly or in part, at full speed and with every
        shape right. So the result is inspected, and anything short of an exact match
        is an error. Total mismatch gets its own message because it is the usual cause:
        a file laid out differently from what ``key_prefix`` says.
        """
        state = torch.load(checkpoint_path, map_location="cpu")
        for key in ("state_dict", "model", "model_state_dict"):
            if isinstance(state, dict) and key in state:
                state = state[key]
                break
        prefix = spec.key_prefix
        weights = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}

        network = self.body.vit if isinstance(self.body, _PooledViT) else self.body
        result = network.load_state_dict(weights, strict=False)
        matched = len(network.state_dict()) - len(result.missing_keys)
        if matched == 0:
            raise ValueError(
                f"checkpoint {checkpoint_path} matched no parameter of {self.backbone}. "
                f"Its keys look like {list(state)[:3]}, and backbone weights were "
                f"expected under {prefix!r}. Training would have started from random "
                f"weights while reporting itself as pretrained."
            )
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(
                f"checkpoint {checkpoint_path} only partly fits {self.backbone}: "
                f"{matched} of {matched + len(result.missing_keys)} weights loaded. "
                f"Not in the file: {result.missing_keys[:5]}. In the file under "
                f"{prefix!r} but not in the network: {result.unexpected_keys[:5]}. "
                f"Training would have mixed pretrained and random layers."
            )

    @staticmethod
    def supported_backbones() -> tuple[str, ...]:
        """Every backbone name this encoder accepts."""
        return tuple(_BACKBONES)

    @staticmethod
    def pretrained_checkpoints() -> tuple[str, ...]:
        """Every checkpoint name :meth:`__init__` accepts for ``pretrained``."""
        return tuple(_CHECKPOINTS)

    @staticmethod
    def checkpoint_provenance(name: str) -> PretrainedCheckpoint:
        """Where a checkpoint came from, what it was trained on, and its licence."""
        if name not in _CHECKPOINTS:
            raise ValueError(f"unknown checkpoint {name!r}")
        return _CHECKPOINTS[name]

    def forward(self, scans: torch.Tensor) -> torch.Tensor:
        """Encode a batch of scans.

        Args:
            scans: ``(batch, in_channels, D, H, W)``.

        Returns:
            ``(batch, embedding_dim)``.

        Raises:
            ValueError: If ``scans`` is not 5-dimensional, or its spatial shape is one
                this backbone cannot take. Caught here rather than deep inside the
                backbone, where the error names a convolution the caller never wrote.
        """
        if scans.dim() != 5:
            raise ValueError(
                f"expected a 5-dimensional (batch, channels, D, H, W) tensor, got "
                f"shape {tuple(scans.shape)}. A sequence batch must be flattened to "
                f"(batch * visits, channels, D, H, W) before encoding."
            )
        spatial = tuple(scans.shape[2:])
        if self.fixed_input and spatial != self.volume_shape:
            raise ValueError(
                f"{self.backbone} was built for volume_shape {self.volume_shape} and "
                f"cannot take {spatial}. A ViT's positional embeddings are sized at "
                f"construction, so build a second encoder rather than resizing input "
                f"that a recorded run depends on."
            )
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
