"""The temporal module: a subject's visit embeddings in, one representation out.

Conditioned on elapsed time, because position in a sequence is not time (ADR-0006).
Two subjects with identical embedding sequences four months and four years apart must
not be read identically — atrophy *rate* is the signal.
"""

from __future__ import annotations

import math

import torch
from torch import nn

__all__ = ["ElapsedTimeEncoding", "TemporalAttention"]


class ElapsedTimeEncoding(nn.Module):
    """Turns elapsed days into a vector the temporal module can attend over.

    Replaces the standard positional encoding, which would encode "third visit" rather
    than "four years later" (ADR-0006).

    Args:
        embedding_dim: Must match the visit embedding width, since the encoding is
            combined with it.
        scale_days: Divisor putting elapsed days on a sensible numeric range before
            the sinusoidal basis is applied. 365.25 makes one unit one year.
        min_period_years: Shortest period in the basis. Below the spacing the cohort
            actually contains, the extra components only alias.
        max_period_years: Longest period. Must exceed the longest follow-up, or the
            slowest component wraps and two far-apart visits encode alike.

    The basis is **fixed, not learned**. On ~450 training sequences a learned time
    embedding fits the particular visit spacings this cohort happens to contain rather
    than the passage of time, and OASIS-3 spacing is clinic-driven and clustered.

    Periods span 3 months to 20 years, which is the range OASIS-3 follow-up occupies.
    A textbook ``10000 ** (2i/d)`` schedule is built for token positions in the
    thousands; applied to elapsed years it puts nearly all its resolution where this
    cohort has no data.

    Raises:
        ValueError: If ``embedding_dim`` is odd. The basis is sin/cos pairs, so an odd
            width would silently drop a frequency.
    """

    def __init__(
        self,
        embedding_dim: int,
        scale_days: float = 365.25,
        min_period_years: float = 0.25,
        max_period_years: float = 20.0,
    ) -> None:
        super().__init__()
        if embedding_dim % 2:
            raise ValueError(f"embedding_dim must be even for a sin/cos basis, got {embedding_dim}")
        if not 0 < min_period_years < max_period_years:
            raise ValueError(
                f"need 0 < min_period_years < max_period_years, got "
                f"{min_period_years} and {max_period_years}"
            )
        self.embedding_dim = embedding_dim
        self.scale_days = scale_days

        half = embedding_dim // 2
        periods = torch.logspace(math.log10(min_period_years), math.log10(max_period_years), half)
        # A buffer, not a parameter: it travels with .to(device) and never trains.
        self.register_buffer("frequencies", 2 * math.pi / periods)

    def forward(self, elapsed_days: torch.Tensor) -> torch.Tensor:
        """Args: ``(batch, n_visits)`` elapsed days. Returns ``(batch, n_visits, dim)``."""
        if elapsed_days.dim() != 2:
            raise ValueError(
                f"expected (batch, n_visits) elapsed days, got shape {tuple(elapsed_days.shape)}"
            )
        years = elapsed_days.to(self.frequencies.dtype) / self.scale_days
        angles = years.unsqueeze(-1) * self.frequencies
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class TemporalAttention(nn.Module):
    """Attention over a subject's visit embeddings, conditioned on elapsed time.

    Must honour ``mask``: attending over padded positions fabricates visits that were
    never acquired, which is both a silent bug and, in a study about longitudinal
    change, a fabricated finding.

    Args:
        embedding_dim: Width of the visit embeddings.
        n_heads: Attention heads.
        n_layers: Stacked blocks. Kept small by default — the cohort is small and
            Transformer components overfit it readily (docs/audits/).
        dropout: Applied within each block.
        elapsed_time_scale_days: Passed to :class:`ElapsedTimeEncoding`.

    Raises:
        ValueError: If ``embedding_dim`` is not divisible by ``n_heads``, or the
            encoding cannot be built for this width.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        elapsed_time_scale_days: float = 365.25,
    ) -> None:
        super().__init__()
        if n_heads < 1 or n_layers < 1:
            raise ValueError("n_heads and n_layers must be positive")
        if embedding_dim % n_heads:
            raise ValueError(
                f"embedding_dim {embedding_dim} is not divisible by n_heads {n_heads}; "
                f"multi-head attention splits the width evenly across heads."
            )
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout

        self.time_encoding = ElapsedTimeEncoding(embedding_dim, scale_days=elapsed_time_scale_days)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            # 2x rather than the 2048 default. The cohort is ~450 sequences and the
            # feed-forward block is where most of a Transformer's parameters live.
            dim_feedforward=2 * embedding_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            # Off deliberately. The nested-tensor fast path activates precisely when a
            # padding mask is supplied, and it changes how padded positions are handled.
            # Padding invariance is a correctness property here (see tests), not a
            # performance detail, so the slower uniform path is the right one.
            enable_nested_tensor=False,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        elapsed_days: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Combine visit embeddings into one representation of the target visit.

        Args:
            embeddings: ``(batch, n_visits, embedding_dim)``, target visit last.
            elapsed_days: ``(batch, n_visits)``, 0 at the target visit.
            mask: ``(batch, n_visits)`` boolean, True for real visits.

        Returns:
            ``(batch, embedding_dim)``, read from the final position.

        Raises:
            ValueError: If the three inputs disagree on batch or sequence length, or a
                sequence has no real visit at all.
        """
        if embeddings.dim() != 3:
            raise ValueError(
                f"expected (batch, n_visits, embedding_dim) embeddings, got "
                f"{tuple(embeddings.shape)}"
            )
        batch, n_visits, width = embeddings.shape
        if width != self.embedding_dim:
            raise ValueError(
                f"embeddings are {width}-wide but this module was built for {self.embedding_dim}"
            )
        if elapsed_days.shape != (batch, n_visits) or mask.shape != (batch, n_visits):
            raise ValueError(
                f"elapsed_days {tuple(elapsed_days.shape)} and mask "
                f"{tuple(mask.shape)} must both be {(batch, n_visits)}"
            )
        if not mask.any(dim=1).all():
            raise ValueError(
                "a sequence in this batch has no real visit. Attention over an "
                "entirely padded row has nothing to attend to and returns NaN."
            )

        combined = embeddings + self.time_encoding(elapsed_days)
        # Padded positions are zeroed as well as masked. Attention already excludes
        # them as keys, but zeroing means whatever the collate left there cannot reach
        # a number through any later change to this module.
        combined = combined * mask.unsqueeze(-1)

        encoded = self.blocks(combined, src_key_padding_mask=~mask)

        # The final position. `collate_sequences` left-pads precisely so that this is
        # the target visit for every sample in the batch; right-padding would make this
        # line read padding for most of them while still producing a plausible number.
        last: torch.Tensor = encoded[:, -1, :]
        return last
