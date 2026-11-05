"""The temporal module: a subject's visit embeddings in, one representation out.

Conditioned on elapsed time, because position in a sequence is not time (ADR-0006).
Two subjects with identical embedding sequences four months and four years apart must
not be read identically — atrophy *rate* is the signal.
"""

from __future__ import annotations

import torch
from torch import nn


class ElapsedTimeEncoding(nn.Module):
    """Turns elapsed days into a vector the temporal module can attend over.

    Replaces the standard positional encoding, which would encode "third visit" rather
    than "four years later" (ADR-0006).

    Args:
        embedding_dim: Must match the visit embedding width, since the encoding is
            combined with it.
        scale_days: Divisor putting elapsed days on a sensible numeric range before
            the sinusoidal basis is applied. 365.25 makes one unit one year.
    """

    def __init__(self, embedding_dim: int, scale_days: float = 365.25) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.scale_days = scale_days

    def forward(self, elapsed_days: torch.Tensor) -> torch.Tensor:
        """Args: ``(batch, n_visits)`` elapsed days. Returns ``(batch, n_visits, dim)``."""
        raise NotImplementedError


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
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        raise NotImplementedError

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
            ``(batch, embedding_dim)``.
        """
        raise NotImplementedError
