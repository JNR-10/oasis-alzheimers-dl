"""Output heads.

The classification head runs at training and inference. The anatomy head runs at
training only and is discarded afterwards (ADR-0007) — which is what makes the second
core ablation clean, since PA and PB have identical inference pathways and any
difference between them comes from the representation the objective shaped.
"""

from __future__ import annotations

import torch
from torch import nn


class ClassificationHead(nn.Module):
    """Representation to a single logit for the impaired class.

    One logit rather than two, so that threshold selection (chosen on validation
    subjects — see docs/protocol/evaluation.md) is a decision about one number.
    """

    def __init__(self, embedding_dim: int = 512, dropout: float = 0.3) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        raise NotImplementedError

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        """Args: ``(batch, embedding_dim)``. Returns ``(batch,)`` logits."""
        raise NotImplementedError


class AnatomyHead(nn.Module):
    """Representation to predicted structural quantities. Training only.

    Targets must be standardized using **training-partition** statistics only. Fitting
    the scaler on the full cohort would leak test-subject anatomy into training through
    the normalization constants — a quiet leak no subject-level split catches.

    Args:
        embedding_dim: Width of the incoming representation.
        target_names: The structural quantities predicted, in a fixed order. Named
            rather than counted, so that a reordered target vector is a loud error.
            Required — the set comes from the cohort's available derivatives
            (docs/data/README.md, open question 1).
        hidden_dim: Width of the single hidden layer.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        target_names: tuple[str, ...] = (),
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.target_names = target_names
        self.hidden_dim = hidden_dim
        raise NotImplementedError

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        """Args: ``(batch, embedding_dim)``. Returns ``(batch, len(target_names))``."""
        raise NotImplementedError
