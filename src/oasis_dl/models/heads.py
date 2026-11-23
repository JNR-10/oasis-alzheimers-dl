"""Output heads.

The classification head runs at training and inference. The anatomy head runs at
training only and is discarded afterwards (ADR-0007) — which is what makes the second
core ablation clean, since PA and PB have identical inference pathways and any
difference between them comes from the representation the objective shaped.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["AnatomyHead", "ClassificationHead"]


def _check_width(representation: torch.Tensor, expected: int, owner: str) -> None:
    """Refuse a representation of the wrong width here, not three layers down."""
    if representation.dim() != 2 or representation.shape[1] != expected:
        raise ValueError(
            f"{owner} expects (batch, {expected}) representations, got "
            f"{tuple(representation.shape)}"
        )


class ClassificationHead(nn.Module):
    """Representation to a single logit for the impaired class.

    One logit rather than two, so that threshold selection (chosen on validation
    subjects — see docs/protocol/evaluation.md) is a decision about one number.
    """

    def __init__(self, embedding_dim: int = 512, dropout: float = 0.3) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must lie in [0, 1), got {dropout}")
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        # Dropout then a single linear layer. No hidden layer: the representation is
        # already the model's learned feature space, and a wider head on ~450 sequences
        # adds capacity where there is nothing left to learn from.
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(embedding_dim, 1))

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        """Args: ``(batch, embedding_dim)``. Returns ``(batch,)`` logits.

        A raw logit, never a probability. The loss applies its own sigmoid, so squashing
        here would apply it twice and flatten the gradient.
        """
        _check_width(representation, self.embedding_dim, "ClassificationHead")
        logits: torch.Tensor = self.net(representation).squeeze(-1)
        return logits


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
        if embedding_dim < 1 or hidden_dim < 1:
            raise ValueError("embedding_dim and hidden_dim must be positive")
        if not target_names:
            raise ValueError(
                "target_names is required. The auxiliary head predicts a fixed vector, "
                "and naming its entries is what makes a reordered target a loud error "
                "rather than a quietly wrong regression."
            )
        duplicates = sorted({n for n in target_names if target_names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate target names: {', '.join(duplicates)}")

        self.embedding_dim = embedding_dim
        self.target_names = tuple(target_names)
        self.hidden_dim = hidden_dim
        # Linear output, deliberately unbounded: targets are standardized volumes, so
        # they are signed and roughly unit-scale, and a bounded activation would clip
        # exactly the tails where atrophy lives.
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(self.target_names)),
        )

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        """Args: ``(batch, embedding_dim)``. Returns ``(batch, len(target_names))``."""
        _check_width(representation, self.embedding_dim, "AnatomyHead")
        predicted: torch.Tensor = self.net(representation)
        return predicted
