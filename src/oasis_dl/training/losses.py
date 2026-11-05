"""The combined objective of ADR-0007.

``L_total = L_classification + lambda * L_anatomy``

``lambda`` is tuned on **validation** subjects. Tuning it on test subjects would turn
the second core ablation into a search for a favorable number rather than a test of
whether the auxiliary objective helps.
"""

from __future__ import annotations

import torch
from torch import nn


class CombinedObjective(nn.Module):
    """Classification loss, optionally plus weighted anatomy regression.

    With ``anatomy_weight`` at 0 or no anatomy targets supplied, this must reduce
    **exactly** to the classification loss — so arms PA and PB share one implementation
    and the ablation cannot drift through two diverging copies of the objective.

    Args:
        anatomy_weight: ``lambda``. 0 disables the auxiliary term.
        pos_weight: Weight on the impaired class, for imbalance. Computed from the
            **training** partition only.
    """

    def __init__(
        self,
        anatomy_weight: float = 0.0,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.anatomy_weight = anatomy_weight
        self.pos_weight = pos_weight
        raise NotImplementedError

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        anatomy_prediction: torch.Tensor | None = None,
        anatomy_target: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the objective and its parts.

        Args:
            logits: ``(batch,)`` impaired-class logits.
            labels: ``(batch,)`` binary labels.
            anatomy_prediction: ``(batch, n_targets)``, or None outside arm PB.
            anatomy_target: ``(batch, n_targets)``, standardized with
                training-partition statistics only.

        Returns:
            ``{"total", "classification", "anatomy"}``. The parts are returned so both
            can be logged — a total that improves while classification worsens is the
            auxiliary term taking over, and that is invisible if only the total is kept.
        """
        raise NotImplementedError
