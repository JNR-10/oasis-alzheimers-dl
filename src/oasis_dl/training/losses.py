"""The combined objective of ADR-0007.

``L_total = L_classification + lambda * L_anatomy``

``lambda`` is tuned on **validation** subjects. Tuning it on test subjects would turn
the second core ablation into a search for a favorable number rather than a test of
whether the auxiliary objective helps.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

__all__ = ["AnatomyTargetScaler", "CombinedObjective"]


class AnatomyTargetScaler:
    """Standardizes anatomy targets using **training-partition** statistics only.

    Fitting on the full cohort would leak test-subject anatomy into training through the
    normalization constants. It is the quiet kind of leak: no row crosses a partition
    boundary, so no subject-level check catches it, and the only symptom is a number
    that is slightly too good.

    Not an ``nn.Module``. These constants are fitted, not learned, and making them
    parameters would let an optimizer move them.
    """

    def __init__(self) -> None:
        self.mean: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None

    def fit(self, training_targets: torch.Tensor) -> AnatomyTargetScaler:
        """Record per-target mean and spread from the training partition.

        Args:
            training_targets: ``(n_training_sequences, n_targets)``. Training rows only.
        """
        if training_targets.dim() != 2 or training_targets.shape[0] < 1:
            raise ValueError(
                f"expected (n_rows, n_targets) training targets, got "
                f"{tuple(training_targets.shape)}"
            )
        self.mean = training_targets.mean(dim=0)
        spread = training_targets.std(dim=0, unbiased=False)
        # A target identical across every training subject has zero spread. Dividing by
        # it yields inf, and the auxiliary loss is NaN for the rest of the run. Such a
        # target carries no signal anyway, so it is centred and left unscaled.
        self.scale = torch.where(spread > 0, spread, torch.ones_like(spread))
        return self

    def _fitted(self, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mean is None or self.scale is None:
            raise RuntimeError(
                "scaler is not fitted. Call fit() on the training partition's targets "
                "first — fitting it on anything wider leaks test anatomy into training."
            )
        if targets.dim() != 2 or targets.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"expected (n_rows, {self.mean.shape[0]}) targets, got {tuple(targets.shape)}"
            )
        return self.mean, self.scale

    def transform(self, targets: torch.Tensor) -> torch.Tensor:
        """Standardize with the fitted constants. Used for every partition.

        The constants follow the batch onto its device: they are fitted once on the CPU
        and a training batch is already on the GPU by the time it reaches here.
        """
        mean, scale = self._fitted(targets)
        return (targets - mean.to(targets.device)) / scale.to(targets.device)

    def inverse_transform(self, targets: torch.Tensor) -> torch.Tensor:
        """Back to the original units, for reporting a prediction in mm^3."""
        mean, scale = self._fitted(targets)
        return targets * scale.to(targets.device) + mean.to(targets.device)


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
        if anatomy_weight < 0:
            raise ValueError(f"anatomy_weight is a magnitude, got {anatomy_weight}")
        if pos_weight is not None and pos_weight <= 0:
            raise ValueError(f"pos_weight must be positive, got {pos_weight}")
        self.anatomy_weight = anatomy_weight
        self.pos_weight = pos_weight

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        anatomy_prediction: torch.Tensor | None = None,
        anatomy_target: torch.Tensor | None = None,
        anatomy_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the objective and its parts.

        Args:
            logits: ``(batch,)`` impaired-class logits.
            labels: ``(batch,)`` binary labels.
            anatomy_prediction: ``(batch, n_targets)``, or None outside arm PB.
            anatomy_target: ``(batch, n_targets)``, standardized with
                training-partition statistics only (:class:`AnatomyTargetScaler`).
            anatomy_mask: ``(batch,)`` boolean. False marks a sequence with no
                FreeSurfer output, excluded from the auxiliary term alone (ADR-0009).

        Returns:
            ``{"total", "classification", "anatomy"}``. The parts are returned so both
            can be logged — a total that improves while classification worsens is the
            auxiliary term taking over, and that is invisible if only the total is kept.
        """
        weight = (
            None
            if self.pos_weight is None
            else torch.tensor(self.pos_weight, dtype=logits.dtype, device=logits.device)
        )
        classification = functional.binary_cross_entropy_with_logits(
            logits, labels.to(logits.dtype), pos_weight=weight
        )
        anatomy = self._anatomy_term(anatomy_prediction, anatomy_target, anatomy_mask)

        # Branch rather than add ``0.0 * anatomy``. The reduction must be *exact* at
        # lambda = 0 so arms PA and PB share one objective, and multiplying by zero is
        # only exact while the auxiliary term stays finite — one inf in a target and
        # 0 * inf is NaN, which would take PA down with PB.
        total = (
            classification
            if self.anatomy_weight == 0
            else classification + self.anatomy_weight * anatomy
        )
        return {"total": total, "classification": classification, "anatomy": anatomy}

    @staticmethod
    def _anatomy_term(
        prediction: torch.Tensor | None,
        target: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Mean squared error over the sequences that actually carry targets.

        Masked per sequence, never per batch (ADR-0009). Not every session ships
        FreeSurfer output, so a training subject without targets still contributes
        classification loss and drops out of this term alone — otherwise whether the
        auxiliary objective runs depends on how the sampler filled the batch.
        """
        if prediction is None or target is None:
            return torch.zeros((), dtype=torch.float32)
        if prediction.shape != target.shape:
            raise ValueError(
                f"anatomy prediction {tuple(prediction.shape)} and target "
                f"{tuple(target.shape)} must match"
            )
        if mask is None:
            return functional.mse_loss(prediction, target)

        if mask.shape != (prediction.shape[0],):
            raise ValueError(
                f"anatomy_mask must be ({prediction.shape[0]},), got {tuple(mask.shape)}"
            )
        keep = mask.to(prediction.dtype).unsqueeze(-1)
        counted = keep.sum() * prediction.shape[1]
        if counted == 0:
            # Nobody in this batch has targets. Normal, not an error — and returning a
            # zero rather than 0/0 keeps the total finite.
            return torch.zeros((), dtype=prediction.dtype, device=prediction.device)
        # Masked rows are multiplied out before the sum, so whatever collate left in
        # them cannot reach the loss through the mean.
        return ((prediction - target).pow(2) * keep).sum() / counted
