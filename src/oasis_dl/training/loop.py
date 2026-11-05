"""The training loop. Shared by every arm.

One loop, so that an arm cannot receive a training budget, an early-stopping rule, or a
schedule that another arm did not. Where an architecture forces a difference, the
difference is recorded in the run and reported (docs/protocol/README.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainingConfig:
    """Training hyperparameters.

    Attributes:
        max_epochs: Upper bound; early stopping usually ends the run sooner.
        learning_rate: Initial rate.
        weight_decay: L2 regularization. Matters here — the cohort is small and 3D
            models overfit it readily.
        batch_size: Small by necessity. 3D volumes times sequence length dominate GPU
            memory, which is why mixed precision and gradient accumulation exist below.
        accumulate_grad_batches: Effective batch size multiplier when memory binds.
        mixed_precision: Halves activation memory on supported GPUs.
        early_stopping_patience: Epochs without validation improvement before stopping.
        early_stopping_metric: Watched metric. ROC-AUC rather than loss, since the
            reported comparison is on ROC-AUC and the two can diverge.
        seed: Recorded with the run.
    """

    max_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 2
    accumulate_grad_batches: int = 8
    mixed_precision: bool = True
    early_stopping_patience: int = 15
    early_stopping_metric: str = "roc_auc"
    seed: int = 42


def train(
    model: nn.Module,
    train_loader: DataLoader[object],
    val_loader: DataLoader[object],
    objective: nn.Module,
    config: TrainingConfig,
    output_dir: Path,
    device: torch.device | None = None,
) -> Path:
    """Train one arm and return the path of its best checkpoint.

    Early stopping and every other hyperparameter are decided on ``val_loader``. The
    test partition is not touched here at all — it is read once, by
    :mod:`oasis_dl.cli.evaluate`, after training is complete.

    Returns:
        Path to the checkpoint with the best validation metric.
    """
    raise NotImplementedError("Blocked on the dataset and model implementations.")


def evaluate_split(
    model: nn.Module,
    loader: DataLoader[object],
    device: torch.device | None = None,
) -> tuple[list[int], list[float], list[str]]:
    """Run inference over a partition.

    Returns labels, scores, and subject ids — the three things
    :mod:`oasis_dl.evaluation.metrics` and :mod:`oasis_dl.evaluation.bootstrap` need.
    Returning raw predictions rather than metrics keeps scoring in one place and keeps
    this function usable for validation and test alike.
    """
    raise NotImplementedError("Blocked on the dataset and model implementations.")
