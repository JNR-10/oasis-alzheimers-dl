"""The training loop. Shared by every arm.

One loop, so that an arm cannot receive a training budget, an early-stopping rule, or a
schedule that another arm did not. Where an architecture forces a difference, the
difference is recorded in the run and reported (docs/protocol/README.md).
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader

from oasis_dl.evaluation.metrics import compute_metrics
from oasis_dl.training.losses import AnatomyTargetScaler
from oasis_dl.utils.provenance import PROVENANCE_FILENAME, RunProvenance

__all__ = [
    "TrainingConfig",
    "evaluate_split",
    "pos_weight_from_labels",
    "train",
]

HISTORY_FILENAME = "history.csv"

THRESHOLD_FREE_METRICS = frozenset({"roc_auc"})
"""Metrics early stopping may watch.

Every other metric depends on a threshold, and during training there is none: the
operating point is chosen on validation subjects *after* training (#4). Watching one
mid-training means watching it at an arbitrary 0.5 — a stopping rule tuned to a
threshold that will never be used.
"""
BEST_CHECKPOINT_FILENAME = "best.pt"


def pos_weight_from_labels(labels: Iterable[int]) -> float:
    """Class weight for the impaired class, from the **training** partition only.

    Counting the full cohort would let the test partition's class composition reach the
    objective — a leak that moves every number and that no subject-level check sees,
    because no row crosses a boundary.

    Returns:
        ``n_negative / n_positive``, so 1.0 on a balanced partition.

    Raises:
        ValueError: If either class is absent. Returning ``inf`` would make the loss
            ignore one class entirely while training proceeded normally.
    """
    values = list(labels)
    positives = sum(1 for label in values if int(label) == 1)
    negatives = len(values) - positives
    if positives == 0 or negatives == 0:
        raise ValueError(
            f"training partition has {positives} impaired and {negatives} normal "
            f"subjects; a class weight needs both."
        )
    return negatives / positives


def _grad_scaler(enabled: bool) -> Any:
    """A GradScaler that works across the torch versions this project spans.

    ``torch.cuda.amp.GradScaler`` is the only spelling in torch 2.2 and is deprecated by
    2.15; ``torch.amp.GradScaler`` is the reverse. The repo declares ``torch>=2.2`` and
    CI resolves the newest, so both have to work or the loop is untestable on one of
    them.
    """
    modern = getattr(torch.amp, "GradScaler", None)
    if modern is not None:
        return modern("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


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
    standardize_anatomy: bool = False,
    provenance: RunProvenance | None = None,
) -> Path:
    """Train one arm and return the path of its best checkpoint.

    Early stopping and every other hyperparameter are decided on ``val_loader``. The
    test partition is not touched here at all — it is read once, by
    :mod:`oasis_dl.cli.evaluate`, after training is complete.

    Args:
        standardize_anatomy: Arm PB only. Standardizes anatomy targets with statistics
            fitted **inside this function, from ``train_loader``'s own sequences**
            (ADR-0007). The loop fits the scaler rather than accepting one, because a
            scaler handed in cannot say what it was fitted on — and fitting on the full
            cohort is the quiet leak AGENTS.md names: no row crosses a partition, so no
            split check sees it. Here the only data within reach is the training
            partition, so the wrong fit is not expressible.
        provenance: Written to ``output_dir`` **before the first batch** (ADR-0008), so
            a run that crashes still leaves a record of what was attempted. A record
            written at the end only ever describes runs that succeeded, which are the
            ones least in need of explaining.

    Returns:
        Path to the checkpoint with the best validation metric.

    Raises:
        ValueError: If ``early_stopping_metric`` depends on a decision threshold.
    """
    if config.early_stopping_metric not in THRESHOLD_FREE_METRICS:
        raise ValueError(
            f"early_stopping_metric {config.early_stopping_metric!r} depends on a "
            f"decision threshold, and none exists until training is over — it is chosen "
            f"on validation subjects afterwards. Watch one of "
            f"{sorted(THRESHOLD_FREE_METRICS)} instead."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Before anything can fail. This is the whole reason provenance is captured at the
    # start of a run rather than written out with the results.
    if provenance is not None:
        provenance.write(output_dir / PROVENANCE_FILENAME)

    anatomy_scaler = _fit_anatomy_scaler(train_loader) if standardize_anatomy else None

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    # Mixed precision is a CUDA feature. Enabling it on CPU would silently do nothing
    # while the run record claimed otherwise.
    use_amp = config.mixed_precision and device.type == "cuda"
    scaler = _grad_scaler(use_amp)
    accumulate = max(1, config.accumulate_grad_batches)

    best_value = -math.inf
    best_path = output_dir / BEST_CHECKPOINT_FILENAME
    since_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(config.max_epochs):
        # Augmentation must differ between epochs or it degenerates into fixed
        # preprocessing; the dataset owns that draw.
        set_epoch = getattr(getattr(train_loader, "dataset", None), "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)

        model.train()
        running = {"total": 0.0, "classification": 0.0, "anatomy": 0.0}
        seen = 0
        optimizer.zero_grad(set_to_none=True)

        for index, raw in enumerate(train_loader):
            batch = raw.to(device)
            targets = batch.anatomy_targets
            if targets is not None and anatomy_scaler is not None:
                targets = anatomy_scaler.transform(targets)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(batch)
                losses = objective(
                    outputs["logits"],
                    batch.labels,
                    outputs.get("anatomy"),
                    targets,
                    anatomy_mask=batch.anatomy_mask,
                )
            # Divided before backward, so the accumulated gradient is the *mean* over
            # the effective batch rather than its sum. Without this the effective
            # learning rate scales with accumulate_grad_batches.
            scaler.scale(losses["total"] / accumulate).backward()

            if (index + 1) % accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            for key in running:
                running[key] += float(losses[key].detach())
            seen += 1

        # A final partial group would otherwise be computed and thrown away, so the
        # last few sequences of every epoch would never reach the weights.
        if seen % accumulate:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if seen == 0:
            raise ValueError(
                "the training loader yielded no batches. With drop_last=True a "
                "partition smaller than one batch vanishes entirely, and the run would "
                "otherwise proceed through every epoch updating nothing."
            )

        labels, scores, _ = evaluate_split(model, val_loader, device)
        # 0.5 is a placeholder the watched metric never reads: only threshold-free
        # metrics are allowed here (checked on entry), and ROC-AUC ignores it.
        metrics = compute_metrics(labels, scores, threshold=0.5)
        watched = float(getattr(metrics, config.early_stopping_metric))

        divisor = max(seen, 1)
        record = {
            "epoch": float(epoch),
            "train_loss": running["total"] / divisor,
            "train_classification": running["classification"] / divisor,
            "train_anatomy": running["anatomy"] / divisor,
            "val_loss": _validation_loss(labels, scores),
            "val_roc_auc": metrics.roc_auc,
            f"val_{config.early_stopping_metric}": watched,
        }
        history.append(record)
        _write_history(output_dir / HISTORY_FILENAME, history)

        # NaN counts as no improvement, never as a best. A single-class validation
        # split leaves ROC-AUC undefined, and treating that as a win would checkpoint a
        # model nothing was measured on.
        if math.isfinite(watched) and watched > best_value + 1e-6:
            best_value = watched
            since_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    config.early_stopping_metric: watched,
                    "config": vars(config),
                    # Kept with the weights so a prediction can be reported back in
                    # mm^3, and so the statistics the run used are recoverable.
                    "anatomy_scaler": _scaler_state(anatomy_scaler),
                },
                best_path,
            )
        else:
            since_improvement += 1
            if since_improvement >= config.early_stopping_patience:
                break

    if not best_path.exists():
        # Every epoch was unmeasurable. The run still has to leave a checkpoint, or the
        # failure surfaces much later as a missing file.
        torch.save(
            {"model_state": model.state_dict(), "epoch": -1, "config": vars(config)},
            best_path,
        )
    return best_path


def _fit_anatomy_scaler(train_loader: DataLoader[object]) -> AnatomyTargetScaler:
    """Fit the anatomy scaler from the training loader's own sequences.

    Reads the sequence records, not the scans, so it costs a pass over a list rather
    than over the data.

    Raises:
        ValueError: If the training dataset does not expose its sequences. Fitting from
            anything else would reopen the question this function exists to close.
    """
    sequences = getattr(getattr(train_loader, "dataset", None), "sequences", None)
    if sequences is None:
        raise ValueError(
            "standardize_anatomy needs a training dataset that exposes `.sequences`, so "
            "the scaler is fitted from the training partition and nothing else."
        )
    return AnatomyTargetScaler.fit_from_sequences(sequences)


def _scaler_state(scaler: AnatomyTargetScaler | None) -> dict[str, list[float]] | None:
    """The fitted constants as plain lists, so a checkpoint round-trips through JSON."""
    if scaler is None or scaler.mean is None or scaler.scale is None:
        return None
    return {"mean": scaler.mean.tolist(), "scale": scaler.scale.tolist()}


def _validation_loss(labels: list[int], scores: list[float]) -> float:
    """Unweighted BCE on the validation partition, for the history only.

    Logged rather than watched. Early stopping reads ROC-AUC because that is what the
    report compares on, and the two genuinely diverge — a model can grow confident
    about nothing, lowering loss while its ranking stays at chance.
    """
    probabilities = torch.tensor(scores, dtype=torch.float32).clamp(1e-7, 1 - 1e-7)
    return float(
        functional.binary_cross_entropy(probabilities, torch.tensor(labels, dtype=torch.float32))
    )


def _write_history(path: Path, history: list[dict[str, float]]) -> None:
    """Rewrite the history after every epoch, so a crashed run still leaves one."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


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

    Leaves the model in eval mode. Returns **probabilities**, not logits: a fixed
    threshold has to mean the same thing for every arm.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    labels: list[int] = []
    scores: list[float] = []
    subjects: list[str] = []
    with torch.no_grad():
        for raw in loader:
            batch = raw.to(device)
            outputs = model(batch)
            scores.extend(torch.sigmoid(outputs["logits"].float()).cpu().tolist())
            labels.extend(int(value) for value in batch.labels.cpu().tolist())
            subjects.extend(str(subject) for subject in batch.subject_ids)
    return labels, scores, subjects
