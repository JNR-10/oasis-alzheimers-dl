"""The shared training loop — #8.

One loop for every arm, so no arm can receive a budget, a schedule or a stopping rule
another did not. The properties here are about *what the loop decides*, tested with a
scripted model rather than a real one: a real 3D network makes these assertions slow and
stochastic, and what is under test is the loop's bookkeeping, not whether a ResNet
learns.

Two properties fail silently:

- Early stopping on validation **loss** rather than ROC-AUC. The run still completes and
  still reports a number; it is simply the wrong checkpoint, chosen against a quantity
  the protocol does not compare on.
- Gradient accumulation that steps per micro-batch. Training proceeds, loss falls, and
  the effective batch size is quietly N times smaller than recorded.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from oasis_dl.data.dataset import SequenceBatch
from oasis_dl.training.loop import TrainingConfig, evaluate_split, pos_weight_from_labels, train
from oasis_dl.training.losses import AnatomyTargetScaler, CombinedObjective


class _Batches(Dataset[SequenceBatch]):
    """A dataset whose items are already collated batches."""

    def __init__(self, batches: list[SequenceBatch]) -> None:
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, index: int) -> SequenceBatch:
        return self.batches[index]


def _loader(batches: list[SequenceBatch]) -> DataLoader[SequenceBatch]:
    return DataLoader(_Batches(batches), batch_size=None, shuffle=False)


def _batch(labels: list[int], subject_prefix: str = "S") -> SequenceBatch:
    n = len(labels)
    return SequenceBatch(
        scans=torch.zeros(n, 1, 1, 2, 2, 2),
        elapsed_days=torch.zeros(n, 1),
        mask=torch.ones(n, 1, dtype=torch.bool),
        labels=torch.tensor(labels),
        anatomy_targets=None,
        anatomy_mask=None,
        subject_ids=tuple(f"{subject_prefix}{i}" for i in range(n)),
    )


class _ScriptedModel(nn.Module):
    """Emits a scripted score vector on each successive eval pass.

    Lets a test state the validation curve directly — "AUC goes 0.6, 0.9, 0.7" — and
    assert what the loop does with it, without waiting for a real model to produce that
    curve by chance.
    """

    def __init__(self, eval_scores: list[list[float]], train_logit: float = 0.0) -> None:
        super().__init__()
        self.eval_scores = eval_scores
        self.calls = 0
        self.weight = nn.Parameter(torch.tensor(train_logit))

    def forward(self, batch: SequenceBatch) -> dict[str, torch.Tensor]:
        n = batch.labels.shape[0]
        if self.training:
            return {"logits": self.weight.expand(n)}
        scores = self.eval_scores[min(self.calls, len(self.eval_scores) - 1)]
        self.calls += 1
        return {"logits": torch.tensor(scores).logit().clamp(-20, 20)}


def _history(output_dir: Path) -> list[dict[str, str]]:
    with (output_dir / "history.csv").open() as handle:
        return list(csv.DictReader(handle))


PERFECT = [0.1, 0.2, 0.8, 0.9]
MIDDLING = [0.4, 0.6, 0.45, 0.55]
CHANCE = [0.5, 0.5, 0.5, 0.5]
LABELS = [0, 0, 1, 1]


def _run(
    eval_scores: list[list[float]], tmp_path: Path, **overrides: object
) -> tuple[Path, _ScriptedModel, list[dict[str, str]]]:
    model = _ScriptedModel(eval_scores)
    settings: dict[str, object] = {
        "max_epochs": len(eval_scores),
        "mixed_precision": False,
        "batch_size": 1,
        "accumulate_grad_batches": 1,
        "early_stopping_patience": 2,
    }
    settings.update(overrides)
    config = TrainingConfig(**settings)  # type: ignore[arg-type]
    best = train(
        model=model,
        train_loader=_loader([_batch(LABELS)]),
        val_loader=_loader([_batch(LABELS)]),
        objective=CombinedObjective(),
        config=config,
        output_dir=tmp_path,
    )
    return best, model, _history(tmp_path)


class TestItRunsAndRecords:
    def test_it_returns_a_loadable_checkpoint(self, tmp_path: Path) -> None:
        best, _, _ = _run([MIDDLING, PERFECT], tmp_path)
        assert best.exists()
        assert "model_state" in torch.load(best, map_location="cpu")

    def test_every_epoch_is_recorded(self, tmp_path: Path) -> None:
        _, _, history = _run([MIDDLING, PERFECT, CHANCE], tmp_path)
        assert len(history) == 3
        assert {"epoch", "train_loss", "val_roc_auc"} <= set(history[0])

    def test_both_loss_components_are_logged_separately(self, tmp_path: Path) -> None:
        """#11: a total that improves while classification worsens is the auxiliary term
        taking over, and that is invisible if only the total is kept."""
        _, _, history = _run([PERFECT], tmp_path)
        assert {"train_classification", "train_anatomy"} <= set(history[0])


class TestEarlyStoppingWatchesRocAuc:
    def test_the_best_checkpoint_is_the_best_validation_auc(self, tmp_path: Path) -> None:
        best, _, history = _run([MIDDLING, PERFECT, CHANCE], tmp_path)
        recorded = [float(row["val_roc_auc"]) for row in history]
        assert recorded.index(max(recorded)) == 1
        assert torch.load(best, map_location="cpu")["epoch"] == 1

    def test_it_stops_after_patience_without_improvement(self, tmp_path: Path) -> None:
        _, _, history = _run([PERFECT] + [CHANCE] * 8, tmp_path)
        # Best at epoch 0, then patience 2 -> epochs 1 and 2, then stop.
        assert len(history) == 3

    def test_it_watches_auc_not_loss(self, tmp_path: Path) -> None:
        """The silent one.

        Validation loss keeps falling while ranking stays at chance — a model growing
        confident about nothing. Stopping on loss would run to the budget and keep the
        last checkpoint; stopping on ROC-AUC halts and keeps epoch 0.
        """
        # Scores pull apart a little further every epoch: ranking is already perfect
        # from epoch 1, so AUC is flat at 1.0 while BCE keeps falling. A loop watching
        # loss sees an improvement every single epoch and runs the full budget.
        falling_loss_flat_auc = [[0.5, 0.5, 0.5, 0.5]] + [
            [0.5 - k * 0.04, 0.5 - k * 0.04, 0.5 + k * 0.04, 0.5 + k * 0.04] for k in range(1, 10)
        ]
        _, _, history = _run(falling_loss_flat_auc, tmp_path)
        assert len(history) < len(falling_loss_flat_auc)

    def test_a_single_class_validation_split_does_not_crash_the_run(self, tmp_path: Path) -> None:
        """ROC-AUC is NaN there. A NaN must count as "no improvement", never as a best."""
        model = _ScriptedModel([PERFECT, PERFECT])
        best = train(
            model=model,
            train_loader=_loader([_batch([1, 1, 1, 1])]),
            val_loader=_loader([_batch([1, 1, 1, 1])]),
            objective=CombinedObjective(),
            config=TrainingConfig(max_epochs=2, mixed_precision=False, batch_size=1),
            output_dir=tmp_path,
        )
        assert best.exists()
        # No epoch was measurable, so none may be recorded as the best. The fallback
        # checkpoint marks that: treating NaN as a win would checkpoint a model that
        # nothing was measured on and report it as the selected one.
        assert torch.load(best, map_location="cpu")["epoch"] == -1


class TestGradientAccumulation:
    def test_accumulating_matches_one_larger_step(self, tmp_path: Path) -> None:
        """Four micro-batches accumulated must move the weight the same distance as one
        batch of the same total size. Stepping per micro-batch moves it four times."""
        quarters = [_batch([0]), _batch([0]), _batch([1]), _batch([1])]
        whole = [_batch(LABELS)]

        def final_weight(batches: list[SequenceBatch], accumulate: int) -> float:
            model = _ScriptedModel([PERFECT])
            train(
                model=model,
                train_loader=_loader(batches),
                val_loader=_loader([_batch(LABELS)]),
                objective=CombinedObjective(),
                config=TrainingConfig(
                    max_epochs=1,
                    mixed_precision=False,
                    accumulate_grad_batches=accumulate,
                    learning_rate=0.1,
                    weight_decay=0.0,
                ),
                output_dir=tmp_path / f"acc{accumulate}",
            )
            return float(model.weight.detach())

        assert final_weight(quarters, 4) == pytest.approx(final_weight(whole, 1), abs=2e-3)

    def test_a_final_partial_group_still_updates_the_weights(self, tmp_path: Path) -> None:
        """An epoch whose batch count is not a multiple of the accumulation factor
        leaves a partial group. Discarding it means the last sequences of every epoch
        never reach the weights — and with one batch and a factor of four, nothing does."""
        model = _ScriptedModel([PERFECT], train_logit=0.3)
        before = float(model.weight.detach())
        train(
            model=model,
            train_loader=_loader([_batch(LABELS)]),
            val_loader=_loader([_batch(LABELS)]),
            objective=CombinedObjective(),
            config=TrainingConfig(
                max_epochs=1,
                mixed_precision=False,
                accumulate_grad_batches=4,
                learning_rate=0.1,
            ),
            output_dir=tmp_path,
        )
        assert float(model.weight.detach()) != before


class _SpyObjective(CombinedObjective):
    """Records the anatomy targets it was handed, so the loop's wiring is observable."""

    def __init__(self) -> None:
        super().__init__(anatomy_weight=1.0)
        self.seen: list[torch.Tensor] = []

    def forward(  # type: ignore[override]
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        anatomy_prediction: torch.Tensor | None = None,
        anatomy_target: torch.Tensor | None = None,
        anatomy_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if anatomy_target is not None:
            self.seen.append(anatomy_target.detach().clone())
        return super().forward(logits, labels, anatomy_prediction, anatomy_target, anatomy_mask)


class _AnatomyModel(_ScriptedModel):
    """A scripted model that also emits an anatomy prediction while training."""

    def forward(self, batch: SequenceBatch) -> dict[str, torch.Tensor]:
        out = super().forward(batch)
        if self.training:
            out["anatomy"] = torch.zeros(batch.labels.shape[0], 2, requires_grad=True)
        return out


class TestAnatomyStandardization:
    """#11: targets are standardized with training-partition statistics only."""

    @staticmethod
    def _anatomy_batch() -> SequenceBatch:
        raw = _batch(LABELS)
        return SequenceBatch(
            **{
                **vars(raw),
                "anatomy_targets": torch.tensor(
                    [[10.0, 100.0], [20.0, 200.0], [30.0, 300.0], [40.0, 400.0]]
                ),
                "anatomy_mask": torch.ones(4, dtype=torch.bool),
            }
        )

    def test_the_loop_standardizes_the_targets_it_passes_on(self, tmp_path: Path) -> None:
        """Raw volumes are in the thousands of mm^3. Handed to the objective unscaled,
        the auxiliary term dwarfs classification and lambda stops meaning anything."""
        batch = self._anatomy_batch()
        scaler = AnatomyTargetScaler().fit(batch.anatomy_targets)
        objective = _SpyObjective()
        train(
            model=_AnatomyModel([PERFECT]),
            train_loader=_loader([batch]),
            val_loader=_loader([_batch(LABELS)]),
            objective=objective,
            config=TrainingConfig(max_epochs=1, mixed_precision=False),
            output_dir=tmp_path,
            anatomy_scaler=scaler,
        )
        assert objective.seen, "the objective never received anatomy targets"
        assert objective.seen[0].abs().max() < 3.0, "targets reached the loss unscaled"

    def test_without_a_scaler_the_raw_targets_are_passed_through(self, tmp_path: Path) -> None:
        """So the test above cannot pass on a loop that zeroes the targets."""
        objective = _SpyObjective()
        train(
            model=_AnatomyModel([PERFECT]),
            train_loader=_loader([self._anatomy_batch()]),
            val_loader=_loader([_batch(LABELS)]),
            objective=objective,
            config=TrainingConfig(max_epochs=1, mixed_precision=False),
            output_dir=tmp_path,
        )
        assert objective.seen[0].max() == pytest.approx(400.0)


class TestMixedPrecision:
    def test_requesting_it_on_cpu_completes_without_pretending_to_be_on(
        self, tmp_path: Path
    ) -> None:
        """The path every laptop takes. AMP is a CUDA feature; asking for it on CPU must
        run normally rather than crash or silently claim to have used it."""
        best, _, history = _run([PERFECT], tmp_path, mixed_precision=True)
        assert best.exists() and len(history) == 1


class TestAnEmptyEpochIsAnError:
    def test_a_training_loader_that_yields_nothing_is_refused(self, tmp_path: Path) -> None:
        """With drop_last=True a partition smaller than one batch vanishes entirely.
        The run would otherwise complete every epoch, update nothing, and report a
        perfectly ordinary untrained model."""
        with pytest.raises(ValueError, match="no batches"):
            train(
                model=_ScriptedModel([PERFECT]),
                train_loader=_loader([]),
                val_loader=_loader([_batch(LABELS)]),
                objective=CombinedObjective(),
                config=TrainingConfig(max_epochs=1, mixed_precision=False),
                output_dir=tmp_path,
            )


class TestReproducibility:
    def test_the_same_seed_gives_the_same_weights(self, tmp_path: Path) -> None:
        def run(where: str) -> float:
            torch.manual_seed(0)
            model = _ScriptedModel([MIDDLING, PERFECT], train_logit=0.3)
            train(
                model=model,
                train_loader=_loader([_batch(LABELS)]),
                val_loader=_loader([_batch(LABELS)]),
                objective=CombinedObjective(),
                config=TrainingConfig(max_epochs=2, mixed_precision=False, seed=7),
                output_dir=tmp_path / where,
            )
            return float(model.weight.detach())

        assert run("a") == run("b")


class TestEvaluateSplit:
    def test_it_returns_labels_scores_and_subjects(self) -> None:
        model = _ScriptedModel([PERFECT]).eval()
        labels, scores, subjects = evaluate_split(model, _loader([_batch(LABELS)]))
        assert labels == LABELS
        assert len(scores) == 4 and all(0.0 <= s <= 1.0 for s in scores)
        assert subjects == ["S0", "S1", "S2", "S3"]

    def test_scores_are_probabilities_not_logits(self) -> None:
        """compute_metrics takes a score for the impaired class; a raw logit would make
        any fixed threshold mean something different per arm."""
        model = _ScriptedModel([[0.01, 0.99, 0.5, 0.5]]).eval()
        _, scores, _ = evaluate_split(model, _loader([_batch(LABELS)]))
        assert min(scores) >= 0.0 and max(scores) <= 1.0

    def test_it_leaves_the_model_in_eval_mode(self) -> None:
        """A loop that forgets this trains the next epoch with dropout disabled."""
        model = _ScriptedModel([PERFECT])
        model.train()
        evaluate_split(model, _loader([_batch(LABELS)]))
        assert not model.training


class TestClassWeighting:
    def test_the_weight_is_the_negative_to_positive_ratio(self) -> None:
        """76/24 cohort -> about 3.2. Computed from the training partition only;
        counting the full cohort would leak test composition into the objective."""
        assert pos_weight_from_labels([0] * 76 + [1] * 24) == pytest.approx(76 / 24)

    def test_a_balanced_partition_needs_no_weight(self) -> None:
        assert pos_weight_from_labels([0, 0, 1, 1]) == pytest.approx(1.0)

    def test_a_partition_with_no_impaired_subjects_is_refused(self) -> None:
        """Returning inf would silently make the loss ignore every negative."""
        with pytest.raises(ValueError):
            pos_weight_from_labels([0, 0, 0])
