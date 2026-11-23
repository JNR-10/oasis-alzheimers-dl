"""The three arms, end to end on the synthetic cohort — #8, #10, #11.

The first complete vertical slice: cohort in, trained model out, metrics table written.
When this is green the pipeline is proven, and every later arm is an addition to a
working system rather than a new system.

What is actually being checked is not that the models *learn* — 40 synthetic subjects
cannot support that claim, and asserting a ROC-AUC here would be asserting noise. It is
that the pipeline is **wired correctly**: that the partitions hold, that the test
partition is untouched until the end, that the threshold comes from validation, and
that the three arms differ only where the protocol says they may.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from oasis_dl.data.dataset import (
    VisitSequenceDataset,
    collate_sequences,
    single_scan_collate,
)
from oasis_dl.data.splits import (
    assign_sequences,
    check_no_leakage,
    partition_subjects,
    subject_label,
)
from oasis_dl.data.synthetic import SyntheticCohort, SyntheticCohortSpec
from oasis_dl.evaluation.metrics import (
    compute_metrics,
    select_threshold,
    write_metrics_table,
)
from oasis_dl.models.arms import LongitudinalModel, SingleScanModel
from oasis_dl.models.encoder import VisitEncoder
from oasis_dl.models.heads import AnatomyHead, ClassificationHead
from oasis_dl.models.temporal import TemporalAttention
from oasis_dl.training.loop import TrainingConfig, evaluate_split, pos_weight_from_labels, train
from oasis_dl.training.losses import AnatomyTargetScaler, CombinedObjective
from oasis_dl.training.seed import seed_everything
from oasis_dl.types import Partition, VisitSequence

SHAPE = (8, 8, 8)
DIM = 16
RATIOS = (0.50, 0.25, 0.25)
TARGETS = ("hippocampus_left_mm3", "hippocampus_right_mm3", "total_gray_matter_mm3")


class _CountingDataset(VisitSequenceDataset):
    """Records how many times its rows are read.

    Stands in for the test partition so the protocol's third control — the test
    partition is read once, *after* training and tuning are complete — becomes an
    observable count rather than a convention.
    """

    reads: int = 0

    def __getitem__(self, index: int) -> object:
        type(self).reads += 1
        return super().__getitem__(index)


@pytest.fixture(scope="module")
def partitioned(tmp_path_factory: pytest.TempPathFactory) -> dict[Partition, list[VisitSequence]]:
    """A cohort split by subject, once, with the leak check run on the result."""
    root = tmp_path_factory.mktemp("arm_cohort")
    cohort = SyntheticCohort(
        root=root,
        spec=SyntheticCohortSpec(
            n_subjects=40,
            max_visits=3,
            impaired_fraction=0.30,
            unlabeled_fraction=0.0,
            volume_shape=SHAPE,
            seed=20261123,
        ),
    ).generate()

    sequences = cohort.sequences(max_length=3)
    labels = {
        subject: subject_label([s for s in sequences if s.subject_id == subject])
        for subject in {s.subject_id for s in sequences}
    }
    grouped = assign_sequences(sequences, partition_subjects(labels, ratios=RATIOS, seed=42))
    check_no_leakage(grouped)
    return grouped


def _loaders(
    grouped: dict[Partition, list[VisitSequence]], collate: object, anatomy: bool = False
) -> tuple[DataLoader[object], DataLoader[object]]:
    def build(partition: Partition) -> DataLoader[object]:
        training = partition is Partition.TRAIN
        dataset = VisitSequenceDataset(grouped[partition], load_anatomy_targets=anatomy, seed=0)
        return DataLoader(
            dataset,
            batch_size=2,
            shuffle=training,
            # Training only. The encoder's BatchNorm cannot take a batch of one once
            # the feature map has collapsed to a single voxel, which is what a partial
            # final batch produces. Shuffling means a different sequence is dropped
            # each epoch, so nothing is permanently excluded from training. Validation
            # and test run in eval mode, where BatchNorm uses running statistics and a
            # batch of one is fine — so they keep every row.
            drop_last=training,
            collate_fn=collate,  # type: ignore[arg-type]
        )

    return build(Partition.TRAIN), build(Partition.VAL)


def _training_anatomy(grouped: dict[Partition, list[VisitSequence]]) -> torch.Tensor:
    """Stack the training partition's anatomy targets, in the order collation uses.

    Training rows only. Fitting the scaler on anything wider would leak test-subject
    anatomy into training through the normalization constants — the quiet kind of leak,
    because no row crosses a partition boundary and no split check can see it.
    """
    rows = [
        sequence.target_visit.anatomy_targets
        for sequence in grouped[Partition.TRAIN]
        if sequence.target_visit.anatomy_targets
    ]
    keys = sorted(rows[0])
    return torch.tensor([[row[key] for key in keys] for row in rows], dtype=torch.float32)


def _evaluate_and_write(
    model: torch.nn.Module,
    grouped: dict[Partition, list[VisitSequence]],
    val_loader: DataLoader[object],
    collate: object,
    arm: str,
    table_path: Path,
) -> None:
    """Choose the threshold on validation, then read the test partition once."""
    device = torch.device("cpu")
    val_labels, val_scores, _ = evaluate_split(model, val_loader, device)
    threshold = select_threshold(val_labels, val_scores)

    held_out = DataLoader(
        VisitSequenceDataset(grouped[Partition.TEST], seed=0),
        batch_size=2,
        collate_fn=collate,  # type: ignore[arg-type]
    )
    labels, scores, _ = evaluate_split(model, held_out, device)
    write_metrics_table({arm: compute_metrics(labels, scores, threshold=threshold)}, table_path)


def _encoder() -> VisitEncoder:
    return VisitEncoder(backbone="resnet10", embedding_dim=DIM, volume_shape=SHAPE)


def _config(**overrides: object) -> TrainingConfig:
    return TrainingConfig(
        max_epochs=2,
        batch_size=2,
        accumulate_grad_batches=2,
        mixed_precision=False,
        early_stopping_patience=5,
        learning_rate=1e-3,
        seed=7,
        **overrides,  # type: ignore[arg-type]
    )


def _objective(
    grouped: dict[Partition, list[VisitSequence]], **kwargs: object
) -> CombinedObjective:
    """Class weight from the training partition only.

    Counting the whole cohort would let the test partition's composition reach the
    objective — a leak that moves every number and that no split check can see.
    """
    weight = pos_weight_from_labels([int(s.label) for s in grouped[Partition.TRAIN]])
    return CombinedObjective(pos_weight=weight, **kwargs)  # type: ignore[arg-type]


@pytest.mark.slow
class TestTheThreeArmsRunEndToEnd:
    def test_b1_trains_and_writes_a_metrics_table(
        self, partitioned: dict[Partition, list[VisitSequence]], tmp_path: Path
    ) -> None:
        """#8. The whole slice: cohort in, trained model out, table written."""
        seed_everything(7, deterministic=False)
        train_loader, val_loader = _loaders(partitioned, single_scan_collate)
        model = SingleScanModel(_encoder(), ClassificationHead(embedding_dim=DIM))

        best = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            objective=_objective(partitioned),
            config=_config(),
            output_dir=tmp_path / "b1",
            device=torch.device("cpu"),
        )
        assert best.exists()
        assert (tmp_path / "b1" / "history.csv").exists()

        # Threshold from validation. Choosing it on test would turn the held-out set
        # into a tuning set and the number would no longer be held out.
        val_labels, val_scores, _ = evaluate_split(model, val_loader, torch.device("cpu"))
        threshold = select_threshold(val_labels, val_scores)

        test_loader = DataLoader(
            VisitSequenceDataset(partitioned[Partition.TEST], seed=0),
            batch_size=2,
            collate_fn=single_scan_collate,
        )
        labels, scores, subjects = evaluate_split(model, test_loader, torch.device("cpu"))
        metrics = compute_metrics(labels, scores, threshold=threshold)

        table = write_metrics_table({"B1": metrics}, tmp_path / "results" / "metrics.csv")
        assert table.exists()
        assert "roc_auc" in table.read_text().splitlines()[0]
        assert metrics.n == len(subjects) == len(partitioned[Partition.TEST])
        assert metrics.threshold == pytest.approx(threshold)

    def test_pa_trains_on_the_same_sequences_through_the_same_loop(
        self, partitioned: dict[Partition, list[VisitSequence]], tmp_path: Path
    ) -> None:
        """#10. PA differs from B1 by the temporal module and the collate function."""
        seed_everything(7, deterministic=False)
        train_loader, val_loader = _loaders(partitioned, collate_sequences)
        model = LongitudinalModel(
            encoder=_encoder(),
            temporal=TemporalAttention(embedding_dim=DIM, n_heads=2, n_layers=1),
            head=ClassificationHead(embedding_dim=DIM),
        )
        best = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            objective=_objective(partitioned),
            config=_config(),
            output_dir=tmp_path / "pa",
            device=torch.device("cpu"),
        )
        assert best.exists()

        table = tmp_path / "pa" / "metrics.csv"
        _evaluate_and_write(model, partitioned, val_loader, collate_sequences, "PA", table)
        assert "PA" in table.read_text()

    def test_pb_trains_with_the_auxiliary_objective(
        self, partitioned: dict[Partition, list[VisitSequence]], tmp_path: Path
    ) -> None:
        """#11. PA plus the anatomy head and the combined objective."""
        seed_everything(7, deterministic=False)
        train_loader, val_loader = _loaders(partitioned, collate_sequences, anatomy=True)

        # Fitted on the training partition alone, before training starts.
        training_targets = _training_anatomy(partitioned)
        assert training_targets.shape[1] == len(TARGETS)
        scaler = AnatomyTargetScaler().fit(training_targets)

        model = LongitudinalModel(
            encoder=_encoder(),
            temporal=TemporalAttention(embedding_dim=DIM, n_heads=2, n_layers=1),
            head=ClassificationHead(embedding_dim=DIM),
            anatomy_head=AnatomyHead(embedding_dim=DIM, target_names=TARGETS),
        )
        best = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            objective=_objective(partitioned, anatomy_weight=0.1),
            config=_config(),
            output_dir=tmp_path / "pb",
            device=torch.device("cpu"),
            anatomy_scaler=scaler,
        )
        assert best.exists()

        # The auxiliary term actually ran: without this the arm is PA wearing a
        # different name, and ablation 2 would compare a model against itself.
        history = (tmp_path / "pb" / "history.csv").read_text().splitlines()
        anatomy_column = history[0].split(",").index("train_anatomy")
        assert float(history[1].split(",")[anatomy_column]) > 0.0

        table = tmp_path / "pb" / "metrics.csv"
        _evaluate_and_write(model, partitioned, val_loader, collate_sequences, "PB", table)
        assert "PB" in table.read_text()


@pytest.mark.slow
class TestTheControlsHold:
    def test_the_test_partition_is_untouched_until_training_is_done(
        self, partitioned: dict[Partition, list[VisitSequence]], tmp_path: Path
    ) -> None:
        """The third control, as a count rather than a convention.

        `train` cannot read the test partition — it is never handed one, which is the
        structural half of the guarantee. What this adds is the other half: that the
        *caller's* order of operations is right. The counter must still be zero after
        training and after the threshold is chosen, and only then rise.
        """
        seed_everything(7, deterministic=False)
        _CountingDataset.reads = 0
        train_loader, val_loader = _loaders(partitioned, single_scan_collate)
        held_out = DataLoader(
            _CountingDataset(partitioned[Partition.TEST], seed=0),
            batch_size=2,
            collate_fn=single_scan_collate,
        )
        model = SingleScanModel(_encoder(), ClassificationHead(embedding_dim=DIM))

        best = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            objective=_objective(partitioned),
            config=_config(),
            output_dir=tmp_path / "control",
            device=torch.device("cpu"),
        )
        assert best.exists()
        assert _CountingDataset.reads == 0, "training read the held-out partition"

        val_labels, val_scores, _ = evaluate_split(model, val_loader, torch.device("cpu"))
        select_threshold(val_labels, val_scores)
        assert _CountingDataset.reads == 0, "threshold selection read the held-out partition"

        evaluate_split(model, held_out, torch.device("cpu"))
        assert _CountingDataset.reads == len(partitioned[Partition.TEST])

    def test_train_accepts_no_test_partition(self) -> None:
        """The structural half, stated so a future signature change has to break it."""
        import inspect

        assert "test" not in " ".join(inspect.signature(train).parameters)

    def test_no_subject_spans_two_partitions(
        self, partitioned: dict[Partition, list[VisitSequence]]
    ) -> None:
        check_no_leakage(partitioned)
        seen: dict[str, Partition] = {}
        for partition, sequences in partitioned.items():
            for sequence in sequences:
                assert seen.setdefault(str(sequence.subject_id), partition) is partition

    def test_every_partition_has_both_classes(
        self, partitioned: dict[Partition, list[VisitSequence]]
    ) -> None:
        """Otherwise ROC-AUC is undefined on test and the arm reports nothing."""
        for partition, sequences in partitioned.items():
            assert len({int(s.label) for s in sequences}) == 2, partition
