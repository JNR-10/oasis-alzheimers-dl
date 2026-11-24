"""Loading visit sequences and batching them.

The collation detail that carries the most weight is **left-padding**. Right-padding
would put the target visit at a different index for every sequence in the batch, and a
temporal module reading the final position would read padding for most of them — while
still training, still converging, and still producing a plausible number.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest
import torch

from oasis_dl.data.dataset import (
    LoadedSequence,
    SequenceBatch,
    VisitSequenceDataset,
    collate_sequences,
    single_scan_collate,
)
from oasis_dl.data.synthetic import SyntheticCohort, SyntheticCohortSpec
from oasis_dl.types import Label, SessionId, SubjectId, Visit, VisitSequence


@pytest.fixture(scope="module")
def cohort(tmp_path_factory: pytest.TempPathFactory) -> SyntheticCohort:
    root = tmp_path_factory.mktemp("dataset_cohort")
    spec = SyntheticCohortSpec(
        n_subjects=20, min_visits=1, max_visits=4, volume_shape=(8, 8, 8), seed=99
    )
    return SyntheticCohort(root=root, spec=spec).generate()


@pytest.fixture(scope="module")
def mixed_length_batch(cohort: SyntheticCohort):  # type: ignore[no-untyped-def]
    """A batch deliberately containing sequences of differing length."""
    sequences = cohort.sequences()
    by_length: dict[int, VisitSequence] = {}
    for sequence in sequences:
        by_length.setdefault(len(sequence.visits), sequence)
    chosen = [by_length[n] for n in sorted(by_length)][:3]
    assert len({len(s.visits) for s in chosen}) > 1, "cohort gave no length variety"
    dataset = VisitSequenceDataset(chosen)
    return [dataset[i] for i in range(len(chosen))], chosen


class TestDatasetLoading:
    def test_loads_one_volume_per_visit_with_a_channel_axis(self, cohort: SyntheticCohort) -> None:
        sequence = next(s for s in cohort.sequences() if s.is_longitudinal)
        item = VisitSequenceDataset([sequence])[0]
        assert item.scans.shape == (len(sequence.visits), 1, 8, 8, 8)
        assert item.scans.dtype == torch.float32

    def test_length_matches_the_sequences_given(self, cohort: SyntheticCohort) -> None:
        sequences = cohort.sequences()[:5]
        assert len(VisitSequenceDataset(sequences)) == 5

    def test_the_same_spatial_transform_is_applied_across_a_subjects_visits(
        self, tmp_path: Path
    ) -> None:
        """Independent augmentation per visit injects fake anatomical change.

        That fake change is exactly the signal the temporal module reads, so this is a
        silent corruption of the thing under test rather than a loading bug.

        Built so the visits hold *identical* volumes: if the transform is shared, the
        outputs stay identical; if drawn per visit, they diverge.
        """
        import nibabel as nib
        import numpy as np

        volume = np.full((6, 6, 6), 0.5, dtype=np.float32)
        visits = []
        for day in (0, 300, 700):
            path = tmp_path / f"s_{day}.nii.gz"
            nib.Nifti1Image(volume, np.eye(4)).to_filename(path)
            visits.append(
                Visit(
                    subject_id=SubjectId("S1"),
                    session_id=SessionId(f"S1_d{day}"),
                    days_from_entry=day,
                    scan_path=path,
                    label=Label.IMPAIRED,
                )
            )
        sequence = VisitSequence(subject_id=SubjectId("S1"), visits=tuple(visits))

        def random_shift(scan: torch.Tensor) -> torch.Tensor:
            return scan + torch.randn(1).item()

        item = VisitSequenceDataset([sequence], transform=random_shift)[0]
        torch.testing.assert_close(item.scans[0], item.scans[1])
        torch.testing.assert_close(item.scans[1], item.scans[2])


class TestAugmentationRng:
    """The transform must not reach the process-global RNG, and must vary by epoch."""

    def _dataset(self, tmp_path: Path) -> VisitSequenceDataset:
        import nibabel as nib
        import numpy as np

        path = tmp_path / "v.nii.gz"
        nib.Nifti1Image(np.zeros((6, 6, 6), dtype=np.float32), np.eye(4)).to_filename(path)
        visit = Visit(
            subject_id=SubjectId("S1"),
            session_id=SessionId("S1_d0"),
            days_from_entry=0,
            scan_path=path,
            label=Label.IMPAIRED,
        )
        sequence = VisitSequence(subject_id=SubjectId("S1"), visits=(visit,))
        return VisitSequenceDataset([sequence], transform=lambda s: s + torch.randn(1).item())

    def test_loading_does_not_disturb_the_global_rng(self, tmp_path: Path) -> None:
        """Seeding torch from inside a dataset makes model init depend on fetch order.

        Nothing would report that: training still runs, still converges, and is simply
        no longer reproducible from its seed (ADR-0008).
        """
        dataset = self._dataset(tmp_path)
        torch.manual_seed(1234)
        expected = torch.randn(3)

        torch.manual_seed(1234)
        dataset[0]
        after_loading = torch.randn(3)

        torch.testing.assert_close(expected, after_loading)

    def test_the_draw_changes_between_epochs(self, tmp_path: Path) -> None:
        """Otherwise augmentation is fixed preprocessing wearing augmentation's name."""
        dataset = self._dataset(tmp_path)
        dataset.set_epoch(0)
        first = dataset[0].scans.clone()
        dataset.set_epoch(1)
        second = dataset[0].scans
        assert not torch.allclose(first, second)

    def test_the_draw_is_reproducible_within_an_epoch(self, tmp_path: Path) -> None:
        dataset = self._dataset(tmp_path)
        dataset.set_epoch(4)
        torch.testing.assert_close(dataset[0].scans, dataset[0].scans)


class TestCollationPadding:
    def test_batches_variable_lengths_to_the_longest(self, mixed_length_batch) -> None:  # type: ignore[no-untyped-def]
        items, chosen = mixed_length_batch
        batch = collate_sequences(items)
        longest = max(len(s.visits) for s in chosen)
        assert batch.scans.shape[:2] == (len(chosen), longest)
        assert batch.mask.shape == (len(chosen), longest)

    def test_the_target_visit_sits_at_the_final_position_for_every_sample(
        self,
        mixed_length_batch,  # type: ignore[no-untyped-def]
    ) -> None:
        """The reason for left-padding, asserted directly."""
        items, _ = mixed_length_batch
        batch = collate_sequences(items)
        assert batch.mask[:, -1].all(), "a sample has padding at the final position"
        for row, item in enumerate(items):
            torch.testing.assert_close(batch.scans[row, -1], item.scans[-1])

    def test_padding_sits_at_the_start_and_is_masked_out(
        self,
        mixed_length_batch,  # type: ignore[no-untyped-def]
    ) -> None:
        items, _ = mixed_length_batch
        batch = collate_sequences(items)
        for row, item in enumerate(items):
            n_real = item.scans.shape[0]
            n_pad = batch.mask.shape[1] - n_real
            assert not batch.mask[row, :n_pad].any()
            assert batch.mask[row, n_pad:].all()
            assert batch.scans[row, :n_pad].abs().sum() == 0

    def test_mask_counts_the_real_visits(self, mixed_length_batch) -> None:  # type: ignore[no-untyped-def]
        items, chosen = mixed_length_batch
        batch = collate_sequences(items)
        assert batch.mask.sum(dim=1).tolist() == [len(s.visits) for s in chosen]


class TestCollationElapsedTime:
    def test_elapsed_days_read_zero_at_the_target_and_negative_before(
        self,
        mixed_length_batch,  # type: ignore[no-untyped-def]
    ) -> None:
        items, _ = mixed_length_batch
        batch = collate_sequences(items)
        assert (batch.elapsed_days[:, -1] == 0).all()
        for row, item in enumerate(items):
            n_real = item.scans.shape[0]
            n_pad = batch.mask.shape[1] - n_real
            real = batch.elapsed_days[row, n_pad:]
            assert (real[:-1] <= 0).all(), "earlier visits must be negative or zero"

    def test_elapsed_days_match_the_sequence_exactly(
        self,
        mixed_length_batch,  # type: ignore[no-untyped-def]
    ) -> None:
        items, chosen = mixed_length_batch
        batch = collate_sequences(items)
        for row, sequence in enumerate(chosen):
            n_real = len(sequence.visits)
            n_pad = batch.mask.shape[1] - n_real
            expected = torch.tensor(sequence.elapsed_days, dtype=batch.elapsed_days.dtype)
            torch.testing.assert_close(batch.elapsed_days[row, n_pad:], expected)


class TestCollationLabelsAndIdentity:
    def test_labels_come_from_the_target_visit(self, mixed_length_batch) -> None:  # type: ignore[no-untyped-def]
        items, chosen = mixed_length_batch
        batch = collate_sequences(items)
        assert batch.labels.tolist() == [int(s.label) for s in chosen]

    def test_subject_ids_are_carried_for_per_subject_aggregation(
        self,
        mixed_length_batch,  # type: ignore[no-untyped-def]
    ) -> None:
        """Metrics aggregate per subject as well as per visit, so identity must survive."""
        items, chosen = mixed_length_batch
        batch = collate_sequences(items)
        assert batch.subject_ids == tuple(str(s.subject_id) for s in chosen)

    def test_rejects_an_empty_batch(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            collate_sequences([])


class TestSingleScanCollate:
    def test_returns_only_the_target_visit(self, mixed_length_batch) -> None:  # type: ignore[no-untyped-def]
        items, _ = mixed_length_batch
        batch = single_scan_collate(items)
        assert batch.scans.shape[1] == 1
        assert batch.mask.shape[1] == 1
        assert batch.mask.all()

    def test_the_scan_it_keeps_is_the_target_one(self, mixed_length_batch) -> None:  # type: ignore[no-untyped-def]
        items, _ = mixed_length_batch
        batch = single_scan_collate(items)
        for row, item in enumerate(items):
            torch.testing.assert_close(batch.scans[row, 0], item.scans[-1])

    def test_elapsed_time_is_zero_because_there_is_no_history(
        self,
        mixed_length_batch,  # type: ignore[no-untyped-def]
    ) -> None:
        items, _ = mixed_length_batch
        batch = single_scan_collate(items)
        assert (batch.elapsed_days == 0).all()

    def test_draws_the_same_labels_as_the_longitudinal_collate(
        self,
        mixed_length_batch,  # type: ignore[no-untyped-def]
    ) -> None:
        """Ablation 1 varies architecture alone, so both arms must see one cohort."""
        items, _ = mixed_length_batch
        assert (
            single_scan_collate(items).labels.tolist() == collate_sequences(items).labels.tolist()
        )
        assert single_scan_collate(items).subject_ids == collate_sequences(items).subject_ids


class TestAnatomyTargets:
    def _sequence_with_targets(self, tmp_path: Path, values: dict[str, float]):  # type: ignore[no-untyped-def]
        import nibabel as nib
        import numpy as np

        path = tmp_path / "v.nii.gz"
        nib.Nifti1Image(np.zeros((6, 6, 6), dtype=np.float32), np.eye(4)).to_filename(path)
        visit = Visit(
            subject_id=SubjectId("S1"),
            session_id=SessionId("S1_d0"),
            days_from_entry=0,
            scan_path=path,
            label=Label.IMPAIRED,
            anatomy_targets=values,
        )
        return VisitSequence(subject_id=SubjectId("S1"), visits=(visit,))

    def test_targets_are_none_when_not_requested(self, tmp_path: Path) -> None:
        """Test sequences need no targets (ADR-0007), so this must stay cheap."""
        sequence = self._sequence_with_targets(tmp_path, {"hippocampus_left_mm3": 3.0})
        item = VisitSequenceDataset([sequence], load_anatomy_targets=False)[0]
        batch = collate_sequences([item])
        assert batch.anatomy_targets is None
        assert batch.anatomy_mask is None

    def test_one_target_less_sequence_does_not_disable_the_batch(self, tmp_path: Path) -> None:
        """ADR-0009: not every session ships FreeSurfer output.

        A training subject without targets is excluded from the auxiliary loss alone,
        not from the batch. Dropping targets for the whole batch would make whether the
        auxiliary objective runs at all depend on how the sampler filled the batch.
        """
        sequence = self._sequence_with_targets(tmp_path, {"a_mm3": 1.0})
        with_targets = VisitSequenceDataset([sequence], load_anatomy_targets=True)[0]
        without_targets = LoadedSequence(
            sequence=with_targets.sequence,
            scans=with_targets.scans,
            anatomy_targets=None,
        )

        batch = collate_sequences([with_targets, without_targets])

        assert batch.anatomy_mask is not None
        assert batch.anatomy_mask.tolist() == [True, False]
        assert batch.anatomy_targets is not None
        assert batch.anatomy_targets[1].abs().sum() == 0, "masked row must be inert"
        assert batch.labels.shape == (2,), "both sequences still train the classifier"

    def test_targets_are_stacked_in_a_stable_key_order(self, tmp_path: Path) -> None:
        sequence = self._sequence_with_targets(tmp_path, {"b_mm3": 2.0, "a_mm3": 1.0})
        item = VisitSequenceDataset([sequence], load_anatomy_targets=True)[0]
        batch = collate_sequences([item])
        assert batch.anatomy_targets is not None
        torch.testing.assert_close(batch.anatomy_targets, torch.tensor([[1.0, 2.0]]))
        assert batch.anatomy_mask is not None and batch.anatomy_mask.all()


class TestBatchDeviceMove:
    """``SequenceBatch.to`` — small, but it is the one place that knows every field."""

    @staticmethod
    def _batch(with_anatomy: bool) -> SequenceBatch:
        return SequenceBatch(
            scans=torch.zeros(2, 1, 1, 4, 4, 4),
            elapsed_days=torch.zeros(2, 1),
            mask=torch.ones(2, 1, dtype=torch.bool),
            labels=torch.ones(2, dtype=torch.long),
            anatomy_targets=torch.zeros(2, 3) if with_anatomy else None,
            anatomy_mask=torch.ones(2, dtype=torch.bool) if with_anatomy else None,
            subject_ids=("A", "B"),
        )

    def test_every_tensor_field_comes_back_as_a_tensor(self) -> None:
        moved = self._batch(with_anatomy=True).to(torch.device("cpu"))
        for field in fields(moved):
            value = getattr(moved, field.name)
            assert value is not None
            assert isinstance(value, (torch.Tensor, tuple))

    def test_absent_anatomy_stays_absent(self) -> None:
        """None means "this sequence has no FreeSurfer output" (ADR-0009). Turning it
        into a zero tensor here would hand the auxiliary loss a plausible number."""
        moved = self._batch(with_anatomy=False).to(torch.device("cpu"))
        assert moved.anatomy_targets is None and moved.anatomy_mask is None

    def test_subject_ids_survive_unchanged(self) -> None:
        assert self._batch(with_anatomy=True).to(torch.device("cpu")).subject_ids == ("A", "B")
