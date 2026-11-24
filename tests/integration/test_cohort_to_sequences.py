"""Where the cohort builder meets the domain types — the seam neither side tests.

`build_sequences` (#14) constructs `VisitSequence`, and `VisitSequence` enforces four
invariants at construction (#7): at least one visit, no foreign subjects, chronological
order, and a labelled target. The two landed in separate PRs with **no overlapping
files**, so git merged them without complaint and neither suite exercises the other:
the cohort tests build their own `Visit` fixtures, and the type tests never call the
builder.

That is exactly the shape of a defect that survives review. This module is the join.

The invariants are enforced inside `__post_init__`, so a violation raises during
`build_sequences` rather than here — the explicit assertions below exist so a failure
names *which* property broke instead of surfacing as a constructor traceback.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from oasis_dl.data.cohort import build_sequences
from oasis_dl.data.dataset import VisitSequenceDataset, collate_sequences
from oasis_dl.data.synthetic import SyntheticCohort, SyntheticCohortSpec
from oasis_dl.types import Visit, VisitSequence

SHAPE = (8, 8, 8)


@pytest.fixture(scope="module")
def visits(tmp_path_factory: pytest.TempPathFactory) -> list[Visit]:
    """A cohort with irregular visit counts and a real share of unlabelled sessions.

    `unlabeled_fraction` is the point: an unlabelled visit may serve as context and may
    never be a target (ADR-0004), which is a rule only the builder can get wrong.
    """
    cohort = SyntheticCohort(
        root=tmp_path_factory.mktemp("cohort_seam"),
        spec=SyntheticCohortSpec(
            n_subjects=40,
            min_visits=1,
            max_visits=4,
            unlabeled_fraction=0.15,
            volume_shape=SHAPE,
            seed=20261124,
        ),
    ).generate()
    return cohort.visits()


@pytest.fixture(scope="module")
def shuffled(visits: list[Visit]) -> list[Visit]:
    """The same visits in an order no caller would produce deliberately."""
    return sorted(visits, key=lambda v: (str(v.session_id)[::-1], v.days_from_entry))


class TestEveryBuiltSequenceSatisfiesTheTypeInvariants:
    @pytest.mark.parametrize("max_length", [None, 3, 1])
    def test_the_four_construction_invariants_hold(
        self, shuffled: list[Visit], max_length: int | None
    ) -> None:
        built = build_sequences(shuffled, max_length=max_length)
        assert built, "the builder produced nothing to check"
        for sequence in built:
            assert sequence.visits, "empty sequence"
            assert all(v.subject_id == sequence.subject_id for v in sequence.visits), (
                f"{sequence.subject_id} carries another subject's visit"
            )
            days = [v.days_from_entry for v in sequence.visits]
            assert days == sorted(days), f"visits out of order: {days}"
            assert sequence.target_visit.label is not None, "unlabelled target visit"

    def test_the_target_visit_is_the_last_one(self, shuffled: list[Visit]) -> None:
        """Collation left-pads so the final position is the target for every sample.
        If the builder ever put it elsewhere, the temporal module would read the wrong
        visit while still producing a plausible number."""
        for sequence in build_sequences(shuffled, max_length=3):
            assert sequence.target_visit is sequence.visits[-1]

    def test_no_sequence_contains_a_visit_later_than_its_target(
        self, shuffled: list[Visit]
    ) -> None:
        """The future leak no partition check can see, because it happens inside one
        subject. `elapsed_days` anchors on the target, so every other visit must be
        zero or negative."""
        for sequence in build_sequences(shuffled, max_length=3):
            elapsed = sequence.elapsed_days
            assert elapsed[-1] == 0, "the target visit is not the time anchor"
            assert all(days <= 0 for days in elapsed), f"a visit follows its target: {elapsed}"


class TestTheBuilderRespectsTheLabelRule:
    def test_unlabelled_visits_never_become_targets(self, visits: list[Visit]) -> None:
        unlabelled = {v.session_id for v in visits if v.label is None}
        assert unlabelled, "fixture has no unlabelled visits, so this proves nothing"
        for sequence in build_sequences(visits, max_length=None):
            assert sequence.target_visit.session_id not in unlabelled

    def test_unlabelled_visits_may_still_serve_as_context(self, visits: list[Visit]) -> None:
        """They are usable history even when they cannot be scored (ADR-0004)."""
        unlabelled = {v.session_id for v in visits if v.label is None}
        used_as_context = {
            v.session_id
            for sequence in build_sequences(visits, max_length=None)
            for v in sequence.visits[:-1]
        }
        assert unlabelled & used_as_context, "unlabelled visits were dropped, not used"


class TestMaxLength:
    def test_it_truncates_history_not_the_number_of_sequences(self, shuffled: list[Visit]) -> None:
        """One sequence per labelled visit either way — ADR-0010's cap bounds how far
        back each one reaches, never how many subjects contribute."""
        full = build_sequences(shuffled, max_length=None)
        capped = build_sequences(shuffled, max_length=3)
        assert len(full) == len(capped)
        assert all(len(s.visits) <= 3 for s in capped)

    def test_length_one_leaves_nothing_longitudinal(self, shuffled: list[Visit]) -> None:
        assert not any(s.is_longitudinal for s in build_sequences(shuffled, max_length=1))


class TestOrderIndependence:
    def test_input_order_cannot_change_the_output(self, shuffled: list[Visit]) -> None:
        """The builder claims this and the partitioner depends on it. A cohort that
        differs by how the visit stream happened to be ordered is a different cohort."""

        def shape(sequences: list[VisitSequence]) -> list[tuple[str, tuple[str, ...]]]:
            return [
                (str(s.subject_id), tuple(str(v.session_id) for v in s.visits)) for s in sequences
            ]

        assert shape(build_sequences(shuffled, max_length=3)) == shape(
            build_sequences(list(reversed(shuffled)), max_length=3)
        )


class TestTheOutputFeedsTheTrainingPath:
    def test_built_sequences_collate_into_a_batch(self, shuffled: list[Visit]) -> None:
        """The whole point of the seam: what the cohort builder emits has to be what
        the dataset and collate consume, with no adapter in between."""
        sequences = build_sequences(shuffled, max_length=3)
        loader = DataLoader(
            VisitSequenceDataset(sequences[:6], seed=0),
            batch_size=3,
            collate_fn=collate_sequences,
        )
        batch = next(iter(loader))
        assert batch.scans.shape[0] == 3
        assert batch.scans.shape[2:] == torch.Size((1, *SHAPE))
        assert batch.mask[:, -1].all(), "the target visit must never be padding"
        assert batch.labels.shape == (3,)
