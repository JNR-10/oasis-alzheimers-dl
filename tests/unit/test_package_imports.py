"""Every module in the package imports cleanly, and the builders produce valid records.

Cheap, and it earns its place twice.

An unimplemented seam still has to *import*. A circular import, a typo in a type
annotation evaluated at module scope, or a dependency that moved all fail here rather
than three tickets later inside someone's half-written training loop.

It also means ``pytest`` collects something. With no tests at all, pytest exits 5 and
CI reads that as failure, so a contributor running ``make check`` on a clean branch
sees red for reasons that have nothing to do with their work.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest
from tests.helpers import make_sequence, make_sequences, make_subject_labels, make_visit

import oasis_dl
from oasis_dl.types import Label, Partition


def _module_names() -> list[str]:
    """Every importable module under ``oasis_dl``, found rather than listed.

    Walking the package means a new module is covered the moment it is written, which
    is the only version of this test that stays true.
    """
    return [
        name
        for _, name, _ in pkgutil.walk_packages(oasis_dl.__path__, prefix=f"{oasis_dl.__name__}.")
    ]


class TestPackageImports:
    def test_the_package_exposes_modules_to_walk(self) -> None:
        assert _module_names(), "found no modules — the package layout moved"

    @pytest.mark.parametrize("module_name", _module_names())
    def test_module_imports(self, module_name: str) -> None:
        """A seam may be unimplemented. It may not be unimportable."""
        importlib.import_module(module_name)


class TestDomainEnums:
    def test_partitions_are_the_three_the_protocol_names(self) -> None:
        assert {p.value for p in Partition} == {"train", "val", "test"}

    def test_the_label_is_binary_and_impaired_is_the_positive_class(self) -> None:
        """Metrics, thresholds and class weighting all assume this orientation."""
        assert int(Label.IMPAIRED) == 1
        assert int(Label.COGNITIVELY_NORMAL) == 0


class TestHelpers:
    """The builders other tickets will lean on, checked so a failure points here."""

    def test_a_visit_knows_whether_it_can_be_a_target(self) -> None:
        assert make_visit("S1", 0, Label.IMPAIRED).label is Label.IMPAIRED
        assert make_visit("S1", 0, None).label is None

    def test_a_sequence_ends_at_its_labeled_target_visit(self) -> None:
        sequence = make_sequence("S1", (0, 400, 900), Label.IMPAIRED)
        assert sequence.visits[-1].days_from_entry == 900
        assert sequence.visits[-1].label is Label.IMPAIRED
        assert all(v.label is None for v in sequence.visits[:-1])

    def test_a_sequence_refuses_days_out_of_order(self) -> None:
        with pytest.raises(ValueError, match="ascending"):
            make_sequence("S1", (900, 400))

    def test_subject_labels_hit_the_requested_balance(self) -> None:
        labels = make_subject_labels(n_subjects=100, impaired_fraction=0.25, seed=1)
        impaired = sum(1 for v in labels.values() if v is Label.IMPAIRED)
        assert impaired == 25

    def test_subject_labels_are_reproducible_from_the_seed(self) -> None:
        assert make_subject_labels(50, seed=7) == make_subject_labels(50, seed=7)

    def test_subject_labels_are_not_in_sorted_order(self) -> None:
        """Deliberate: a partitioner that depends on dict order must fail, not pass.

        Sorted input hides an order dependency, because sorted input is what a
        correctly-written partitioner produces internally anyway.
        """
        ids = list(make_subject_labels(50, seed=3))
        assert ids != sorted(ids)

    def test_sequences_cover_every_subject_exactly_once(self) -> None:
        sequences = make_sequences(n_subjects=20, seed=2)
        subjects = [s.subject_id for s in sequences]
        assert len(subjects) == len(set(subjects)) == 20

    def test_sequence_spacing_is_irregular(self) -> None:
        """A uniform grid lets code that ignores elapsed time look correct."""
        gaps = {
            b.days_from_entry - a.days_from_entry
            for sequence in make_sequences(n_subjects=30, seed=4)
            for a, b in zip(sequence.visits, sequence.visits[1:], strict=False)
        }
        assert len(gaps) > 1, "every visit gap is identical"

    def test_sequences_are_reproducible_from_the_seed(self) -> None:
        left = make_sequences(n_subjects=10, seed=5)
        right = make_sequences(n_subjects=10, seed=5)
        assert [s.subject_id for s in left] == [s.subject_id for s in right]
        assert [len(s.visits) for s in left] == [len(s.visits) for s in right]

    def test_the_scan_path_does_not_exist(self) -> None:
        """These records are for record-level tests. Opening one should fail loudly."""
        assert not make_visit("S1", 0).scan_path.exists()
