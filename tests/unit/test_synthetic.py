"""The synthetic cohort generator.

This cohort stands in for OASIS-3 everywhere downstream, so the properties tested here
are the ones the rest of the pipeline assumes: reproducibility from a seed, irregular
visit spacing, a controllable label balance, and sequences that cannot contain the
future.

It carries a weak learnable signal on purpose. A cohort of pure noise cannot tell a
training loop that silently fails to optimise apart from one that works, and that is
exactly the failure #8 exists to rule out.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oasis_dl.data.synthetic import SyntheticCohort, SyntheticCohortSpec
from oasis_dl.types import Label


@pytest.fixture
def spec() -> SyntheticCohortSpec:
    return SyntheticCohortSpec(n_subjects=24, seed=7)


@pytest.fixture
def cohort(tmp_path: Path, spec: SyntheticCohortSpec) -> SyntheticCohort:
    return SyntheticCohort(root=tmp_path, spec=spec).generate()


class TestReproducibility:
    def test_the_same_seed_gives_the_same_sessions(self, tmp_path: Path) -> None:
        a = SyntheticCohort(tmp_path / "a", SyntheticCohortSpec(seed=3)).generate()
        b = SyntheticCohort(tmp_path / "b", SyntheticCohortSpec(seed=3)).generate()
        left, right = a.sessions(), b.sessions()
        assert list(left["session_id"]) == list(right["session_id"])
        assert list(left["days_from_entry"]) == list(right["days_from_entry"])
        # .equals rather than ==: unlabeled sessions are NaN, and NaN != NaN.
        assert left["label"].equals(right["label"])

    def test_a_different_seed_gives_a_different_cohort(self, tmp_path: Path) -> None:
        a = SyntheticCohort(tmp_path / "a", SyntheticCohortSpec(seed=3)).generate()
        b = SyntheticCohort(tmp_path / "b", SyntheticCohortSpec(seed=4)).generate()
        assert list(a.sessions()["days_from_entry"]) != list(b.sessions()["days_from_entry"])


class TestCohortShape:
    def test_visit_counts_stay_inside_the_configured_range(self, tmp_path: Path) -> None:
        spec = SyntheticCohortSpec(n_subjects=40, min_visits=1, max_visits=4, seed=1)
        sessions = SyntheticCohort(tmp_path, spec).generate().sessions()
        counts = sessions.groupby("subject_id").size()
        assert counts.min() >= 1
        assert counts.max() <= 4

    def test_some_subjects_have_more_than_one_visit(self, cohort: SyntheticCohort) -> None:
        """Otherwise the longitudinal arm has nothing to read."""
        counts = cohort.sessions().groupby("subject_id").size()
        assert (counts > 1).any()

    def test_visit_spacing_is_irregular_not_a_grid(self, cohort: SyntheticCohort) -> None:
        sessions = cohort.sessions().sort_values(["subject_id", "days_from_entry"])
        gaps = sessions.groupby("subject_id")["days_from_entry"].diff().dropna()
        assert len(gaps.unique()) > 1, "visit spacing is a uniform grid"

    def test_label_balance_is_near_the_configured_fraction(self, tmp_path: Path) -> None:
        spec = SyntheticCohortSpec(n_subjects=300, impaired_fraction=0.25, seed=11)
        sessions = SyntheticCohort(tmp_path, spec).generate().sessions()
        labeled = sessions[sessions["label"].notna()]
        observed = (labeled["label"] == int(Label.IMPAIRED)).mean()
        assert 0.15 < observed < 0.35, f"impaired fraction was {observed:.2f}"


class TestScansOnDisk:
    def test_every_session_has_a_readable_volume_at_the_configured_shape(
        self, tmp_path: Path
    ) -> None:
        import nibabel as nib

        spec = SyntheticCohortSpec(n_subjects=4, volume_shape=(12, 14, 16), seed=2)
        cohort = SyntheticCohort(tmp_path, spec).generate()
        for row in cohort.sessions().itertuples():
            path = cohort.scan_path(row.subject_id, row.session_id)
            assert path.exists(), f"missing volume for {row.session_id}"
            assert nib.load(path).shape == (12, 14, 16)


class TestSequences:
    def test_every_labeled_visit_becomes_one_sequence(self, cohort: SyntheticCohort) -> None:
        sessions = cohort.sessions()
        n_labeled = int(sessions["label"].notna().sum())
        assert len(cohort.sequences()) == n_labeled

    def test_a_sequence_never_contains_a_visit_after_its_target(
        self, cohort: SyntheticCohort
    ) -> None:
        """The future-leak invariant, checked on real generated data."""
        for sequence in cohort.sequences():
            target_day = sequence.target_visit.days_from_entry
            assert all(v.days_from_entry <= target_day for v in sequence.visits)

    def test_max_length_truncates_from_the_oldest_end(self, cohort: SyntheticCohort) -> None:
        """Truncation drops history, never the target visit."""
        full = {s.target_visit.session_id: s for s in cohort.sequences()}
        capped = {s.target_visit.session_id: s for s in cohort.sequences(max_length=2)}
        assert full.keys() == capped.keys()
        for session_id, short in capped.items():
            assert len(short.visits) <= 2
            assert short.target_visit == full[session_id].target_visit
            assert short.visits == full[session_id].visits[-len(short.visits) :]

    def test_all_sequences_belong_to_one_subject_each(self, cohort: SyntheticCohort) -> None:
        for sequence in cohort.sequences():
            assert {v.subject_id for v in sequence.visits} == {sequence.subject_id}


class TestLearnableSignal:
    def test_impaired_scans_differ_from_normal_ones_in_the_signal_region(
        self, tmp_path: Path
    ) -> None:
        """A pipeline that cannot learn this cohort is broken, not unlucky.

        Without a signal, a training loop that never updates its weights produces the
        same ~0.5 AUC as one that works, and #8 could not tell them apart.
        """
        import nibabel as nib
        import numpy as np

        spec = SyntheticCohortSpec(n_subjects=120, seed=5)
        cohort = SyntheticCohort(tmp_path, spec).generate()
        sessions = cohort.sessions()
        labeled = sessions[sessions["label"].notna()]

        means: dict[int, list[float]] = {0: [], 1: []}
        for row in labeled.itertuples():
            volume = np.asarray(nib.load(cohort.scan_path(row.subject_id, row.session_id)).dataobj)
            d, h, w = volume.shape
            centre = volume[d // 3 : 2 * d // 3, h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
            means[int(row.label)].append(float(centre.mean()))

        assert means[0] and means[1]
        normal, impaired = np.mean(means[0]), np.mean(means[1])
        assert impaired < normal, (
            f"impaired centre mean {impaired:.4f} should sit below normal "
            f"{normal:.4f} — the atrophy proxy is absent"
        )

    def test_the_signal_can_be_switched_off(self, tmp_path: Path) -> None:
        """Pure noise stays available, for testing that a model cannot cheat."""
        import nibabel as nib
        import numpy as np

        spec = SyntheticCohortSpec(n_subjects=40, signal_strength=0.0, seed=5)
        cohort = SyntheticCohort(tmp_path, spec).generate()
        sessions = cohort.sessions()
        labeled = sessions[sessions["label"].notna()]
        means: dict[int, list[float]] = {0: [], 1: []}
        for row in labeled.itertuples():
            volume = np.asarray(nib.load(cohort.scan_path(row.subject_id, row.session_id)).dataobj)
            means[int(row.label)].append(float(volume.mean()))
        assert abs(np.mean(means[0]) - np.mean(means[1])) < 0.1


class TestProductionShapePreset:
    def test_a_named_preset_matches_the_adr_0010_configuration(self) -> None:
        """#3 asks for the real configuration on request; naming it is what makes it
        requestable rather than something each caller re-derives."""
        spec = SyntheticCohortSpec.production_shape(n_subjects=2, seed=1)
        assert spec.volume_shape == (96, 96, 96)
        assert spec.voxel_size_mm == 2.0
        assert spec.max_visits == 3

    @pytest.mark.slow
    def test_the_preset_actually_generates(self, tmp_path: Path) -> None:
        """A preset nobody runs is a comment. 96^3 volumes are slow, hence the mark."""
        spec = SyntheticCohortSpec.production_shape(n_subjects=2, seed=1)
        cohort = SyntheticCohort(tmp_path, spec).generate()
        import nibabel as nib

        row = next(cohort.sessions().itertuples())
        assert nib.load(cohort.scan_path(row.subject_id, row.session_id)).shape == (
            96,
            96,
            96,
        )


class TestCohortSourceProtocol:
    def test_it_satisfies_the_cohort_source_protocol(self, cohort: SyntheticCohort) -> None:
        """The point of the generator: downstream code cannot tell it from OASIS-3."""
        from oasis_dl.data.cohort import CohortSource

        source: CohortSource = cohort
        assert not source.sessions().empty

    def test_sessions_carries_the_columns_downstream_code_reads(
        self, cohort: SyntheticCohort
    ) -> None:
        required = {"subject_id", "session_id", "days_from_entry", "label"}
        assert required <= set(cohort.sessions().columns)

    def test_it_exposes_no_cognitive_score_column(self, cohort: SyntheticCohort) -> None:
        """ADR-0002: a source that hands MMSE downstream invites the circularity."""
        columns = {c.lower() for c in cohort.sessions().columns}
        assert not {"mmse", "cdr"} & columns

    def test_no_column_in_the_frame_predicts_the_label(self, cohort: SyntheticCohort) -> None:
        """The property check, where the one above is only a spelling check.

        A name-based test catches leaks someone already thought to name. This one
        catches the generator handing out its own ``atrophy`` latent — the quantity
        the label is a threshold of, i.e. this cohort's CDR — which the mmse/cdr
        check sails straight past.
        """
        import pandas as pd
        from sklearn.metrics import roc_auc_score

        labeled = cohort.sessions()[lambda f: f["label"].notna()]
        identifiers = {"label", "subject_id", "session_id", "scan_path"}

        for column in labeled.columns:
            if column in identifiers:
                continue
            values = pd.to_numeric(labeled[column], errors="coerce")
            if values.isna().all():
                continue
            auc = roc_auc_score(labeled["label"], values)
            assert 0.25 < auc < 0.75, (
                f"column {column!r} separates the label at AUC {auc:.2f}. Nothing a "
                f"model can read may be a near-deterministic function of the label "
                f"(ADR-0002)."
            )

    def test_the_latent_stays_reachable_for_debugging_the_generator(
        self, cohort: SyntheticCohort
    ) -> None:
        """Off the frame a model is handed, not off the object."""
        latent = cohort.latent_atrophy()
        assert "atrophy" in latent.columns
        assert len(latent) == len(cohort.sessions())


class TestAnatomyTargets:
    """Training targets for the auxiliary objective (ADR-0007), never inference features.

    Correlating them with the label is the *point* here, not a leak — which is exactly
    why they must stay out of :meth:`sessions`, the frame a model is allowed to read.
    """

    def test_visits_carry_named_structural_volumes(self, synthetic_cohort: SyntheticCohort) -> None:
        carried = [v.anatomy_targets for v in synthetic_cohort.visits() if v.anatomy_targets]
        assert carried
        assert set(carried[0]) == set(synthetic_cohort.spec.anatomy_target_names)

    def test_the_volumes_are_in_a_plausible_range(self, synthetic_cohort: SyntheticCohort) -> None:
        """Realistic mm^3, so a pipeline that forgets to standardize them produces an
        auxiliary term large enough to be noticed rather than one that quietly works."""
        for visit in synthetic_cohort.visits():
            if not visit.anatomy_targets:
                continue
            assert 1000 < visit.anatomy_targets["hippocampus_left_mm3"] < 6000
            assert 300_000 < visit.anatomy_targets["total_gray_matter_mm3"] < 900_000

    def test_some_visits_have_none(self, synthetic_cohort: SyntheticCohort) -> None:
        """ADR-0009: not every OASIS-3 session ships FreeSurfer output. A cohort where
        every visit has targets cannot exercise the per-sequence mask that requires."""
        present = [bool(v.anatomy_targets) for v in synthetic_cohort.visits()]
        assert any(present) and not all(present)

    def test_targets_never_appear_in_the_readable_session_frame(
        self, synthetic_cohort: SyntheticCohort
    ) -> None:
        """The leak this separation prevents. These correlate with the label by design,
        so a model that could read them would be reading the answer."""
        columns = set(synthetic_cohort.sessions().columns)
        for name in synthetic_cohort.spec.anatomy_target_names:
            assert name not in columns
        assert not any("hippocamp" in c or "gray_matter" in c for c in columns)

    def test_they_shrink_as_the_latent_atrophy_grows(
        self, synthetic_cohort: SyntheticCohort
    ) -> None:
        """Otherwise the auxiliary objective would be regression onto noise, and
        ablation 2 would measure nothing either way."""
        latent = synthetic_cohort.latent_atrophy().set_index("session_id")["atrophy"]
        pairs = [
            (float(latent[str(v.session_id)]), v.anatomy_targets["total_gray_matter_mm3"])
            for v in synthetic_cohort.visits()
            if v.anatomy_targets
        ]
        atrophy = np.array([p[0] for p in pairs])
        volume = np.array([p[1] for p in pairs])
        assert np.corrcoef(atrophy, volume)[0, 1] < -0.5

    def test_an_empty_target_set_produces_a_cohort_with_none(self, tmp_path: Path) -> None:
        """What arm PA sees: no auxiliary targets anywhere."""
        cohort = SyntheticCohort(
            root=tmp_path,
            spec=SyntheticCohortSpec(
                n_subjects=4, volume_shape=(8, 8, 8), anatomy_target_names=(), seed=0
            ),
        ).generate()
        assert all(v.anatomy_targets is None for v in cohort.visits())

    def test_duplicate_target_names_are_refused(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            SyntheticCohortSpec(anatomy_target_names=("a", "b", "a"))

    def test_the_same_seed_reproduces_the_targets(self, tmp_path: Path) -> None:
        def build(where: str) -> list[dict[str, float] | None]:
            cohort = SyntheticCohort(
                root=tmp_path / where,
                spec=SyntheticCohortSpec(n_subjects=6, volume_shape=(8, 8, 8), seed=3),
            ).generate()
            return [v.anatomy_targets for v in cohort.visits()]

        assert build("a") == build("b")
