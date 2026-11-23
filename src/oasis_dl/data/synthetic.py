"""A synthetic cohort shaped like OASIS-3, for building the pipeline before data.

Implements :class:`~oasis_dl.data.cohort.CohortSource`, so every module downstream
consumes it exactly as it will consume the real adapter. That is the point: when the
real cohort is frozen (#14) the work is swapping a source, not starting a pipeline.

**The volumes carry a weak learnable signal, not pure noise.** A cohort of noise cannot
distinguish a training loop that silently fails to optimise from one that works — both
land at chance. The signal is a crude atrophy proxy: impaired subjects lose intensity
in a central region, and the loss deepens with elapsed time, so a longitudinal model
has a trajectory to read and a single-scan model has only a level. Set
``signal_strength=0`` to get the noise cohort back, which is how you check a model
cannot cheat.

Nothing here claims to resemble a brain. It resembles the *shape* of the problem:
irregular visit counts, irregular spacing, unlabeled sessions, and a minority class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from oasis_dl.types import Label, SessionId, SubjectId, Visit, VisitSequence

__all__ = ["SyntheticCohort", "SyntheticCohortSpec"]

_IMPAIRMENT_THRESHOLD = 0.5
"""Atrophy level above which a visit is labeled impaired.

Arbitrary, and deliberately placed so that a few subjects cross it mid-sequence. Those
converters are the subjects the longitudinal arm exists to get right.
"""


@dataclass(frozen=True)
class SyntheticCohortSpec:
    """Parameters of a generated cohort.

    Attributes:
        n_subjects: How many subjects to generate.
        min_visits: Fewest visits a subject may have. 1 keeps single-visit subjects in
            the cohort, because the real one will have them.
        max_visits: Most visits a subject may have.
        impaired_fraction: Share of subjects seeded as impaired. The observed
            *visit-level* share lands slightly higher, because a few subjects convert.
            Defaults near the predecessor's 76/24 split.
        unlabeled_fraction: Share of sessions with no resolvable label, standing in for
            imaging sessions with no clinical assessment inside the admissible gap
            (ADR-0004). These are usable as context and never as a target visit.
        volume_shape: Voxel dimensions. Tiny by default so a full training run finishes
            in seconds on CPU.
        voxel_size_mm: Written into the NIfTI affine. 2 mm to match ADR-0010, so a
            pipeline that reads spacing from the header sees a realistic value.
        min_gap_days: Shortest gap between consecutive visits.
        max_gap_days: Longest gap between consecutive visits. Irregular spacing is the
            point — a uniform grid would let the temporal module ignore elapsed time
            and still look correct (ADR-0006).
        anatomy_target_names: Structural quantities attached to each visit as
            **training targets** for the auxiliary objective (ADR-0007). An empty tuple
            produces a cohort with no targets at all, which is what arm PA sees.
        anatomy_missing_fraction: Share of visits with no targets. Not every OASIS-3
            session ships FreeSurfer output (ADR-0009), and a cohort where every visit
            has them cannot exercise the per-sequence mask that fact requires.
        signal_strength: Depth of the atrophy proxy. 0 produces pure noise.
        noise_sigma: Standard deviation of the background noise.
        seed: Everything below is a deterministic function of this.
    """

    n_subjects: int = 40
    min_visits: int = 1
    max_visits: int = 4
    impaired_fraction: float = 0.24
    unlabeled_fraction: float = 0.08
    volume_shape: tuple[int, int, int] = (16, 16, 16)
    voxel_size_mm: float = 2.0
    min_gap_days: int = 150
    max_gap_days: int = 900
    anatomy_target_names: tuple[str, ...] = (
        "hippocampus_left_mm3",
        "hippocampus_right_mm3",
        "total_gray_matter_mm3",
    )
    anatomy_missing_fraction: float = 0.15
    signal_strength: float = 0.6
    noise_sigma: float = 0.1
    seed: int = 0

    @classmethod
    def production_shape(cls, **overrides: object) -> SyntheticCohortSpec:
        """The ADR-0010 configuration: 2 mm, 96³, at most 3 visits per sequence.

        For measuring throughput and memory on the data path before any real scan is
        preprocessed. The defaults elsewhere are deliberately tiny so the suite stays
        fast; this is the one that answers "will the real thing fit".
        """
        settings: dict[str, object] = {
            "volume_shape": (96, 96, 96),
            "voxel_size_mm": 2.0,
            "max_visits": 3,
        }
        settings.update(overrides)
        return cls(**settings)  # type: ignore[arg-type]

    def __post_init__(self) -> None:
        if self.min_visits < 1:
            raise ValueError("min_visits must be at least 1")
        if self.max_visits < self.min_visits:
            raise ValueError("max_visits must be >= min_visits")
        if not 0.0 <= self.impaired_fraction <= 1.0:
            raise ValueError("impaired_fraction must be a proportion")
        if not 0.0 <= self.unlabeled_fraction <= 1.0:
            raise ValueError("unlabeled_fraction must be a proportion")
        if self.min_gap_days < 1 or self.max_gap_days < self.min_gap_days:
            raise ValueError("gap days must be positive with max >= min")
        if not 0.0 <= self.anatomy_missing_fraction <= 1.0:
            raise ValueError("anatomy_missing_fraction must be a proportion")
        if len(set(self.anatomy_target_names)) != len(self.anatomy_target_names):
            raise ValueError("anatomy_target_names must be unique")


@dataclass
class _SubjectPlan:
    """The latent trajectory a subject's visits are drawn from."""

    subject_id: SubjectId
    baseline_atrophy: float
    atrophy_per_year: float
    days: list[int] = field(default_factory=list)


class SyntheticCohort:
    """A generated cohort on disk, usable anywhere a ``CohortSource`` is.

    Args:
        root: Directory for the generated volumes. Created if absent.
        spec: What to generate. Defaults to a small cohort.
    """

    def __init__(self, root: Path, spec: SyntheticCohortSpec | None = None) -> None:
        self.root = Path(root)
        self.spec = spec if spec is not None else SyntheticCohortSpec()
        self._sessions: pd.DataFrame | None = None
        self._atrophy: list[float] = []
        self._anatomy: list[dict[str, float] | None] = []

    # ---------------------------------------------------------------- generation

    def generate(self) -> SyntheticCohort:
        """Draw the cohort and write its volumes. Returns self, so calls chain."""
        rng = np.random.default_rng(self.spec.seed)
        self.root.mkdir(parents=True, exist_ok=True)

        plans = self._plan_subjects(rng)
        rows: list[dict[str, object]] = []
        atrophies: list[float] = []
        anatomy: list[dict[str, float] | None] = []
        for plan in plans:
            for day in plan.days:
                atrophy = plan.baseline_atrophy + plan.atrophy_per_year * (day / 365.0)
                session_id = SessionId(f"{plan.subject_id}_d{day:04d}")
                labeled = rng.random() >= self.spec.unlabeled_fraction
                label = (
                    (
                        Label.IMPAIRED
                        if atrophy > _IMPAIRMENT_THRESHOLD
                        else Label.COGNITIVELY_NORMAL
                    )
                    if labeled
                    else None
                )
                path = self._write_scan(plan.subject_id, session_id, atrophy, rng)
                atrophies.append(float(atrophy))
                anatomy.append(self._draw_anatomy(float(atrophy), rng))
                rows.append(
                    {
                        "subject_id": str(plan.subject_id),
                        "session_id": str(session_id),
                        "days_from_entry": day,
                        "scan_path": str(path),
                        "label": None if label is None else int(label),
                    }
                )

        self._sessions = pd.DataFrame(rows)
        self._atrophy = atrophies
        self._anatomy = anatomy
        return self

    def _draw_anatomy(self, atrophy: float, rng: np.random.Generator) -> dict[str, float] | None:
        """Structural volumes for one session, shrinking with the latent atrophy.

        Training targets, never inference features (ADR-0007), so correlating them with
        the label is the point rather than a leak — they are deliberately absent from
        :meth:`sessions`, which is the frame a model may read.

        Returns None for a share of sessions, standing in for the OASIS-3 sessions with
        no FreeSurfer output (ADR-0009).
        """
        if not self.spec.anatomy_target_names:
            return None
        if rng.random() < self.spec.anatomy_missing_fraction:
            return None
        # Rough adult volumes in mm^3, so a pipeline that forgets to standardize them
        # produces an auxiliary term large enough to notice.
        bases = {"hippocampus": 3800.0, "total_gray_matter": 620000.0}
        drawn: dict[str, float] = {}
        for name in self.spec.anatomy_target_names:
            base = next((value for key, value in bases.items() if name.startswith(key)), 5000.0)
            shrink = 1.0 - 0.25 * atrophy
            drawn[name] = float(base * shrink * (1.0 + 0.02 * rng.standard_normal()))
        return drawn

    def _plan_subjects(self, rng: np.random.Generator) -> list[_SubjectPlan]:
        """Draw each subject's trajectory and visit days.

        Impaired subjects start already atrophied; normal subjects start low but may
        progress fast enough to cross the threshold mid-sequence. That overlap is
        deliberate — a cohort where the two groups never meet is separable by a
        threshold and teaches the pipeline nothing about the hard cases.
        """
        plans: list[_SubjectPlan] = []
        for index in range(self.spec.n_subjects):
            subject_id = SubjectId(f"SYN{index:04d}")
            if rng.random() < self.spec.impaired_fraction:
                baseline = float(rng.uniform(0.55, 0.95))
                rate = float(rng.uniform(0.02, 0.08))
            else:
                baseline = float(rng.uniform(0.05, 0.45))
                rate = float(rng.uniform(0.00, 0.06))

            n_visits = int(rng.integers(self.spec.min_visits, self.spec.max_visits + 1))
            day = int(rng.integers(0, 120))
            days = [day]
            for _ in range(n_visits - 1):
                day += int(rng.integers(self.spec.min_gap_days, self.spec.max_gap_days + 1))
                days.append(day)

            plans.append(_SubjectPlan(subject_id, baseline, rate, days))
        return plans

    def _write_scan(
        self,
        subject_id: SubjectId,
        session_id: SessionId,
        atrophy: float,
        rng: np.random.Generator,
    ) -> Path:
        """Write one scan: noise, with a dimmed central region scaled by atrophy."""
        shape = self.spec.volume_shape
        volume = rng.normal(1.0, self.spec.noise_sigma, size=shape)

        if self.spec.signal_strength > 0.0:
            slices = tuple(slice(dimension // 3, 2 * dimension // 3) for dimension in shape)
            volume[slices] -= self.spec.signal_strength * atrophy

        directory = self.root / str(subject_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{session_id}.nii.gz"
        affine = np.diag(
            [self.spec.voxel_size_mm, self.spec.voxel_size_mm, self.spec.voxel_size_mm, 1.0]
        )
        nib.Nifti1Image(volume.astype(np.float32), affine).to_filename(path)
        return path

    # ------------------------------------------------------------- CohortSource

    def sessions(self) -> pd.DataFrame:
        """Every generated session.

        Carries no cognitive-score column by construction. A source that hands MMSE
        downstream alongside the imaging columns is one careless feature list away from
        the circularity this project exists to avoid (ADR-0002).

        The generative ``atrophy`` latent is deliberately **not** here either. The label
        is a threshold of it, so a column holding it is this cohort's CDR: any model
        reading the frame would score perfectly and the pipeline would look validated.
        ADR-0002 covers "any feature that is a near-deterministic function" of the
        label, and a synthetic leak teaches exactly the wrong lesson about a repo built
        to catch real ones. Read it from :meth:`latent_atrophy` when debugging the
        generator — never from the frame a model is handed.
        """
        if self._sessions is None:
            raise RuntimeError("call generate() before reading sessions()")
        return self._sessions.copy()

    def latent_atrophy(self) -> pd.DataFrame:
        """The generative latent behind each session's label. **Debugging only.**

        Separate from :meth:`sessions` so that handing a model the cohort cannot
        hand it the answer. Recomputed rather than stored, so there is no column to
        leak back into the frame by accident.
        """
        if self._sessions is None:
            raise RuntimeError("call generate() before reading latent_atrophy()")
        frame = self._sessions[["subject_id", "session_id"]].copy()
        frame["atrophy"] = self._atrophy
        return frame

    def scan_path(self, subject_id: SubjectId, session_id: str) -> Path:
        """Where a session's volume was written.

        Raises:
            KeyError: If the session was never generated.
        """
        frame = self.sessions()
        match = frame[
            (frame["subject_id"] == str(subject_id)) & (frame["session_id"] == str(session_id))
        ]
        if match.empty:
            raise KeyError(f"no generated session {session_id} for {subject_id}")
        return Path(str(match.iloc[0]["scan_path"]))

    # ------------------------------------------------------------------- visits

    def visits(self) -> list[Visit]:
        """Every session as a :class:`~oasis_dl.types.Visit`, chronological per subject."""
        unsorted = self.sessions()
        # Keyed by session before sorting, so reordering cannot shuffle a subject's
        # targets onto someone else's visit. `strict=True` because a length mismatch
        # here would silently pair every target with the wrong session.
        by_session = dict(zip(unsorted["session_id"], self._anatomy, strict=True))
        frame = unsorted.sort_values(["subject_id", "days_from_entry"])
        out: list[Visit] = []
        for row in frame.itertuples():
            raw_label = row.label
            out.append(
                Visit(
                    subject_id=SubjectId(str(row.subject_id)),
                    session_id=SessionId(str(row.session_id)),
                    days_from_entry=int(row.days_from_entry),
                    scan_path=Path(str(row.scan_path)),
                    label=None if pd.isna(raw_label) else Label(int(raw_label)),
                    anatomy_targets=by_session[str(row.session_id)],
                )
            )
        return out

    def sequences(self, max_length: int | None = None) -> list[VisitSequence]:
        """One sequence per labeled visit: that visit plus the subject's earlier ones.

        Later visits are excluded at construction rather than masked afterwards — a
        sequence that cannot physically contain the future cannot leak it.

        Args:
            max_length: Keep at most this many visits, counting back from the target.
                Truncation drops the oldest history and never the target visit.

        Returns:
            One sequence per labeled visit, so a subject with three labeled visits
            contributes three correlated sequences. That correlation is handled by
            sampling or loss weighting, never by splitting the subject across
            partitions (ADR-0001).
        """
        if max_length is not None and max_length < 1:
            raise ValueError("max_length must be at least 1")

        by_subject: dict[SubjectId, list[Visit]] = {}
        for visit in self.visits():
            by_subject.setdefault(visit.subject_id, []).append(visit)

        out: list[VisitSequence] = []
        for subject_id, visits in by_subject.items():
            for index, visit in enumerate(visits):
                if visit.label is None:
                    continue
                history = visits[: index + 1]
                if max_length is not None:
                    history = history[-max_length:]
                out.append(VisitSequence(subject_id=subject_id, visits=tuple(history)))
        return out
