"""OASIS-3 cohort source: clinical/tabular metadata in, sessions out.

The primary :class:`~oasis_dl.data.cohort.CohortSource` implementation. Reads only the
metadata spreadsheets that ship with the OASIS-3 data-files archive — no imaging is
opened here, so the cohort can be decided before any scan is preprocessed (ADR-0010).

Sources read from ``OASIS_ROOT`` (set in the environment, never committed):

- ``OASIS3_MR_json.csv`` — one row per acquired MR scan. The ``scan category`` column
  selects T1-weighted scans; the session ``label`` (``<subject>_MR_d<NNNN>``) carries
  the subject id and the days-from-entry (``_dNNNN``), which is also the temporal
  module's elapsed-time anchor (ADR-0006).
- ``OASIS3_UDSb4_cdr.csv`` — one row per clinical assessment, with ``CDRTOT`` (the
  global CDR score) and ``days_to_visit``. The nearest assessment in time labels a
  session, subject to the ADR-0004 gap window.

**Cognitive scores never leave this adapter.** ``MMSE`` is read only to be dropped
explicitly from the frame handed downstream, so a future column addition cannot leak
it (ADR-0002). The label is a threshold of ``CDRTOT`` alone.

A session may carry several T1w scans (multiple runs). One is chosen deterministically
— the lowest run number, tie-broken by filename — so the manifest is reproducible and
the choice is auditable. The scan path points at the on-disk location of the chosen
``.nii.gz`` when it is present, and otherwise at the location #15 will write the
preprocessed volume to; the manifest records which, so a missing scan is not silently
relabelled as a preprocessing failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from oasis_dl.types import Label, SubjectId

__all__ = ["Oasis3CohortSource", "Oasis3SourceConfig"]

_MR_CSV = "MRI-json-MRI_json_information/resources/csv/files/OASIS3_MR_json.csv"
_CDR_CSV = "UDSb4-Form_B4__Global_Staging__CDR__Standard_and_Supplemental/resources/csv/files/OASIS3_UDSb4_cdr.csv"

_SESSION_DAY_RE = re.compile(r"_d(\d+)$")
_RUN_RE = re.compile(r"run-(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class Oasis3SourceConfig:
    """Where the OASIS-3 metadata lives and how T1w scans are chosen.

    Attributes:
        root: ``OASIS_ROOT``. Local, environment-specific, never committed.
        preprocessed_subdir: Directory under ``root`` where #15 writes preprocessed
            volumes. Used only to point at a future path when a raw scan is absent.
        prefer_on_disk_raw: When true, ``scan_path`` points at the raw ``.nii.gz`` if
            it is present on disk; otherwise at the future preprocessed path. The
            manifest's ``exclusion_reason`` is the source of truth for which.
    """

    root: Path
    preprocessed_subdir: str = "preprocessed"
    prefer_on_disk_raw: bool = True


def _parse_session_day(session_label: str) -> int | None:
    """Days from study entry encoded in the ``_dNNNN`` suffix of an OASIS label.

    Returns None for labels without the suffix rather than raising, so a malformed row
    is excluded by the inclusion rule rather than crashing the whole build.
    """
    match = _SESSION_DAY_RE.search(str(session_label))
    return int(match.group(1)) if match else None


def _run_number(filename: str) -> int:
    """The acquisition run index from a BIDS-ish filename, or a large sentinel.

    A large sentinel rather than ``0`` so a no-run filename sorts *after* named runs
    when we want a named run, and *before* when we do not. Here we always want the
    lowest run, so scans without a run token get ``maxsize`` and lose to any named
    run — the same outcome a BIDS reader would reach.
    """
    match = _RUN_RE.search(str(filename))
    return int(match.group(1)) if match else 10**9


class Oasis3CohortSource:
    """The OASIS-3 :class:`~oasis_dl.data.cohort.CohortSource`.

    Construct with an :class:`Oasis3SourceConfig` and call :meth:`sessions` to get
    every T1w session the dataset offers, before any inclusion rule is applied. The
    frame carries no cognitive-score column (ADR-0002).

    The CDR join is done here rather than in ``build_visits`` because it is a property
    of the dataset, not of the cohort rule: the same nearest-assessment match is what
    every gap window would query, and doing it once keeps ``build_visits`` free to
    apply the window without re-reading the clinical file.
    """

    def __init__(self, config: Oasis3SourceConfig) -> None:
        self.config = config
        self._sessions: pd.DataFrame | None = None

    # ----------------------------------------------------------------- loading

    def _mr_path(self) -> Path:
        return self.config.root / "OASIS3_data_files_full_set" / "scans" / _MR_CSV

    def _cdr_path(self) -> Path:
        return self.config.root / "OASIS3_data_files_full_set" / "scans" / _CDR_CSV

    def _load_mr(self) -> pd.DataFrame:
        frame = pd.read_csv(self._mr_path())
        # OASIS-3 ships one row per acquired scan; keep only T1-weighted ones. The
        # ``scan category`` column is the source-side modality tag, not a free-text
        # field, so an exact match is correct and a substring match would catch
        # ``T2w``/``T2star``.
        category = frame["scan category"].map(lambda value: str(value).strip())
        t1w = frame[category == "T1w"].copy()
        if t1w.empty:
            raise ValueError(
                f"no T1w rows in {self._mr_path()}; check that the MR-json CSV is the "
                f"full OASIS-3 metadata archive"
            )
        t1w["days_from_entry"] = t1w["label"].map(_parse_session_day)
        t1w = t1w[t1w["days_from_entry"].notna()].copy()
        t1w["days_from_entry"] = t1w["days_from_entry"].astype(int)
        t1w["run_number"] = t1w["filename"].map(_run_number)
        return t1w

    def _load_cdr(self) -> pd.DataFrame:
        frame = pd.read_csv(self._cdr_path())
        # CDRTOT is the global CDR (0, 0.5, 1, 2, 3). A null CDRTOT means the assessment
        # was not scored; drop it so it cannot label anything by accident.
        frame = frame[frame["CDRTOT"].notna()].copy()
        frame["cdr_days"] = frame["days_to_visit"].astype(float)
        frame["cdr_subject"] = frame["OASISID"].astype(str)
        return frame

    def _choose_scan(self, scans: pd.DataFrame) -> pd.Series:
        """Pick one T1w scan for a session, deterministically.

        Lowest run number, then filename, so the choice is reproducible and auditable.
        A session with a single scan returns it directly.
        """
        if len(scans) == 1:
            return scans.iloc[0]
        ordered = scans.sort_values(by=["run_number", "filename"], kind="mergesort")
        return ordered.iloc[0]

    def _nearest_cdr(
        self, subject: str, mr_day: int, cdr: pd.DataFrame, gap_days: int
    ) -> tuple[float | None, int | None]:
        """Nearest CDRTOT for a session within the gap window, or (None, None).

        Returns the score and the absolute gap in days, so ``build_visits`` can report
        the gap distribution without re-joining.
        """
        subj = cdr[cdr["cdr_subject"] == subject]
        if subj.empty:
            return None, None
        diffs = (subj["cdr_days"] - mr_day).abs()
        nearest_idx = diffs.idxmin()
        gap = int(diffs.loc[nearest_idx])
        if gap > gap_days:
            return None, None
        return float(cdr.loc[nearest_idx, "CDRTOT"]), gap

    # ----------------------------------------------------------- CohortSource

    def sessions(self) -> pd.DataFrame:
        """Every T1w session, with a resolvable label where one exists in-window.

        Columns: ``subject_id``, ``session_id``, ``days_from_entry``, ``scan_path``,
        ``label`` (int 0/1 or NaN), ``cdr_gap_days`` (int or NaN). No cognitive-score
        column is emitted (ADR-0002); ``MMSE`` is read from the clinical file and
        dropped here on purpose.

        The label is ``Label.COGNITIVELY_NORMAL.value`` (0) when the nearest CDRTOT is
        0, ``Label.IMPAIRED.value`` (1) when CDRTOT > 0 (0.5 counts as impaired under
        ADR-0004's ``CDR > 0`` rule), and NaN when no assessment falls in the window —
        those sessions are usable as context and never as a target visit.
        """
        if self._sessions is not None:
            return self._sessions.copy()

        mr = self._load_mr()
        cdr = self._load_cdr()
        # The gap window is read from the cohort rule at build time, but the join is
        # independent of it: we attach the nearest CDRTOT and its gap, and let
        # ``build_visits`` apply the window. A wide default here keeps every session
        # labelled with its nearest assessment so the window can be tightened later
        # without re-reading the file.
        default_gap = 10**9

        rows: list[dict[str, object]] = []
        for session_id, group in mr.groupby("label", sort=False):
            chosen = self._choose_scan(group)
            subject = str(chosen["subject_id"])
            day = int(chosen["days_from_entry"])
            cdrtot, gap = self._nearest_cdr(subject, day, cdr, default_gap)
            if cdrtot is None:
                label: float = float("nan")
            else:
                label = float(
                    Label.IMPAIRED.value if cdrtot > 0 else Label.COGNITIVELY_NORMAL.value
                )
            rows.append(
                {
                    "subject_id": subject,
                    "session_id": str(session_id),
                    "days_from_entry": day,
                    "scan_path": str(self._scan_path_for(subject, str(session_id), chosen)),
                    "label": label,
                    "cdr_gap_days": float("nan") if gap is None else int(gap),
                }
            )

        frame = pd.DataFrame(rows)
        # Stable, meaningful order: subject then time. Independent of input row order,
        # the same property the sampler and partitioner depend on.
        frame = frame.sort_values(
            by=["subject_id", "days_from_entry", "session_id"], kind="mergesort"
        ).reset_index(drop=True)
        self._sessions = frame
        return frame.copy()

    def scan_path(self, subject_id: SubjectId, session_id: str) -> Path:
        """Where the chosen T1w scan for a session lives (raw or preprocessed)."""
        frame = self.sessions()
        match = frame[
            (frame["subject_id"] == str(subject_id)) & (frame["session_id"] == str(session_id))
        ]
        if match.empty:
            raise KeyError(f"no OASIS-3 session {session_id} for {subject_id}")
        return Path(str(match.iloc[0]["scan_path"]))

    # ----------------------------------------------------------------- helpers

    def _scan_path_for(self, subject: str, session_id: str, scan: pd.Series) -> Path:
        """Resolve a scan path, preferring the raw on-disk ``.nii.gz`` when present.

        The raw volume lives under ``scans/<session>/anatN/sub-..._T1w.nii.gz``. The MR
        sidecar filename is the ``.json``; the matching ``.nii.gz`` is the same stem.
        When the raw file is absent (not yet downloaded, or this subject was not in the
        pilot), point at the preprocessed destination #15 will write, so the manifest
        records a future path rather than a missing one.
        """
        raw_dir = self.config.root / "oasis3_download" / "scans" / session_id
        stem = str(scan["filename"]).replace(".json", ".nii.gz")
        # The on-disk layout nests runs under ``anatN/`` subdirectories. Search for the
        # stem under the session directory; the search is shallow and bounded by the
        # number of anat subdirs, which is small.
        if self.config.prefer_on_disk_raw and raw_dir.exists():
            for candidate in raw_dir.rglob(Path(stem).name):
                return candidate
        preprocessed = (
            self.config.root / self.config.preprocessed_subdir / subject / f"{session_id}.nii.gz"
        )
        return preprocessed
