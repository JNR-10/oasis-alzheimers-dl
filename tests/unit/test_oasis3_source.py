"""Unit tests for the OASIS-3 CohortSource and the ADR-0004 label rule.

Fixtures are tiny synthetic CSVs in the OASIS-3 schema, written to a tmp dir that
stands in for ``OASIS_ROOT``. No real OASIS subject identifiers appear in the repo;
the fixtures use ``OAS3XXXX``-shaped ids with invented visit days so the parsing paths
are exercised without reading real data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from oasis_dl.data.cohort import InclusionRules, build_sequences, build_visits
from oasis_dl.data.oasis3 import Oasis3CohortSource, Oasis3SourceConfig
from oasis_dl.types import Label, SubjectId

_MR_CSV_REL = (
    "OASIS3_data_files_full_set/scans/"
    "MRI-json-MRI_json_information/resources/csv/files/OASIS3_MR_json.csv"
)
_CDR_CSV_REL = (
    "OASIS3_data_files_full_set/scans/"
    "UDSb4-Form_B4__Global_Staging__CDR__Standard_and_Supplemental/resources/csv/files/"
    "OASIS3_UDSb4_cdr.csv"
)


def _write_mr(root: Path, rows: list[dict[str, object]]) -> None:
    """Write a minimal MR-json CSV with the columns the adapter reads."""
    path = root / _MR_CSV_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    # The adapter reads subject_id, label (session), scan category, filename.
    frame.to_csv(path, index=False)


def _write_cdr(root: Path, rows: list[dict[str, object]]) -> None:
    """Write a minimal UDSb4 CDR CSV with the columns the adapter reads."""
    path = root / _CDR_CSV_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    # The adapter reads OASISID, days_to_visit, CDRTOT.
    frame.to_csv(path, index=False)


def _mr_row(subject: str, day: int, run: int = 1, category: str = "T1w") -> dict[str, object]:
    session = f"{subject}_MR_d{day:04d}"
    filename = f"sub-{subject}_ses-d{day:04d}_run-{run:02d}_T1w.json"
    return {
        "subject_id": subject,
        "label": session,
        "scan category": category,
        "filename": filename,
    }


def _cdr_row(subject: str, day: int, cdrtot: float) -> dict[str, object]:
    return {
        "OASISID": subject,
        "OASIS_session_label": f"{subject}_UDSb4_d{day:04d}",
        "days_to_visit": day,
        "CDRTOT": cdrtot,
    }


@pytest.fixture()
def source(tmp_path: Path) -> Oasis3CohortSource:
    """A source with three subjects, varied gaps, and a multi-scan session."""
    mr_rows = [
        # Subject A: three T1w sessions at days 10, 100, 400.
        _mr_row("OAS310001", 10),
        _mr_row("OAS310001", 100),
        _mr_row("OAS310001", 400),
        # Subject A day 100 has two runs; the adapter must pick run-01.
        _mr_row("OAS310001", 100, run=2),
        # Subject B: one T1w session far from any assessment.
        _mr_row("OAS310002", 50),
        # Subject C: one T1w session at day 200 with a same-day CDR of 0.5.
        _mr_row("OAS310003", 200),
        # A non-T1w row that the adapter must drop.
        _mr_row("OAS310004", 5, category="T2w"),
    ]
    cdr_rows = [
        # Subject A: assessment at day 12 (gap 2 from MR d10), day 250 (gap 150 from
        # MR d100, gap 150 from MR d400), and day 405 (gap 5 from MR d400).
        _cdr_row("OAS310001", 12, 0.0),
        _cdr_row("OAS310001", 250, 0.5),  # 0.5 > 0 → impaired
        _cdr_row("OAS310001", 405, 1.0),
        # Subject B: assessment at day 600, gap 550 from MR d50 — outside 180d.
        _cdr_row("OAS310002", 600, 0.0),
        # Subject C: assessment at day 200 (gap 0 from MR d200), CDRTOT 0.5.
        _cdr_row("OAS310003", 200, 0.5),
    ]
    _write_mr(tmp_path, mr_rows)
    _write_cdr(tmp_path, cdr_rows)
    return Oasis3CohortSource(Oasis3SourceConfig(root=tmp_path, prefer_on_disk_raw=False))


class TestSessions:
    def test_emits_one_row_per_session_not_per_scan(self, source: Oasis3CohortSource) -> None:
        """The multi-scan session (d100) must collapse to one row."""
        sessions = source.sessions()
        d100 = sessions[sessions["session_id"] == "OAS310001_MR_d0100"]
        assert len(d100) == 1, "multi-scan session was not collapsed"

    def test_picks_the_lowest_run_for_a_multi_scan_session(
        self, source: Oasis3CohortSource
    ) -> None:
        """The chooser returns the lowest run number, tie-broken by filename."""
        mr = source._load_mr()
        d100 = mr[mr["label"] == "OAS310001_MR_d0100"]
        chosen = source._choose_scan(d100)
        assert "run-01" in str(chosen["filename"]), f"chose {chosen['filename']}"

    def test_drops_non_t1w_scans(self, source: Oasis3CohortSource) -> None:
        sessions = source.sessions()
        assert "OAS310004" not in set(sessions["subject_id"])

    def test_exposes_no_cognitive_score_column(self, source: Oasis3CohortSource) -> None:
        """ADR-0002: MMSE/CDR must not travel downstream as a readable column."""
        columns = {c.lower() for c in source.sessions().columns}
        assert not {"mmse", "cdr", "cdrsum", "cdrtot"} & columns

    def test_days_from_entry_is_parsed_from_the_session_label(
        self, source: Oasis3CohortSource
    ) -> None:
        sessions = source.sessions()
        by_session = sessions.set_index("session_id")["days_from_entry"].to_dict()
        assert by_session["OAS310001_MR_d0010"] == 10
        assert by_session["OAS310001_MR_d0400"] == 400

    def test_label_is_set_from_nearest_cdrtot(self, source: Oasis3CohortSource) -> None:
        """The source attaches the nearest CDR's label before the window is applied."""
        sessions = source.sessions()
        by_session = sessions.set_index("session_id")["label"].to_dict()
        # d10 nearest CDR is d12 (gap 2), CDRTOT 0 → normal (0).
        assert by_session["OAS310001_MR_d0010"] == Label.COGNITIVELY_NORMAL.value
        # d400 nearest CDR is d405 (gap 5), CDRTOT 1 → impaired (1).
        assert by_session["OAS310001_MR_d0400"] == Label.IMPAIRED.value

    def test_cdr_gap_days_is_attached(self, source: Oasis3CohortSource) -> None:
        sessions = source.sessions()
        by_session = sessions.set_index("session_id")["cdr_gap_days"].to_dict()
        assert by_session["OAS310001_MR_d0010"] == 2
        assert by_session["OAS310001_MR_d0400"] == 5


class TestBuildVisitsAppliesADR0004:
    def test_sessions_outside_the_gap_window_become_context_visits(
        self, source: Oasis3CohortSource
    ) -> None:
        """ADR-0004: a session whose nearest CDR is outside 180 days is not relabeled
        as normal; it stays as a context visit with no label."""
        rules = InclusionRules(max_assessment_gap_days=180)
        visits, audit = build_visits(source, rules)
        by_session = {str(v.session_id): v for v in visits}
        # Subject B d50: nearest CDR is d600, gap 550 > 180 → unlabeled context.
        assert by_session["OAS310002_MR_d0050"].label is None
        # It is still in the cohort (not excluded).
        assert (
            audit.loc[audit["session_id"] == "OAS310002_MR_d0050", "exclusion_reason"].iloc[0] == ""
        )

    def test_sessions_inside_the_gap_window_keep_their_label(
        self, source: Oasis3CohortSource
    ) -> None:
        rules = InclusionRules(max_assessment_gap_days=180)
        visits, _ = build_visits(source, rules)
        by_session = {str(v.session_id): v for v in visits}
        assert by_session["OAS310001_MR_d0010"].label is Label.COGNITIVELY_NORMAL
        assert by_session["OAS310001_MR_d0400"].label is Label.IMPAIRED

    def test_cdr_05_counts_as_impaired(self, source: Oasis3CohortSource) -> None:
        """ADR-0004: CDRTOT > 0 is impaired, so 0.5 (questionable) is impaired."""
        rules = InclusionRules(max_assessment_gap_days=180)
        visits, _ = build_visits(source, rules)
        # Subject C d200: nearest CDR is d200 (gap 0), CDRTOT 0.5 → impaired.
        by_session = {str(v.session_id): v for v in visits}
        assert by_session["OAS310003_MR_d0200"].label is Label.IMPAIRED

    def test_audit_records_no_exclusion_for_in_window_sessions(
        self, source: Oasis3CohortSource
    ) -> None:
        rules = InclusionRules(max_assessment_gap_days=180)
        _, audit = build_visits(source, rules)
        # Every emitted session has an empty exclusion_reason (no T1w failures here).
        assert (audit["exclusion_reason"] == "").all()

    def test_context_visits_serve_as_history_in_sequences(self, source: Oasis3CohortSource) -> None:
        """The unlabeled d50 visit of subject B must be usable as context, never a
        target — build_sequences enforces the target carries a label."""
        rules = InclusionRules(max_assessment_gap_days=180)
        visits, _ = build_visits(source, rules)
        sequences = build_sequences(visits, max_length=3)
        # Subject B has only one visit and it is unlabeled → no sequence.
        b_sessions = [s for s in sequences if str(s.subject_id) == "OAS310002"]
        assert b_sessions == [], "an unlabeled-only subject produced a target sequence"


class TestSourceProtocol:
    def test_satisfies_the_cohort_source_protocol(self, source: Oasis3CohortSource) -> None:
        from oasis_dl.data.cohort import CohortSource

        typed: CohortSource = source
        assert not typed.sessions().empty

    def test_scan_path_raises_for_unknown_session(self, source: Oasis3CohortSource) -> None:
        with pytest.raises(KeyError):
            source.scan_path(SubjectId("OAS99999"), "nope")
