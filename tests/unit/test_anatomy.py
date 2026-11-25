"""Tests for FreeSurfer anatomy target extraction (ADR-0009 / issue #17)."""

from __future__ import annotations

from pathlib import Path

import pytest

from oasis_dl.preprocessing.anatomy import (
    CANDIDATE_TARGETS,
    extract_anatomy_targets,
    normalize_targets,
)

ASEG_STATS = """\
# Measure BrainSeg, BrainSegVol, Brain Segmentation Volume, 1100000.000000, mm^3
# Measure TotalGray, TotalGrayVol, Total Gray Matter Volume, 600000.000000, mm^3
# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated Total Intracranial Volume, 1500000.000000, mm^3
# ColHeaders Index SegId NVoxels Volume_mm3 StructName
 1 17 4000 4000.0 Left-Hippocampus
 2 53 4100 4100.0 Right-Hippocampus
 3 4 12000 12000.0 Left-Lateral-Ventricle
 4 43 13000 13000.0 Right-Lateral-Ventricle
 5 5 800 800.0 Left-Inf-Lat-Vent
 6 44 900 900.0 Right-Inf-Lat-Vent
 7 14 1000 1000.0 3rd-Ventricle
 8 15 1100 1100.0 4th-Ventricle
 9 72 50 50.0 5th-Ventricle
 10 24 1445 1445.2 CSF
"""

LH_APARC = """\
# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd
entorhinal 100 200 1200.0 3.0 0.5 0.1 0.01 10 1.0
"""

RH_APARC = """\
# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd
entorhinal 110 210 1300.0 3.1 0.5 0.1 0.01 11 1.0
"""


@pytest.fixture
def freesurfer_session(tmp_path: Path) -> Path:
    stats = tmp_path / "stats"
    stats.mkdir()
    (stats / "aseg.stats").write_text(ASEG_STATS, encoding="utf-8")
    (stats / "lh.aparc.stats").write_text(LH_APARC, encoding="utf-8")
    (stats / "rh.aparc.stats").write_text(RH_APARC, encoding="utf-8")
    return tmp_path


class TestExtractAnatomyTargets:
    def test_reads_shipped_stats_into_candidate_names(self, freesurfer_session: Path) -> None:
        targets = extract_anatomy_targets(freesurfer_session)
        assert targets["hippocampus_left_mm3"] == 4000.0
        assert targets["hippocampus_right_mm3"] == 4100.0
        assert targets["lateral_ventricle_left_mm3"] == 12000.0
        assert targets["entorhinal_left_mm3"] == 1200.0
        assert targets["entorhinal_right_mm3"] == 1300.0
        assert targets["total_gray_matter_mm3"] == 600000.0
        assert targets["total_ventricular_volume_mm3"] == (
            12000 + 13000 + 800 + 900 + 1000 + 1100 + 50
        )

    def test_the_csf_segmentation_label_is_not_read_as_total_csf(
        self, freesurfer_session: Path
    ) -> None:
        """FreeSurfer writes no total-CSF measure. Its ``CSF`` row is segmentation label
        24, about 1.4 cm^3 of extra-ventricular CSF, so a ratio built on it would be a
        quietly wrong training target roughly a hundred times too small. The fixture
        follows the real aseg.stats layout (FreeSurfer 5.3), where CSF appears only as
        that table row."""
        targets = extract_anatomy_targets(freesurfer_session)
        assert "csf_to_brain_ratio" not in targets

    def test_a_one_row_per_session_spreadsheet_is_refused_not_read_as_empty(
        self, tmp_path: Path
    ) -> None:
        """A spreadsheet laid out one row per session must not come back as ``{}``,
        which callers read as "this session has no FreeSurfer output" — every training
        subject would silently lose its anatomy targets and arm PB would quietly become
        arm PA."""
        sheet = tmp_path / "freesurfer.csv"
        sheet.write_text(
            "session,Left-Hippocampus,Right-Hippocampus\nS1_d0001,3500.0,3600.0\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="not a recognized"):
            extract_anatomy_targets(sheet)

    def test_stats_files_that_yield_nothing_are_an_error(self, tmp_path: Path) -> None:
        stats = tmp_path / "stats"
        stats.mkdir()
        (stats / "aseg.stats").write_text("this is not FreeSurfer output\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no FreeSurfer field"):
            extract_anatomy_targets(tmp_path)

    def test_missing_directory_returns_empty(self, tmp_path: Path) -> None:
        assert extract_anatomy_targets(tmp_path / "absent") == {}

    def test_partial_stats_omit_rather_than_zero_fill(self, tmp_path: Path) -> None:
        stats = tmp_path / "stats"
        stats.mkdir()
        (stats / "aseg.stats").write_text(
            "# Measure EstimatedTotalIntraCranialVol, eTIV, x, 1000.0, mm^3\n"
            "# ColHeaders Index SegId NVoxels Volume_mm3 StructName\n"
            " 1 17 4000 4000.0 Left-Hippocampus\n",
            encoding="utf-8",
        )
        targets = extract_anatomy_targets(tmp_path)
        assert targets == {"hippocampus_left_mm3": 4000.0}
        assert "total_ventricular_volume_mm3" not in targets
        assert "csf_to_brain_ratio" not in targets

    def test_subset_of_targets(self, freesurfer_session: Path) -> None:
        targets = extract_anatomy_targets(
            freesurfer_session,
            targets=("hippocampus_left_mm3",),
        )
        assert set(targets) == {"hippocampus_left_mm3"}

    def test_csv_flat_file(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "volumes.csv"
        csv_path.write_text(
            "name,volume\nLeft-Hippocampus,3210.5\nRight-Hippocampus,3300.0\n",
            encoding="utf-8",
        )
        targets = extract_anatomy_targets(csv_path, targets=CANDIDATE_TARGETS)
        assert targets["hippocampus_left_mm3"] == 3210.5
        assert targets["hippocampus_right_mm3"] == 3300.0


class TestNormalizeTargets:
    def test_volumes_scale_by_etiv_ratios_do_not(self) -> None:
        values = {
            "hippocampus_left_mm3": 4000.0,
            "csf_to_brain_ratio": 0.1,
        }
        normalized = normalize_targets(values, etiv_mm3=2000.0)
        assert normalized["hippocampus_left_mm3"] == 2.0
        assert normalized["csf_to_brain_ratio"] == 0.1

    def test_non_positive_etiv_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            normalize_targets({"hippocampus_left_mm3": 1.0}, etiv_mm3=0.0)
