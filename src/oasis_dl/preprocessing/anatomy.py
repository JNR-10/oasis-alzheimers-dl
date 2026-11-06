"""Extracting the anatomy targets of ADR-0007.

The derivative question is settled: OASIS-3 ships FreeSurfer outputs per MR session
(subcortical volumes from ``aseg.stats``, cortical region measures from
``aparc.stats``, distributed as CSV spreadsheets alongside the full FreeSurfer output
tree). See ADR-0009 and the dataset card, open question 1.

Targets are read from the shipped outputs, not computed by re-running FreeSurfer in
this project — re-running produces different values than the dataset ships, and a
target source we control would re-introduce the circularity ADR-0007 was written to
prevent. Not every session has FreeSurfer output (only those that passed the data
holder's QC), so a session without derivatives has no targets; under ADR-0007 that
costs nothing for a test subject and excludes a training subject from the
auxiliary-head loss only.
"""

from __future__ import annotations

from pathlib import Path

# FreeSurfer ``aseg.stats`` field names as shipped by OASIS-3. See ADR-0009 for the
# source and the reasoning behind reading shipped outputs rather than re-running.
FREESURFER_ASEG_FIELDS: tuple[str, ...] = (
    "Left-Hippocampus",
    "Right-Hippocampus",
    "Left-Lateral-Ventricle",
    "Right-Lateral-Ventricle",
    "Left-Inf-Lat-Vent",
    "Right-Inf-Lat-Vent",
    "3rd-Ventricle",
    "4th-Ventricle",
    "5th-Ventricle",
    "Left-entorhinal",
    "Right-entorhinal",
    "CortexVol",
    "CorticalWhiteMatterVol",
    "SubCortGrayVol",
    "BrainSegVol",
    "TotalGrayVol",
    "VolumeOfCSF",
    "EstimatedTotalIntraCranialVol",
)
"""FreeSurfer ``aseg.stats`` / ``aparc.stats`` fields we read from the shipped CSV.

Names follow FreeSurfer's stats convention as exposed in the OASIS-3 FreeSurfer output
spreadsheet. Entorhinal volumes come from the cortical parcellation (``aparc.stats``)
but are listed here alongside the subcortical fields for a single target registry.
"""

CANDIDATE_TARGETS: tuple[str, ...] = (
    "hippocampus_left_mm3",
    "hippocampus_right_mm3",
    "lateral_ventricle_left_mm3",
    "lateral_ventricle_right_mm3",
    "total_ventricular_volume_mm3",
    "entorhinal_left_mm3",
    "entorhinal_right_mm3",
    "total_gray_matter_mm3",
    "csf_to_brain_ratio",
)
"""Candidate anatomy targets, supported by the shipped FreeSurfer outputs.

``total_ventricular_volume_mm3`` is the measure that matches the predecessor's
finding (ventricular volume rose 34% / 52% across CDR 0 -> 0.5 -> 1) and is computed
as left/right lateral ventricle + left/right inferior-lateral ventricle + 3rd + 4th
+ 5th ventricle, per the OASIS-3 Imaging Data Dictionary's stated formula. It is
preferred over lateral ventricle alone, which is retained as a component.

``total_gray_matter_mm3`` and ``csf_to_brain_ratio`` are derived from the shipped
``TotalGrayVol`` / ``BrainSegVol`` / ``VolumeOfCSF`` / eTIV fields rather than
computed by an external pipeline.

These are candidates confirmed against the derivative format; the final set is
whatever the cohort's sessions actually contain, and missing targets are omitted
rather than zero-filled.
"""


def extract_anatomy_targets(
    session_derivatives: Path,
    targets: tuple[str, ...] = CANDIDATE_TARGETS,
) -> dict[str, float]:
    """Read structural quantities for one session from the shipped FreeSurfer output.

    The source is the OASIS-3 FreeSurfer output (``aseg.stats`` / ``aparc.stats`` as
    CSV), independent of this project's model and independent of the label
    (ADR-0007). The shipped outputs are used as-is; FreeSurfer is not re-run here
    (ADR-0009).

    Args:
        session_derivatives: Path to the session's FreeSurfer output (the CSV
            ``aseg.stats`` / ``aparc.stats`` or the FreeSurfer output directory).
        targets: Which candidate targets to read. Defaults to ``CANDIDATE_TARGETS``.

    Returns:
        Target name to value in mm^3 or ratio. Missing targets are omitted rather
        than zero-filled — a zero volume is a plausible-looking number that is never
        true, and a session without FreeSurfer output returns an empty dict so the
        caller can record the gap in the manifest.
    """
    raise NotImplementedError(
        "Blocked on OASIS-3 access (issue #1) and the frozen cohort (issue #14). "
        "The derivative format is settled (ADR-0009); the parser is not yet written."
    )


def normalize_targets(
    values: dict[str, float],
    etiv_mm3: float,
) -> dict[str, float]:
    """Scale volumetric targets by intracranial volume.

    Raw regional volumes track head size as much as atrophy, so an unnormalized target
    asks the encoder to predict how big someone's skull is. Ratios are returned
    unchanged — they are already normalized. eTIV is read per session from the shipped
    FreeSurfer output and varies within a subject across sessions (mean subject SD
    ~15.75 cm^3 per the OASIS-3 Imaging Data Dictionary), so normalization is a
    per-session scaling, not a per-subject constant.
    """
    raise NotImplementedError("Blocked on extract_anatomy_targets.")
