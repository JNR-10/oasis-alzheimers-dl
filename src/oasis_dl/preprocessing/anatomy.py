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

import re
from pathlib import Path

# FreeSurfer ``aseg.stats`` / ``aparc.stats`` field names as shipped by OASIS-3.
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
    "lh.entorhinal",
    "rh.entorhinal",
    "CortexVol",
    "CorticalWhiteMatterVol",
    "SubCortGrayVol",
    "BrainSegVol",
    "TotalGrayVol",
    "CSF",
    "EstimatedTotalIntraCranialVol",
)
"""FreeSurfer fields we may read from shipped ``aseg.stats`` / ``aparc.stats``.

Entorhinal volumes come from the cortical parcellation (``aparc.stats``) and appear
under either ``lh.entorhinal`` / ``rh.entorhinal`` (surface table) or the aseg-style
names depending on how the spreadsheet was exported; both are accepted.
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
+ 5th ventricle, per the OASIS-3 Imaging Data Dictionary's stated formula.

These are candidates confirmed against the derivative format; the final set is
whatever the cohort's sessions actually contain, and missing targets are omitted
rather than zero-filled.
"""

_RATIO_TARGETS: frozenset[str] = frozenset({"csf_to_brain_ratio"})
"""Targets that are already normalized — ``normalize_targets`` leaves them alone."""

_MEASURE_RE = re.compile(
    r"^#\s*Measure\s+([^,]+),\s*([^,]+),\s*[^,]+,\s*([-+0-9.eE]+)",
)
_TABLE_RE = re.compile(
    r"^\s*\d+\s+\d+\s+\d+\s+([-+0-9.eE]+)\s+(\S+)",
)
_APARC_TABLE_RE = re.compile(
    r"^\s*(\S+)\s+\d+\s+\d+\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",
)


def _load_stats_map(path: Path) -> dict[str, float]:
    """Parse one FreeSurfer ``.stats`` file into ``{StructName: Volume_mm3}``."""
    values: dict[str, float] = {}
    hemi = ""
    lower_name = path.name.lower()
    if lower_name.startswith("lh."):
        hemi = "lh."
    elif lower_name.startswith("rh."):
        hemi = "rh."

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        measure = _MEASURE_RE.match(line)
        if measure:
            key_a, key_b, number = measure.groups()
            value = float(number)
            values[key_a.strip()] = value
            values[key_b.strip()] = value
            continue
        table = _TABLE_RE.match(line)
        if table:
            volume, name = table.groups()
            values[name] = float(volume)
            continue
        aparc = _APARC_TABLE_RE.match(line)
        if aparc and not line.startswith("#"):
            # aparc.stats: StructName NumVert SurfArea GrayVol ThickAvg ...
            name, gray, _thick = aparc.groups()
            volume = float(gray)
            values[name] = volume
            if hemi:
                values[f"{hemi}{name}"] = volume
            values[f"ctx-{name}"] = volume
            if hemi:
                values[f"ctx-{hemi}{name}"] = volume
    return values


def _collect_fields(session_derivatives: Path) -> dict[str, float]:
    """Load every readable stats file under a session's FreeSurfer output."""
    if not session_derivatives.exists():
        return {}

    files: list[Path]
    if session_derivatives.is_file():
        files = [session_derivatives]
    else:
        files = sorted(
            {
                *session_derivatives.rglob("aseg.stats"),
                *session_derivatives.rglob("lh.aparc.stats"),
                *session_derivatives.rglob("rh.aparc.stats"),
                *session_derivatives.rglob("aparc.stats"),
            }
        )
        # Also accept a flat CSV/TSV of name,volume rows if present.
        files.extend(sorted(session_derivatives.glob("*.csv")))

    values: dict[str, float] = {}
    for path in files:
        if path.suffix == ".csv":
            for raw in path.read_text(encoding="utf-8").splitlines():
                if not raw.strip() or raw.lstrip().startswith("#"):
                    continue
                parts = [p.strip() for p in raw.split(",")]
                if len(parts) < 2 or parts[0].lower() in {"name", "struct", "structname"}:
                    continue
                try:
                    values[parts[0]] = float(parts[1])
                except ValueError:
                    continue
            continue
        values.update(_load_stats_map(path))
    return values


def _first(*names: str, fields: dict[str, float]) -> float | None:
    for name in names:
        if name in fields:
            return fields[name]
    return None


def _build_targets(
    fields: dict[str, float],
    targets: tuple[str, ...],
) -> dict[str, float]:
    """Map FreeSurfer fields onto the project's candidate target names."""
    built: dict[str, float] = {}

    mapping: dict[str, tuple[str, ...]] = {
        "hippocampus_left_mm3": ("Left-Hippocampus",),
        "hippocampus_right_mm3": ("Right-Hippocampus",),
        "lateral_ventricle_left_mm3": ("Left-Lateral-Ventricle",),
        "lateral_ventricle_right_mm3": ("Right-Lateral-Ventricle",),
        "entorhinal_left_mm3": ("lh.entorhinal", "Left-entorhinal", "ctx-lh-entorhinal"),
        "entorhinal_right_mm3": ("rh.entorhinal", "Right-entorhinal", "ctx-rh-entorhinal"),
        "total_gray_matter_mm3": ("TotalGrayVol",),
    }

    for name, keys in mapping.items():
        if name not in targets:
            continue
        value = _first(*keys, fields=fields)
        if value is not None:
            built[name] = value

    if "total_ventricular_volume_mm3" in targets:
        parts = [
            _first("Left-Lateral-Ventricle", fields=fields),
            _first("Right-Lateral-Ventricle", fields=fields),
            _first("Left-Inf-Lat-Vent", fields=fields),
            _first("Right-Inf-Lat-Vent", fields=fields),
            _first("3rd-Ventricle", fields=fields),
            _first("4th-Ventricle", fields=fields),
            _first("5th-Ventricle", fields=fields),
        ]
        if all(part is not None for part in parts):
            built["total_ventricular_volume_mm3"] = float(sum(parts))  # type: ignore[arg-type]

    if "csf_to_brain_ratio" in targets:
        csf = _first("CSF", "VolumeOfCSF", fields=fields)
        etiv = _first("EstimatedTotalIntraCranialVol", "eTIV", fields=fields)
        if csf is not None and etiv is not None and etiv > 0:
            built["csf_to_brain_ratio"] = csf / etiv

    return built


def extract_anatomy_targets(
    session_derivatives: Path,
    targets: tuple[str, ...] = CANDIDATE_TARGETS,
) -> dict[str, float]:
    """Read structural quantities for one session from the shipped FreeSurfer output.

    The source is the OASIS-3 FreeSurfer output (``aseg.stats`` / ``aparc.stats`` as
    files or CSV), independent of this project's model and independent of the label
    (ADR-0007). The shipped outputs are used as-is; FreeSurfer is not re-run here
    (ADR-0009).

    Args:
        session_derivatives: Path to the session's FreeSurfer output directory, or a
            single ``.stats`` / ``.csv`` file.
        targets: Which candidate targets to read. Defaults to ``CANDIDATE_TARGETS``.

    Returns:
        Target name to value in mm^3 or ratio. Missing targets are omitted rather
        than zero-filled — a zero volume is a plausible-looking number that is never
        true, and a session without FreeSurfer output returns an empty dict so the
        caller can record the gap in the manifest.
    """
    fields = _collect_fields(session_derivatives)
    if not fields:
        return {}
    return _build_targets(fields, targets)


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

    Raises:
        ValueError: If ``etiv_mm3`` is not positive.
    """
    if etiv_mm3 <= 0:
        raise ValueError(f"etiv_mm3 must be positive, got {etiv_mm3}")
    return {
        name: (value if name in _RATIO_TARGETS else value / etiv_mm3)
        for name, value in values.items()
    }
