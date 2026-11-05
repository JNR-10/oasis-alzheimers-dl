"""Extracting the anatomy targets of ADR-0007.

Unimplemented, and blocked on a question that must be answered against the real
download: **which derivatives does OASIS-3 ship?**

The predecessor project parsed OASIS-1's bundled FSL segmentations and intersected
Talairach coordinate boxes with them. OASIS-3 is documented as providing FreeSurfer
outputs instead. If so, targets come from FreeSurfer's segmentation volumes — a better
source than coordinate bounding boxes, and a different pipeline rather than a port.
See docs/data/README.md, open question 1.
"""

from __future__ import annotations

from pathlib import Path

CANDIDATE_TARGETS: tuple[str, ...] = (
    "hippocampus_left_mm3",
    "hippocampus_right_mm3",
    "lateral_ventricle_left_mm3",
    "lateral_ventricle_right_mm3",
    "entorhinal_left_mm3",
    "entorhinal_right_mm3",
    "total_gray_matter_mm3",
    "csf_to_brain_ratio",
)
"""Candidate anatomy targets, pending the derivative check.

Chosen because the predecessor measured each one separating the CDR groups on OASIS-1:
hippocampal volume fell 17% / 27% and ventricular volume rose 34% / 52% across
CDR 0 -> 0.5 -> 1. They are candidates, not commitments — the final set is whatever
the cohort's derivatives actually support.
"""


def extract_anatomy_targets(
    session_derivatives: Path,
    targets: tuple[str, ...] = CANDIDATE_TARGETS,
) -> dict[str, float]:
    """Read structural quantities for one session.

    The source must be independent of the label. A target derived from a clinician's
    reading of the same scan that produced the label would reintroduce circularity
    through the back door (ADR-0007).

    Returns:
        Target name to value in mm^3 or ratio. Missing targets are omitted rather than
        zero-filled — a zero volume is a plausible-looking number that is never true.
    """
    raise NotImplementedError(
        "Blocked on confirming OASIS-3's derivative format. See docs/data/README.md."
    )


def normalize_targets(
    values: dict[str, float],
    etiv_mm3: float,
) -> dict[str, float]:
    """Scale volumetric targets by intracranial volume.

    Raw regional volumes track head size as much as atrophy, so an unnormalized target
    asks the encoder to predict how big someone's skull is. Ratios are returned
    unchanged — they are already normalized.
    """
    raise NotImplementedError("Blocked on extract_anatomy_targets.")
