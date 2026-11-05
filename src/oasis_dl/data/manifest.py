"""The cohort manifest — the single source of truth for what was trained on.

A manifest pins the cohort: which subjects, which visits, which partition, which label,
under which seed. Results are reproducible only to the extent that the manifest that
produced them is recoverable, so manifests are written once and treated as immutable
(ADR-0008) — writing over an existing path is an error, not an overwrite.

Manifests carry subject identifiers and therefore never enter version control. They
live under ``data/manifests/``, which is gitignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

COLUMNS: Final[tuple[str, ...]] = (
    "subject_id",
    "session_id",
    "partition",
    "days_from_entry",
    "scan_path",
    "label",
    "is_target_visit",
    "exclusion_reason",
)
"""The manifest schema.

``exclusion_reason`` is non-empty for sessions that did not become visits. Excluded
sessions stay in the file rather than being dropped, because the count of what was
removed and why is part of the cohort definition
(``docs/data/cohort-definition.md``) and cannot be recovered once discarded.
"""


def write_manifest(frame: pd.DataFrame, path: Path) -> None:
    """Write a manifest, refusing to overwrite an existing one.

    Overwriting is refused because a manifest is the record of a run that may already
    have produced results. A new cohort means a new filename, which keeps the old
    results interpretable.

    Raises:
        FileExistsError: If ``path`` already exists.
        ValueError: If any of :data:`COLUMNS` is missing.
    """
    raise NotImplementedError


def read_manifest(path: Path) -> pd.DataFrame:
    """Read a manifest and validate its schema.

    Raises:
        FileNotFoundError: If the manifest is absent.
        ValueError: If any of :data:`COLUMNS` is missing.
    """
    raise NotImplementedError
