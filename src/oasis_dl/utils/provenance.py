"""Recording what produced a result.

ADR-0008: a result whose configuration cannot be recovered does not go in the report.
This module captures the context — commit, seed, versions, hardware — and writes it
beside the metrics, so that "which run was that?" has an answer months later.

Captured at the **start** of a run, not the end, so that a crashed run still leaves a
record of what was attempted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunProvenance:
    """Everything needed to identify and rerun an experiment.

    Attributes:
        run_id: Unique within ``experiments/``.
        experiment: The config name, e.g. ``proposed_b_anatomy``.
        git_commit: Commit the code was at.
        git_dirty: Whether uncommitted changes were present. A dirty run is fine while
            exploring and disqualifying for a reported number, because the code that
            produced it no longer exists anywhere.
        seed: The run's seed.
        timestamp: UTC, ISO 8601.
        manifest_path: The cohort this trained on (ADR-0008).
        python_version, platform, packages, gpu: Environment.
        config: The fully resolved config, so the run can be reconstructed without
            replaying the override chain that produced it.
    """

    run_id: str
    experiment: str
    git_commit: str
    git_dirty: bool
    seed: int
    timestamp: str
    manifest_path: str
    python_version: str
    platform: str
    packages: dict[str, str] = field(default_factory=dict)
    gpu: str | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def write(self, path: Path) -> None:
        """Write as JSON, creating parent directories."""
        raise NotImplementedError


def git_commit() -> str:
    """Current commit hash, or ``"unknown"`` outside a git checkout."""
    raise NotImplementedError


def git_is_dirty() -> bool:
    """Whether tracked files have uncommitted changes.

    Returns True when git is unavailable — the safe answer, since an unverifiable
    checkout should not be reported as clean.
    """
    raise NotImplementedError


def package_versions() -> dict[str, str]:
    """Versions of the libraries that can change a result.

    torch, monai, torchio, numpy, scikit-learn, nibabel, pandas.
    """
    raise NotImplementedError


def gpu_name() -> str | None:
    """Name of the active CUDA device, or None on CPU.

    Imports torch lazily, so that reading provenance from a saved run does not pay for
    a torch import.
    """
    raise NotImplementedError


def capture(
    run_id: str,
    experiment: str,
    seed: int,
    manifest_path: Path,
    config: dict[str, Any] | None = None,
) -> RunProvenance:
    """Capture the current environment as a :class:`RunProvenance`."""
    raise NotImplementedError
