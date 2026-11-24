"""Recording what produced a result.

ADR-0008: a result whose configuration cannot be recovered does not go in the report.
This module captures the context — commit, seed, versions, hardware — and writes it
beside the metrics, so that "which run was that?" has an answer months later.

Captured at the **start** of a run, not the end, so that a crashed run still leaves a
record of what was attempted.
"""

from __future__ import annotations

import json
import platform as platform_module
import subprocess
import sys
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

__all__ = [
    "PROVENANCE_FILENAME",
    "RunProvenance",
    "capture",
    "git_commit",
    "git_is_dirty",
    "gpu_name",
    "package_versions",
]

PROVENANCE_FILENAME = "provenance.json"
"""The one name for a run's provenance record.

Defined once, here, because the loop writes it and the report generator searches for
it. Two independent spellings drift, and a rename on one side would leave every results
table silently empty rather than failing.
"""

TRACKED_PACKAGES = ("torch", "monai", "torchio", "numpy", "scikit-learn", "nibabel", "pandas")
"""Libraries whose version can change a result. Anything else is noise in the record."""


def _git(args: list[str], repo: Path | None) -> str | None:
    """Run a git command, or return None when git cannot answer.

    Every failure mode collapses to None — no git binary, not a checkout, a timeout —
    because the callers must distinguish "the answer is X" from "there is no answer",
    and never invent the former.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo) if repo is not None else None,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


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
        """Write as JSON, creating parent directories.

        Sorted keys and a trailing newline, so two records diff cleanly against each
        other — comparing runs is most of what this file is for.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")


def git_commit(repo: Path | None = None) -> str:
    """Current commit hash, or ``"unknown"`` outside a git checkout.

    ``"unknown"`` rather than an empty string or a plausible-looking placeholder: the
    record has to be able to say it does not know.
    """
    return _git(["rev-parse", "HEAD"], repo) or "unknown"


def git_is_dirty(repo: Path | None = None) -> bool:
    """Whether tracked files have uncommitted changes.

    Returns True when git is unavailable — the safe answer, since an unverifiable
    checkout should not be reported as clean. The asymmetry is deliberate: a false
    "dirty" costs a second look, a false "clean" lets an unreproducible run into the
    report.
    """
    status = _git(["status", "--porcelain", "--untracked-files=no"], repo)
    return True if status is None else bool(status)


def package_versions() -> dict[str, str]:
    """Versions of the libraries that can change a result.

    torch, monai, torchio, numpy, scikit-learn, nibabel, pandas.

    A package that is not installed is **omitted**, never recorded as "unknown" — every
    version in the record is one that was really there.
    """
    found: dict[str, str] = {}
    for package in TRACKED_PACKAGES:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            continue
    return found


def gpu_name() -> str | None:
    """Name of the active CUDA device, or None on CPU.

    Imports torch lazily, so that reading provenance from a saved run does not pay for
    a torch import.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return str(torch.cuda.get_device_name(torch.cuda.current_device()))


def capture(
    run_id: str,
    experiment: str,
    seed: int,
    manifest_path: Path,
    config: dict[str, Any] | None = None,
    repo: Path | None = None,
) -> RunProvenance:
    """Capture the current environment as a :class:`RunProvenance`.

    Warns on a dirty checkout. The flag alone is a field nobody reads; the warning is
    what makes it a control, because the moment to notice is while the run is being
    started rather than while the report is being written.

    Args:
        repo: Checkout to inspect. Defaults to the working directory, which is what a
            CLI run wants; tests pass their own.
    """
    dirty = git_is_dirty(repo)
    if dirty:
        warnings.warn(
            f"the checkout is dirty, so run {run_id!r} is not reproducible from its "
            f"commit alone. Fine while exploring; disqualifying for a reported number, "
            f"because the code that produced it does not exist anywhere (ADR-0008).",
            stacklevel=2,
        )
    return RunProvenance(
        run_id=run_id,
        experiment=experiment,
        git_commit=git_commit(repo),
        git_dirty=dirty,
        seed=seed,
        timestamp=datetime.now(timezone.utc).isoformat(),
        manifest_path=str(manifest_path),
        python_version=sys.version.split()[0],
        platform=platform_module.platform(),
        packages=package_versions(),
        gpu=gpu_name(),
        config=config or {},
    )
