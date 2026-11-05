"""The seven-stage MRI preprocessing pipeline of docs/protocol/preprocessing.md.

Unimplemented — needs real scans. The contract is written out because preprocessing
failures silently change the cohort, which is how a pipeline quietly becomes a
different experiment.

Stage order is load-bearing: bias correction before brain extraction (the field biases
the extraction), extraction before registration (non-brain tissue misleads alignment).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Stage(str, Enum):
    """The pipeline stages, in order."""

    ORIENT = "orient"
    BIAS_CORRECT = "bias_correct"
    BRAIN_EXTRACT = "brain_extract"
    REGISTER = "register"
    RESAMPLE = "resample"
    CROP_PAD = "crop_pad"
    NORMALIZE_INTENSITY = "normalize_intensity"


@dataclass(frozen=True)
class PreprocessingConfig:
    """Pipeline parameters, identical for every arm.

    A difference in preprocessing between arms would confound the ablations, so this
    config is fixed before the first run and its version recorded with every result.

    Attributes:
        voxel_size_mm: Target isotropic spacing.
        volume_shape: Fixed output dimensions after crop or pad.
        registration_target: ``"template"`` registers each scan to a common template;
            ``"within_subject_then_template"`` aligns a subject's visits to each other
            first. The second is preferred — registering every scan independently to a
            template risks absorbing the within-subject atrophy the temporal module
            exists to detect — but the choice is confirmed empirically on a handful of
            subjects before the cohort is processed, and the outcome recorded as an ADR.
        intensity_normalization: ``"zscore"`` or ``"percentile"``.
    """

    voxel_size_mm: tuple[float, float, float] = (1.5, 1.5, 1.5)
    volume_shape: tuple[int, int, int] = (128, 128, 128)
    registration_target: str = "within_subject_then_template"
    intensity_normalization: str = "zscore"


@dataclass
class StageResult:
    """What one stage did to one session.

    Failures are recorded rather than raised, so that one bad scan does not end a
    cohort-wide run and the failure count reaches the cohort definition instead of a
    log nobody reads.
    """

    stage: Stage
    succeeded: bool
    message: str = ""
    output_path: Path | None = None


def preprocess_session(
    scan_path: Path,
    output_path: Path,
    config: PreprocessingConfig,
) -> list[StageResult]:
    """Run every stage over one scan.

    Returns:
        One result per stage attempted. Stops at the first failure — later stages on a
        failed scan produce output that looks valid and is not.
    """
    raise NotImplementedError("Blocked on OASIS-3 access. See docs/data/access-log.md.")
