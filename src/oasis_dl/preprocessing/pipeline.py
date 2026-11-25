"""The seven-stage MRI preprocessing pipeline of docs/protocol/preprocessing.md.

Unimplemented beyond the config contract. Runs over the sampled subjects only
(ADR-0010), never the whole download. Most of the project's one-time CPU cost is in
these stages, and the sample is what bounds it.

Stage order is load-bearing: bias correction before brain extraction (the field biases
the extraction), extraction before registration (non-brain tissue misleads alignment).

ADR-0011 constrains stages 2-5 to the BrainIAC recipe: N4 bias correction, HD-BET
extraction, rigid registration to the BrainIAC template, then resize to 96^3. The open
question for #16 is whether visits are aligned within-subject before that template
step - not which template.
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
        volume_shape: Fixed output dimensions after crop or pad. 96^3 under ADR-0010 /
            ADR-0011 - the BrainIAC ViT input shape.
        voxel_size_mm: Effective spacing after resize to ``volume_shape``. The BrainIAC
            template is 170x206x162 at 1 mm, so resizing to a cube is anisotropic;
            fill this from the template header in #15 rather than assuming 2 mm
            isotropic. ``None`` means not yet measured.
        registration_target: ``"template"`` registers each scan to the BrainIAC
            template independently; ``"within_subject_then_template"`` aligns a
            subject's visits to each other first. The second is preferred - registering
            every scan independently risks absorbing the within-subject atrophy the
            temporal module exists to detect - but the choice is confirmed empirically
            on a handful of subjects (#16) before the cohort is processed.
        template_path: Path to the BrainIAC template (1 mm LAS, 170x206x162). Read from
            the environment / config, never committed.
        bias_correction: Fixed to N4 under ADR-0011.
        brain_extraction: Fixed to HD-BET under ADR-0011.
        intensity_normalization: ``"zscore"`` or ``"percentile"``.
    """

    volume_shape: tuple[int, int, int] = (96, 96, 96)
    voxel_size_mm: tuple[float, float, float] | None = None
    registration_target: str = "within_subject_then_template"
    template_path: Path | None = None
    bias_correction: str = "n4"
    brain_extraction: str = "hd_bet"
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
    raise NotImplementedError(
        "Blocked on the subject sample from #14 and on settling #16's within-subject "
        "registration choice. Config defaults follow ADR-0011 (N4, HD-BET, BrainIAC "
        "template, 96^3)."
    )
