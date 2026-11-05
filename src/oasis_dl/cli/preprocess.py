"""Run MRI preprocessing over the cohort.

``make preprocess``

Failures are recorded in the manifest, not dropped — the count of what preprocessing
removed is part of the cohort definition (docs/data/cohort-definition.md).
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from oasis_dl.cli import CONFIG_PATH

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    """Preprocess every session in the manifest and record per-stage outcomes."""
    log.info("preprocessing to %s voxels, shape %s", cfg.data.voxel_size_mm, cfg.data.volume_shape)
    raise NotImplementedError("Blocked on OASIS-3 access. See docs/data/access-log.md.")


if __name__ == "__main__":
    main()
