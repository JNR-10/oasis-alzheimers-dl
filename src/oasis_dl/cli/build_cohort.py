"""Build the cohort manifest and draw the partitions. Run once.

``make cohort``

The partitions this writes are drawn once and never redrawn (ADR-0001). Running this
a second time against an existing manifest path fails rather than overwriting, because
the old manifest may already have produced results.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from oasis_dl.cli import CONFIG_PATH

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    """Apply inclusion rules, build sequences, draw partitions, write the manifest."""
    manifest_path = Path(cfg.data.manifest_path)

    log.info("building cohort from %s", cfg.data.root)
    log.info("writing manifest to %s", manifest_path)
    log.info("partition seed: %s", cfg.data.split_seed)

    raise NotImplementedError(
        "Blocked on OASIS-3 access (docs/data/access-log.md) and on ADR-0004 reaching "
        "accepted status. The partitioning this will call — oasis_dl.data.splits — is "
        "written and tested; the cohort construction that feeds it is not."
    )


if __name__ == "__main__":
    main()
