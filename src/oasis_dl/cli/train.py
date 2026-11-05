"""Train one arm.

``make train EXP=proposed_b_anatomy``
``python -m oasis_dl.cli.train experiment=proposed_a_longitudinal``

An ablation is a different ``experiment=`` override and nothing else (ADR-0008). If a
comparison needs more than one override to express, the configs are hiding an
uncontrolled difference between the arms.

Reads the train and validation partitions only. The test partition is read once, by
:mod:`oasis_dl.cli.evaluate`, after training and tuning are complete.

Implementation order: seed via :func:`oasis_dl.training.seed.seed_everything`, capture
provenance into the Hydra run directory *before* training starts, warn when the
checkout is dirty, then build the arm and train.
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from oasis_dl.cli import CONFIG_PATH

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    """Seed, capture provenance, build the arm, train it, record the result."""
    raise NotImplementedError(
        "Blocked on the dataset and model implementations. Config composition is wired: "
        "`python -m oasis_dl.cli.train --cfg job experiment=proposed_b_anatomy` resolves."
    )


if __name__ == "__main__":
    main()
