"""Evaluate a trained arm on the held-out test subjects.

``make evaluate EXP=proposed_b_anatomy``

This is the only place the test partition is read, and it is read once per arm after
that arm's training and tuning are complete. The threshold comes from the validation
partition and is applied unchanged (docs/protocol/evaluation.md).
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
    """Score the test partition, bootstrap over subjects, write the results table.

    Reports per-visit and per-subject aggregations. They diverge when visit count
    correlates with label; when they do, both go in the report rather than whichever
    one reads better.
    """
    checkpoint = Path(cfg.evaluation.checkpoint_path)
    log.info("evaluating %s from %s", cfg.experiment.name, checkpoint)
    log.info("bootstrap: %s resamples, seed %s", cfg.evaluation.n_resamples, cfg.evaluation.seed)

    raise NotImplementedError(
        "Blocked on the training loop producing a checkpoint. The metrics and bootstrap "
        "this will call — oasis_dl.evaluation — are written and tested."
    )


if __name__ == "__main__":
    main()
