"""Seeding, so that a run can be repeated.

Determinism is not free — deterministic cuDNN kernels are slower — but a result whose
seed is unrecorded cannot be reproduced, and an ablation whose arms differ partly by
unrecorded randomness is not a controlled comparison.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["seed_everything", "seed_worker"]


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and Torch, and optionally force deterministic kernels.

    Deterministic mode also needs ``CUBLAS_WORKSPACE_CONFIG`` set, or some CUDA matmul
    reductions stay nondeterministic regardless of the torch flags.

    Args:
        seed: Recorded with the run (see :mod:`oasis_dl.utils.provenance`).
        deterministic: Trade throughput for exact repeatability. On for reported runs;
            worth turning off while exploring.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Set before any CUDA context work. Without it some cuBLAS reductions stay
        # nondeterministic whatever the torch flags say, which would make this
        # function a promise it does not keep.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int) -> None:
    """Seed a dataloader worker. Pass as ``worker_init_fn``.

    Without this, workers inherit a shared seed and several can draw identical
    augmentations — which looks like a smaller, less varied dataset than you have.
    """
    # Derived from torch's own per-worker seed, so the whole scheme still follows from
    # the one seed recorded with the run.
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
