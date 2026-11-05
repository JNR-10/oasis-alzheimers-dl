"""Seeding, so that a run can be repeated.

Determinism is not free — deterministic cuDNN kernels are slower — but a result whose
seed is unrecorded cannot be reproduced, and an ablation whose arms differ partly by
unrecorded randomness is not a controlled comparison.
"""

from __future__ import annotations


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and Torch, and optionally force deterministic kernels.

    Deterministic mode also needs ``CUBLAS_WORKSPACE_CONFIG`` set, or some CUDA matmul
    reductions stay nondeterministic regardless of the torch flags.

    Args:
        seed: Recorded with the run (see :mod:`oasis_dl.utils.provenance`).
        deterministic: Trade throughput for exact repeatability. On for reported runs;
            worth turning off while exploring.
    """
    raise NotImplementedError


def seed_worker(worker_id: int) -> None:
    """Seed a dataloader worker. Pass as ``worker_init_fn``.

    Without this, workers inherit a shared seed and several can draw identical
    augmentations — which looks like a smaller, less varied dataset than you have.
    """
    raise NotImplementedError
