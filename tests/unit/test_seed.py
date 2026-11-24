"""Seeding — ADR-0008.

An ablation whose arms differ partly by unrecorded randomness is not a controlled
comparison, so this is a correctness module rather than a convenience one.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from oasis_dl.training.seed import seed_everything, seed_worker


class TestSeedEverything:
    def test_it_covers_all_three_generators(self) -> None:
        """Torch alone is not enough: augmentation draws, subject sampling and shuffling
        reach for different libraries."""

        def draw() -> tuple[float, float, float]:
            seed_everything(123, deterministic=False)
            return (random.random(), float(np.random.rand()), float(torch.rand(1)))

        assert draw() == draw()

    def test_different_seeds_diverge(self) -> None:
        seed_everything(1, deterministic=False)
        first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        seed_everything(2, deterministic=False)
        assert (random.random(), float(np.random.rand()), float(torch.rand(1))) != first

    def test_deterministic_mode_sets_the_cublas_variable(self) -> None:
        """Without it some CUDA matmul reductions stay nondeterministic however the
        torch flags are set, so the flag alone would be a false promise."""
        import os

        seed_everything(0, deterministic=True)
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        seed_everything(0, deterministic=False)

    def test_deterministic_mode_is_reflected_in_the_torch_flag(self) -> None:
        seed_everything(0, deterministic=True)
        assert torch.backends.cudnn.deterministic is True
        seed_everything(0, deterministic=False)
        assert torch.backends.cudnn.deterministic is False


class TestSeedWorker:
    def test_workers_draw_differently_from_one_another(self) -> None:
        """Sharing a seed across workers makes several draw identical augmentations,
        which looks like a smaller and less varied dataset than you have."""

        def draw(worker_id: int) -> float:
            seed_everything(5, deterministic=False)
            seed_worker(worker_id)
            return random.random()

        assert draw(0) != draw(1)

    def test_a_worker_is_reproducible_for_a_given_base_seed(self) -> None:
        def draw() -> float:
            seed_everything(5, deterministic=False)
            seed_worker(3)
            return random.random()

        assert draw() == draw()
