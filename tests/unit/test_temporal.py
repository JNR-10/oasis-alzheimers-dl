"""The temporal module — ADR-0006, elapsed time as an explicit continuous input.

Two properties carry this ticket and neither is visible by reading the module:

*Time is not position.* Two subjects with identical embedding sequences, four months
and four years apart, must not read identically. Atrophy **rate** is the signal; a
standard positional encoding would discard it and the model would still train, converge
and report a number.

*Padding is not history.* Attention over padded positions fabricates visits that were
never acquired. In a study about longitudinal change that is a fabricated finding, and
it is silent — the shapes are right either way.
"""

from __future__ import annotations

import pytest
import torch

from oasis_dl.models.temporal import ElapsedTimeEncoding, TemporalAttention

DIM = 32


def _module(**kwargs: object) -> TemporalAttention:
    return TemporalAttention(embedding_dim=DIM, n_heads=4, n_layers=2, dropout=0.0, **kwargs).eval()  # type: ignore[arg-type]


def _scalarize(out: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Reduce to a scalar through a random projection rather than ``.sum()``.

    Every path here ends in a LayerNorm, whose output sums to zero for *any* input
    because it is normalised to zero mean. So ``out.sum()`` has an exactly zero
    derivative with respect to everything upstream, and a gradient test built on it
    reports "no gradient" whatever the module does. torch 2.2 leaked enough float noise
    to hide that; torch 2.15 computes the clean zero and the test failed honestly.
    """
    generator = torch.Generator().manual_seed(seed)
    weights = torch.randn(out.shape, generator=generator)
    return (out * weights).sum()


def _years(*days: float) -> torch.Tensor:
    return torch.tensor([list(days)], dtype=torch.float32)


class TestElapsedTimeEncoding:
    def test_maps_elapsed_days_to_one_vector_per_visit(self) -> None:
        encoding = ElapsedTimeEncoding(embedding_dim=DIM)
        out = encoding(_years(-900, -400, 0))
        assert out.shape == (1, 3, DIM)

    def test_different_elapsed_times_encode_differently(self) -> None:
        """The whole point of ADR-0006. Position would make these identical."""
        encoding = ElapsedTimeEncoding(embedding_dim=DIM)
        four_months = encoding(_years(-120, 0))
        four_years = encoding(_years(-1461, 0))
        assert not torch.allclose(four_months, four_years)

    def test_the_same_elapsed_time_encodes_identically(self) -> None:
        encoding = ElapsedTimeEncoding(embedding_dim=DIM)
        torch.testing.assert_close(encoding(_years(-365, 0)), encoding(_years(-365, 0)))

    def test_it_is_continuous_in_time(self) -> None:
        """A continuous input, not a bucket. Two visits a day apart must encode more
        alike than two visits a decade apart."""
        encoding = ElapsedTimeEncoding(embedding_dim=DIM)
        anchor = encoding(_years(0.0))
        near = (encoding(_years(-1.0)) - anchor).abs().sum()
        far = (encoding(_years(-3650.0)) - anchor).abs().sum()
        assert near < far

    def test_it_holds_no_learnable_parameters(self) -> None:
        """A fixed basis. Learned time embeddings on 450 sequences would fit the
        cohort's particular visit spacing rather than the passage of time."""
        assert not list(ElapsedTimeEncoding(embedding_dim=DIM).parameters())

    def test_an_odd_embedding_dim_is_refused(self) -> None:
        """A sin/cos basis needs pairs; a silent off-by-one would drop a frequency."""
        with pytest.raises(ValueError):
            ElapsedTimeEncoding(embedding_dim=31)


class TestShapeContract:
    @pytest.mark.parametrize("n_visits", [1, 2, 3, 5])
    def test_any_number_of_visits_collapses_to_one_representation(self, n_visits: int) -> None:
        module = _module()
        embeddings = torch.randn(2, n_visits, DIM)
        elapsed = torch.zeros(2, n_visits)
        mask = torch.ones(2, n_visits, dtype=torch.bool)
        with torch.no_grad():
            out = module(embeddings, elapsed, mask)
        assert out.shape == (2, DIM)

    def test_a_width_the_heads_do_not_divide_is_refused(self) -> None:
        with pytest.raises(ValueError):
            TemporalAttention(embedding_dim=30, n_heads=4)


class TestPaddingIsNotHistory:
    def test_left_padding_does_not_change_the_output(self) -> None:
        """The acceptance criterion, stated directly.

        The same two real visits, alone and then left-padded to four positions, must
        produce the same representation. If they do not, the padded positions are being
        attended over and the model is reading visits that were never acquired.
        """
        torch.manual_seed(0)
        module = _module()
        real = torch.randn(1, 2, DIM)
        elapsed = _years(-400, 0)

        with torch.no_grad():
            unpadded = module(real, elapsed, torch.ones(1, 2, dtype=torch.bool))
            padded = module(
                torch.cat([torch.randn(1, 2, DIM), real], dim=1),
                torch.cat([_years(-9999, -9999), elapsed], dim=1),
                torch.tensor([[False, False, True, True]]),
            )
        torch.testing.assert_close(unpadded, padded, rtol=1e-4, atol=1e-5)

    def test_what_sits_in_the_padding_cannot_matter(self) -> None:
        """Garbage in the padded slots must be invisible, not merely small."""
        torch.manual_seed(0)
        module = _module()
        real = torch.randn(1, 2, DIM)
        elapsed = torch.cat([_years(0, 0), _years(-400, 0)], dim=1)
        mask = torch.tensor([[False, False, True, True]])

        with torch.no_grad():
            quiet = module(torch.cat([torch.zeros(1, 2, DIM), real], dim=1), elapsed, mask)
            loud = module(torch.cat([torch.full((1, 2, DIM), 1e3), real], dim=1), elapsed, mask)
        torch.testing.assert_close(quiet, loud, rtol=1e-4, atol=1e-5)

    def test_sequences_in_one_batch_do_not_leak_into_each_other(self) -> None:
        """A batch is a convenience, not a context."""
        torch.manual_seed(0)
        module = _module()
        subject = torch.randn(1, 3, DIM)
        elapsed = _years(-900, -400, 0)
        mask = torch.ones(1, 3, dtype=torch.bool)

        with torch.no_grad():
            alone = module(subject, elapsed, mask)
            batched = module(
                torch.cat([subject, torch.randn(1, 3, DIM)], dim=0),
                torch.cat([elapsed, elapsed], dim=0),
                torch.ones(2, 3, dtype=torch.bool),
            )
        torch.testing.assert_close(alone[0], batched[0], rtol=1e-4, atol=1e-5)


class TestTimeIsNotPosition:
    def test_the_same_embeddings_at_different_spacings_read_differently(self) -> None:
        """ADR-0006 as an observable property.

        Identical scans four months apart and four years apart mean different things.
        A module reading sequence position cannot tell them apart; this test fails on
        any such module.
        """
        torch.manual_seed(0)
        module = _module()
        embeddings = torch.randn(1, 3, DIM)
        mask = torch.ones(1, 3, dtype=torch.bool)

        with torch.no_grad():
            rapid = module(embeddings, _years(-240, -120, 0), mask)
            slow = module(embeddings, _years(-2922, -1461, 0), mask)
        assert not torch.allclose(rapid, slow, rtol=1e-3, atol=1e-4)

    def test_elapsed_time_reaches_the_output_through_gradients(self) -> None:
        module = TemporalAttention(embedding_dim=DIM, n_heads=4, n_layers=2, dropout=0.0)
        elapsed = _years(-400, 0).requires_grad_(True)
        out = module(torch.randn(1, 2, DIM), elapsed, torch.ones(1, 2, dtype=torch.bool))
        _scalarize(out).backward()
        assert elapsed.grad is not None and elapsed.grad.abs().sum() > 0


class TestKeptSmall:
    def test_the_defaults_are_small(self) -> None:
        """The cohort is ~450 sequences; Transformer components overfit it readily."""
        module = TemporalAttention(embedding_dim=512)
        assert module.n_layers <= 2
        assert sum(p.numel() for p in module.parameters()) < 7e6

    def test_gradients_reach_every_block(self) -> None:
        module = TemporalAttention(embedding_dim=DIM, n_heads=4, n_layers=2, dropout=0.0)
        out = module(torch.randn(2, 3, DIM), torch.zeros(2, 3), torch.ones(2, 3, dtype=torch.bool))
        _scalarize(out).backward()
        starved = [n for n, p in module.named_parameters() if p.grad is None or not p.grad.any()]
        assert not starved, f"no gradient reached: {starved}"
