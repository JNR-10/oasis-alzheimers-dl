"""The shared visit encoder.

Testable on random tensors — it needs neither the synthetic cohort nor real data,
which is why this ticket was startable on day one.

"Shared" (ADR-0005) means one set of weights encodes every visit of every subject.
The tests below pin that down as an observable property rather than a comment: the
encoder holds no per-position parameters, so two identical volumes anywhere in a batch
must produce identical embeddings.
"""

from __future__ import annotations

import pytest
import torch

from oasis_dl.models.encoder import VisitEncoder


class TestShapeContract:
    def test_maps_a_batch_of_volumes_to_a_batch_of_embeddings(self) -> None:
        encoder = VisitEncoder(embedding_dim=512).eval()
        with torch.no_grad():
            out = encoder(torch.randn(3, 1, 16, 16, 16))
        assert out.shape == (3, 512)

    @pytest.mark.parametrize("embedding_dim", [64, 128, 512])
    def test_output_width_follows_the_configured_embedding_dim(self, embedding_dim: int) -> None:
        encoder = VisitEncoder(embedding_dim=embedding_dim).eval()
        with torch.no_grad():
            out = encoder(torch.randn(2, 1, 16, 16, 16))
        assert out.shape == (2, embedding_dim)

    def test_works_at_the_configured_production_shape(self) -> None:
        """96^3 is the ADR-0010 input shape. If it does not fit, nothing else matters."""
        encoder = VisitEncoder(embedding_dim=128).eval()
        with torch.no_grad():
            out = encoder(torch.randn(1, 1, 96, 96, 96))
        assert out.shape == (1, 128)

    def test_rejects_a_wrongly_shaped_input(self) -> None:
        encoder = VisitEncoder(embedding_dim=64).eval()
        with pytest.raises(ValueError, match="5-dimensional"):
            encoder(torch.randn(2, 16, 16, 16))


class TestBackboneSelection:
    @pytest.mark.parametrize("backbone", ["resnet10", "resnet18", "densenet121"])
    def test_every_supported_backbone_yields_the_configured_width(self, backbone: str) -> None:
        encoder = VisitEncoder(backbone=backbone, embedding_dim=64).eval()
        with torch.no_grad():
            # 32 clears every backbone's minimum, DenseNet's included.
            out = encoder(torch.randn(2, 1, 32, 32, 32))
        assert out.shape == (2, 64)

    def test_a_volume_below_the_backbone_minimum_names_the_cause(self) -> None:
        """DenseNet on the synthetic cohort's default 16^3 is a real mistake to make."""
        encoder = VisitEncoder(backbone="densenet121", embedding_dim=64).eval()
        with pytest.raises(ValueError, match="at least 32 voxels"):
            encoder(torch.randn(1, 1, 16, 16, 16))

    def test_an_unknown_backbone_names_the_supported_ones(self) -> None:
        with pytest.raises(ValueError, match="resnet18"):
            VisitEncoder(backbone="not-a-backbone")

    def test_supported_backbones_is_discoverable(self) -> None:
        assert "resnet18" in VisitEncoder.supported_backbones()


class TestSharedWeights:
    def test_identical_volumes_encode_identically_anywhere_in_the_batch(self) -> None:
        """No per-position parameters: position in the batch cannot change the result."""
        encoder = VisitEncoder(embedding_dim=64).eval()
        volume = torch.randn(1, 1, 16, 16, 16)
        batch = torch.cat([volume, torch.randn(1, 1, 16, 16, 16), volume], dim=0)
        with torch.no_grad():
            out = encoder(batch)
        torch.testing.assert_close(out[0], out[2])

    def test_one_parameter_set_regardless_of_how_many_visits_are_encoded(self) -> None:
        encoder = VisitEncoder(embedding_dim=64)
        before = sum(p.numel() for p in encoder.parameters())
        with torch.no_grad():
            encoder.eval()(torch.randn(7, 1, 16, 16, 16))
        after = sum(p.numel() for p in encoder.parameters())
        assert before == after

    def test_gradients_reach_the_backbone(self) -> None:
        """Every visit must be training signal for the encoder, not just the last."""
        encoder = VisitEncoder(embedding_dim=32)
        out = encoder(torch.randn(2, 1, 16, 16, 16))
        out.sum().backward()
        grads = [p.grad for p in encoder.parameters() if p.requires_grad and p.grad is not None]
        assert grads, "no parameter received a gradient"
        assert any(g.abs().sum() > 0 for g in grads)


class TestDeterminism:
    def test_the_same_seed_builds_the_same_weights(self) -> None:
        torch.manual_seed(0)
        a = VisitEncoder(embedding_dim=32)
        torch.manual_seed(0)
        b = VisitEncoder(embedding_dim=32)
        for left, right in zip(a.parameters(), b.parameters(), strict=True):
            torch.testing.assert_close(left, right)


@pytest.mark.slow
class TestMemoryFootprint:
    """#5: the footprint at the configured input shape, measured not typed.

    This test *is* the record. Run `make test-all` to regenerate it; the numbers are
    machine-specific, so they belong in a run you can repeat rather than typed into a
    protocol document as though they were a property of the project.
    """

    @pytest.mark.parametrize("backbone", ["resnet10", "resnet18", "resnet34"])
    def test_footprint_at_the_production_shape(
        self, backbone: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import resource
        import sys

        def peak_rss_mb() -> float:
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return peak / (1024**2 if sys.platform == "darwin" else 1024)

        encoder = VisitEncoder(backbone=backbone, embedding_dim=512)
        params = sum(p.numel() for p in encoder.parameters())
        param_mb = params * 4 / 1024**2

        before = peak_rss_mb()
        encoder(torch.randn(2, 1, 96, 96, 96)).sum().backward()
        after = peak_rss_mb()

        with capsys.disabled():
            print(
                f"\n  {backbone:11s} {params:>11,} params  {param_mb:>6.1f} MB fp32  "
                f"peak RSS {after:>6.0f} MB (+{max(after - before, 0):.0f})"
            )

        # A loose ceiling only. An assertion tuned to one machine fails on the next
        # and teaches nobody anything; the printed line is the deliverable.
        assert param_mb < 500, "implausibly large for a 3D ResNet"
        assert after < 16_000, "footprint would not fit a typical training GPU"
