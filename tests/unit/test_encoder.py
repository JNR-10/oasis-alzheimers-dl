"""The shared visit encoder.

Testable on random tensors — it needs neither the synthetic cohort nor real data,
which is why this ticket was startable on day one.

"Shared" (ADR-0005) means one set of weights encodes every visit of every subject.
The tests below pin that down as an observable property rather than a comment: the
encoder holds no per-position parameters, so two identical volumes anywhere in a batch
must produce identical embeddings.
"""

from __future__ import annotations

import re
from pathlib import Path

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

    Read the RSS figures as a **process** high-water mark, not a per-backbone cost.
    ``ru_maxrss`` never falls, and these parametrisations share one process, so the
    first to run absorbs the allocator pool and the rest look free. To cost one
    backbone against another, run a single parametrisation in its own process.
    """

    @pytest.mark.parametrize("backbone", ["resnet10", "resnet18", "vit_base_16"])
    def test_footprint_at_the_production_shape(
        self, backbone: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import resource
        import sys

        def peak_rss_mb() -> float:
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return peak / (1024**2 if sys.platform == "darwin" else 1024)

        encoder = VisitEncoder(backbone=backbone, embedding_dim=512, volume_shape=(96, 96, 96))
        params = sum(p.numel() for p in encoder.parameters())
        param_mb = params * 4 / 1024**2

        encoder(torch.randn(2, 1, 96, 96, 96)).sum().backward()
        peak = peak_rss_mb()

        with capsys.disabled():
            print(
                f"\n  {backbone:11s} {params:>11,} params  {param_mb:>6.1f} MB fp32  "
                f"process peak RSS {peak:>6.0f} MB"
            )

        # A loose ceiling only. An assertion tuned to one machine fails on the next
        # and teaches nobody anything; the printed line is the deliverable.
        assert param_mb < 1000, "implausibly large for a 3D encoder"
        assert peak < 16_000, "footprint would not fit a typical training GPU"


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


TINY = (16, 16, 16)
"""A shape every backbone here accepts, and small enough to run on a laptop."""


def _tiny_vit(**kwargs: object) -> VisitEncoder:
    """A ViT at 1/47th of ViT-B, exercising the identical code path.

    Production is ``vit_base_16`` at 96^3 (ADR-0011) and takes well over a second per
    forward pass on CPU. Tests that used it would stop being run.
    """
    return VisitEncoder(backbone="vit_tiny_8", volume_shape=TINY, **kwargs)  # type: ignore[arg-type]


class TestVisionTransformerBackbone:
    def test_emits_the_configured_width(self) -> None:
        encoder = _tiny_vit(embedding_dim=64).eval()
        with torch.no_grad():
            out = encoder(torch.randn(2, 1, *TINY))
        assert out.shape == (2, 64)

    def test_keeps_the_shared_weight_property(self) -> None:
        """ADR-0005 again, for a backbone where it is not obvious.

        A ViT adds positional embeddings per *patch*. Pooling over tokens is what keeps
        the batch position irrelevant, so this is the test that pooling choice earns.
        """
        encoder = _tiny_vit(embedding_dim=32).eval()
        volume = torch.randn(1, 1, *TINY)
        batch = torch.cat([volume, torch.randn(1, 1, *TINY), volume], dim=0)
        with torch.no_grad():
            out = encoder(batch)
        torch.testing.assert_close(out[0], out[2])

    def test_a_shape_it_was_not_built_for_names_the_cause(self) -> None:
        """Positional embeddings are sized at construction; resizing input is not a fix."""
        encoder = _tiny_vit(embedding_dim=32).eval()
        with pytest.raises(ValueError, match="was built for volume_shape"):
            encoder(torch.randn(1, 1, 32, 32, 32))

    def test_a_volume_shape_the_patches_do_not_tile_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not divisible by the patch size"):
            VisitEncoder(backbone="vit_tiny_8", volume_shape=(20, 20, 20))

    def test_gradients_reach_the_transformer(self) -> None:
        encoder = _tiny_vit(embedding_dim=32)
        _scalarize(encoder(torch.randn(2, 1, *TINY))).backward()
        grads = [p.grad for p in encoder.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads)


class TestPretrainedCheckpoints:
    """Loading third-party weights, and the one failure mode that is silent.

    ``strict=False`` is unavoidable across third-party checkpoints, and it reports a
    total key mismatch the same way it reports a harmless one: by returning. A run that
    believed itself pretrained and was not would be indistinguishable from a real result.
    """

    def test_random_initialization_is_the_default(self) -> None:
        assert VisitEncoder(backbone="resnet10").pretrained is None

    def test_an_unknown_checkpoint_names_the_known_ones(self) -> None:
        with pytest.raises(ValueError, match="brainiac_vit_b"):
            VisitEncoder(backbone="vit_base_16", pretrained="not-a-checkpoint")

    def test_a_checkpoint_for_another_backbone_is_refused(self) -> None:
        with pytest.raises(ValueError, match="holds vit_base_16 weights"):
            VisitEncoder(backbone="resnet18", pretrained="brainiac_vit_b")

    def test_naming_a_checkpoint_without_a_path_says_where_to_get_it(self) -> None:
        """Weights never enter git and nothing here downloads them."""
        with pytest.raises(ValueError, match=re.escape("github.com/AIM-KannLab/BrainIAC")):
            VisitEncoder(backbone="vit_base_16", pretrained="brainiac_vit_b")

    def test_provenance_records_the_oasis_overlap(self) -> None:
        """The fact a reviewer will ask about, kept beside the weights that carry it."""
        spec = VisitEncoder.checkpoint_provenance("brainiac_vit_b")
        assert spec.overlaps_oasis is True
        assert any("OASIS" in corpus for corpus in spec.pretraining_corpus)

    def test_loading_a_contaminated_checkpoint_warns_and_names_oasis(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checkpoint = _write_tiny_checkpoint(tmp_path, monkeypatch, overlaps_oasis=True)
        with pytest.warns(UserWarning, match="OASIS-3"):
            VisitEncoder(
                backbone="vit_tiny_8",
                volume_shape=TINY,
                pretrained="test_tiny",
                checkpoint_path=checkpoint,
            )

    def test_weights_actually_arrive_in_the_backbone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checkpoint = _write_tiny_checkpoint(tmp_path, monkeypatch, overlaps_oasis=False)
        loaded = VisitEncoder(
            backbone="vit_tiny_8",
            volume_shape=TINY,
            pretrained="test_tiny",
            checkpoint_path=checkpoint,
        )
        saved = torch.load(checkpoint, map_location="cpu")
        name, expected = next(iter(saved.items()))
        torch.testing.assert_close(dict(loaded.body.state_dict())[name], expected)

    def test_a_checkpoint_matching_nothing_is_an_error_not_a_silent_random_init(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The failure this whole class exists for."""
        _write_tiny_checkpoint(tmp_path, monkeypatch, overlaps_oasis=False)
        wrong = tmp_path / "wrong.pt"
        torch.save({"nothing.like.a.vit.key": torch.zeros(3)}, wrong)
        with pytest.raises(ValueError, match="matched no parameter"):
            VisitEncoder(
                backbone="vit_tiny_8",
                volume_shape=TINY,
                pretrained="test_tiny",
                checkpoint_path=wrong,
            )

    def test_a_missing_checkpoint_file_is_reported_as_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_tiny_checkpoint(tmp_path, monkeypatch, overlaps_oasis=False)
        with pytest.raises(FileNotFoundError):
            VisitEncoder(
                backbone="vit_tiny_8",
                volume_shape=TINY,
                pretrained="test_tiny",
                checkpoint_path=tmp_path / "absent.pt",
            )


def _write_tiny_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, overlaps_oasis: bool
) -> Path:
    """Register a test-only checkpoint and save weights that genuinely fit it.

    Registered through the same registry production uses, so the test exercises the
    real lookup. Monkeypatched, so it disappears with the test rather than becoming a
    fake entry a config could select.
    """
    from oasis_dl.models import encoder as encoder_module

    monkeypatch.setitem(
        encoder_module._CHECKPOINTS,
        "test_tiny",
        encoder_module.PretrainedCheckpoint(
            backbone="vit_tiny_8",
            source="test fixture",
            license_terms="n/a",
            pretraining_corpus=("OASIS-3",) if overlaps_oasis else ("IXI",),
            overlaps_oasis=overlaps_oasis,
        ),
    )
    donor = VisitEncoder(backbone="vit_tiny_8", volume_shape=TINY)
    path = tmp_path / "tiny.pt"
    torch.save(donor.body.state_dict(), path)
    return path


@pytest.mark.slow
def test_the_production_encoder_runs_at_its_configured_shape() -> None:
    """ADR-0011's encoder at ADR-0010's input shape, which is what every arm will use.

    Slow-marked rather than skipped: `vit_base_16` at 96^3 is over a second per forward
    pass on CPU, so it would stop `make test` being run at all. It still has to work,
    so `make test-all` and CI's nightly path keep it honest.
    """
    encoder = VisitEncoder(backbone="vit_base_16", embedding_dim=512, volume_shape=(96, 96, 96))
    encoder.eval()
    with torch.no_grad():
        out = encoder(torch.randn(1, 1, 96, 96, 96))
    assert out.shape == (1, 512)
