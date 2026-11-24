"""The experimental arms — assembly, not components.

B1 and PA are the two sides of ablation 1, so the properties here are about what each
arm is *allowed to see*. Two of them fail silently in production:

- B1 reading more than the target visit would make ablation 1 compare two longitudinal
  models and report a smaller difference than exists.
- PA encoding padded positions would let batch composition change a real visit's
  embedding, because the ResNet backbone's BatchNorm takes statistics over whatever is
  in the batch.
"""

from __future__ import annotations

import pytest
import torch

from oasis_dl.data.dataset import SequenceBatch
from oasis_dl.models.arms import LongitudinalModel, SingleScanModel
from oasis_dl.models.encoder import VisitEncoder
from oasis_dl.models.heads import AnatomyHead, ClassificationHead
from oasis_dl.models.temporal import TemporalAttention

DIM = 64
SHAPE = (16, 16, 16)
TARGETS = ("hippocampus_left_mm3", "total_gray_matter_mm3")


def _encoder() -> VisitEncoder:
    """resnet10 deliberately: it carries BatchNorm, which is what makes the padding
    property observable at all."""
    return VisitEncoder(backbone="resnet10", embedding_dim=DIM, volume_shape=SHAPE)


def _batch(n_visits: int, batch_size: int = 1, seed: int = 0) -> SequenceBatch:
    generator = torch.Generator().manual_seed(seed)
    return SequenceBatch(
        scans=torch.randn(batch_size, n_visits, 1, *SHAPE, generator=generator),
        elapsed_days=torch.zeros(batch_size, n_visits),
        mask=torch.ones(batch_size, n_visits, dtype=torch.bool),
        labels=torch.ones(batch_size, dtype=torch.long),
        anatomy_targets=None,
        anatomy_mask=None,
        subject_ids=tuple(f"S{i}" for i in range(batch_size)),
    )


def _single_scan() -> SingleScanModel:
    return SingleScanModel(_encoder(), ClassificationHead(embedding_dim=DIM, dropout=0.0))


def _longitudinal(anatomy: bool = False) -> LongitudinalModel:
    return LongitudinalModel(
        encoder=_encoder(),
        temporal=TemporalAttention(embedding_dim=DIM, n_heads=4, n_layers=1, dropout=0.0),
        head=ClassificationHead(embedding_dim=DIM, dropout=0.0),
        anatomy_head=(AnatomyHead(embedding_dim=DIM, target_names=TARGETS) if anatomy else None),
    )


class TestSingleScanArm:
    def test_one_logit_per_sequence(self) -> None:
        model = _single_scan().eval()
        with torch.no_grad():
            out = model(_batch(n_visits=1, batch_size=3))
        assert out["logits"].shape == (3,)

    def test_earlier_visits_cannot_change_the_prediction(self) -> None:
        """B1 reads the target visit and nothing else.

        It is handed the same sequences as PA so that ablation 1 compares architectures
        on identical data. If any earlier visit leaked in, B1 would quietly become a
        longitudinal model and the measured effect of the temporal module would shrink.
        """
        model = _single_scan().eval()
        full = _batch(n_visits=3, seed=1)
        altered = SequenceBatch(**{**full.__dict__, "scans": full.scans.clone()})
        altered.scans[:, :-1] = torch.randn_like(altered.scans[:, :-1])

        with torch.no_grad():
            torch.testing.assert_close(model(full)["logits"], model(altered)["logits"])

    def test_the_target_visit_does_change_the_prediction(self) -> None:
        """Otherwise the test above would pass on a model that reads nothing at all."""
        model = _single_scan().eval()
        full = _batch(n_visits=3, seed=1)
        altered = SequenceBatch(**{**full.__dict__, "scans": full.scans.clone()})
        altered.scans[:, -1] = torch.randn_like(altered.scans[:, -1])

        with torch.no_grad():
            assert not torch.allclose(model(full)["logits"], model(altered)["logits"])

    def test_it_emits_no_anatomy(self) -> None:
        model = _single_scan().eval()
        with torch.no_grad():
            assert "anatomy" not in model(_batch(n_visits=1))


class TestLongitudinalArm:
    def test_one_logit_per_sequence(self) -> None:
        model = _longitudinal().eval()
        with torch.no_grad():
            out = model(_batch(n_visits=3, batch_size=2))
        assert out["logits"].shape == (2,)

    def test_a_single_visit_subject_still_predicts(self) -> None:
        """Acceptance criterion of #10. Most OASIS-3 subjects have few visits, and an
        arm that cannot score a one-visit sequence silently shrinks the cohort."""
        model = _longitudinal().eval()
        with torch.no_grad():
            out = model(_batch(n_visits=1, batch_size=2))
        assert out["logits"].shape == (2,)
        assert torch.isfinite(out["logits"]).all()

    def test_padding_cannot_change_a_real_visits_embedding(self) -> None:
        """The silent one.

        The encoder's BatchNorm takes statistics over whatever is in the batch. If
        padded positions are encoded along with real ones, zero volumes enter those
        statistics and a sequence's prediction depends on how long the *longest* other
        sequence in its batch happened to be. Train mode, because that is when
        BatchNorm uses batch statistics rather than running ones.
        """
        model = _longitudinal().train()
        real = _batch(n_visits=2, seed=3)
        padded = SequenceBatch(
            scans=torch.cat([torch.randn(1, 2, 1, *SHAPE), real.scans], dim=1),
            elapsed_days=torch.zeros(1, 4),
            mask=torch.tensor([[False, False, True, True]]),
            labels=real.labels,
            anatomy_targets=None,
            anatomy_mask=None,
            subject_ids=real.subject_ids,
        )
        torch.testing.assert_close(
            model(real)["logits"], model(padded)["logits"], rtol=1e-4, atol=1e-5
        )

    def test_the_encoder_sees_every_visit_as_gradient_signal(self) -> None:
        """Weight sharing's whole point: the encoder trains on all visits, not just the
        ones at the final position."""
        model = _longitudinal()
        batch = _batch(n_visits=3, batch_size=2)
        generator = torch.Generator().manual_seed(0)
        out = model(batch)["logits"]
        (out * torch.randn(out.shape, generator=generator)).sum().backward()
        starved = [
            name
            for name, parameter in model.encoder.named_parameters()
            if parameter.grad is None or not parameter.grad.any()
        ]
        assert not starved, f"encoder parameters received no gradient: {starved[:3]}"


class TestAnatomyHeadIsTrainingOnly:
    """ADR-0007. PA and PB must have identical inference pathways, or ablation 2
    measures extra information at test time rather than a shaped representation."""

    def test_the_anatomy_head_runs_in_training(self) -> None:
        model = _longitudinal(anatomy=True).train()
        out = model(_batch(n_visits=2))
        assert out["anatomy"].shape == (1, len(TARGETS))

    def test_the_anatomy_head_is_absent_at_inference(self) -> None:
        model = _longitudinal(anatomy=True).eval()
        with torch.no_grad():
            assert "anatomy" not in model(_batch(n_visits=2))

    def test_pa_never_emits_anatomy(self) -> None:
        model = _longitudinal(anatomy=False).train()
        assert "anatomy" not in model(_batch(n_visits=2))

    def test_pa_and_pb_produce_the_same_logits_at_inference(self) -> None:
        """The inference pathway is identical, so with identical weights the two arms
        are indistinguishable at test time by construction."""
        torch.manual_seed(0)
        pa = _longitudinal(anatomy=False).eval()
        torch.manual_seed(0)
        pb = _longitudinal(anatomy=True).eval()
        batch = _batch(n_visits=3, seed=5)
        with torch.no_grad():
            torch.testing.assert_close(pa(batch)["logits"], pb(batch)["logits"])


class TestShapeGuards:
    def test_a_batch_whose_scans_are_not_six_dimensional_is_refused(self) -> None:
        model = _single_scan().eval()
        broken = SequenceBatch(
            scans=torch.randn(1, 1, *SHAPE),
            elapsed_days=torch.zeros(1, 1),
            mask=torch.ones(1, 1, dtype=torch.bool),
            labels=torch.ones(1, dtype=torch.long),
            anatomy_targets=None,
            anatomy_mask=None,
            subject_ids=("S0",),
        )
        with pytest.raises(ValueError, match="batch, visits"):
            model(broken)
