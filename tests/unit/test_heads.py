"""Output heads — ADR-0007.

The classification head runs everywhere. The anatomy head runs at training only and is
discarded afterwards, which is what makes ablation 2 clean: PA and PB have identical
inference pathways, so any difference comes from the representation the objective
shaped rather than from extra information at test time.
"""

from __future__ import annotations

import pytest
import torch

from oasis_dl.models.heads import AnatomyHead, ClassificationHead

DIM = 32
TARGETS = ("hippocampus_left_mm3", "hippocampus_right_mm3", "total_gray_matter_mm3")


class TestClassificationHead:
    def test_one_logit_per_sample(self) -> None:
        """One logit, not two: threshold selection is then a decision about one number."""
        head = ClassificationHead(embedding_dim=DIM).eval()
        with torch.no_grad():
            out = head(torch.randn(5, DIM))
        assert out.shape == (5,)

    def test_the_output_is_a_logit_not_a_probability(self) -> None:
        """Probabilities here would be squashed twice once the loss applies a sigmoid."""
        head = ClassificationHead(embedding_dim=DIM, dropout=0.0).eval()
        with torch.no_grad():
            out = head(torch.randn(200, DIM) * 50)
        assert out.min() < 0 and out.max() > 0
        assert out.abs().max() > 1.0

    def test_dropout_is_active_in_training_and_quiet_in_eval(self) -> None:
        head = ClassificationHead(embedding_dim=DIM, dropout=0.9)
        x = torch.randn(64, DIM)
        head.train()
        assert not torch.allclose(head(x), head(x))
        head.eval()
        with torch.no_grad():
            torch.testing.assert_close(head(x), head(x))

    def test_a_wrong_width_is_refused(self) -> None:
        head = ClassificationHead(embedding_dim=DIM).eval()
        with pytest.raises(ValueError, match=str(DIM)):
            head(torch.randn(4, DIM + 1))

    @pytest.mark.parametrize("dropout", [-0.1, 1.0, 1.5])
    def test_an_impossible_dropout_is_refused(self, dropout: float) -> None:
        with pytest.raises(ValueError):
            ClassificationHead(embedding_dim=DIM, dropout=dropout)


class TestAnatomyHead:
    def test_one_prediction_per_named_target(self) -> None:
        head = AnatomyHead(embedding_dim=DIM, target_names=TARGETS).eval()
        with torch.no_grad():
            out = head(torch.randn(4, DIM))
        assert out.shape == (4, len(TARGETS))

    def test_targets_must_be_named_not_counted(self) -> None:
        """Named, so that a reordered target vector is a loud error rather than a
        quietly wrong regression."""
        with pytest.raises(ValueError):
            AnatomyHead(embedding_dim=DIM, target_names=())

    def test_duplicate_target_names_are_refused(self) -> None:
        """A duplicate makes two output columns indistinguishable by name."""
        with pytest.raises(ValueError, match="duplicate"):
            AnatomyHead(embedding_dim=DIM, target_names=("a", "b", "a"))

    def test_the_target_order_is_readable(self) -> None:
        """Collation stacks targets in sorted key order; the head must agree, and the
        only way a caller can check is if the head says what it predicts."""
        head = AnatomyHead(embedding_dim=DIM, target_names=TARGETS)
        assert head.target_names == TARGETS

    def test_the_output_is_unbounded(self) -> None:
        """Standardized volumes are signed and roughly unit-scale; a bounded activation
        here would clip the tails where atrophy actually lives."""
        head = AnatomyHead(embedding_dim=DIM, target_names=TARGETS).eval()
        with torch.no_grad():
            out = head(torch.randn(500, DIM) * 20)
        assert out.min() < -1.0 and out.max() > 1.0
