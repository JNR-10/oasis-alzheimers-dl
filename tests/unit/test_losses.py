"""The combined objective — ADR-0007.

``L_total = L_classification + lambda * L_anatomy``

The load-bearing property is that **lambda = 0 reduces exactly to the classification
loss**. PA and PB then share one implementation, and ablation 2 cannot drift through two
diverging copies of the objective. "Exactly" means bitwise: a 1e-9 difference would
compound over epochs into two arms that were never comparable.
"""

from __future__ import annotations

import pytest
import torch

from oasis_dl.training.losses import AnatomyTargetScaler, CombinedObjective

LOGITS = torch.tensor([2.0, -1.0, 0.5, -3.0])
LABELS = torch.tensor([1.0, 0.0, 1.0, 0.0])
PREDICTION = torch.tensor([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6], [0.7, -0.8]])
TARGET = torch.tensor([[0.0, 0.0], [0.5, 0.5], [-1.0, 1.0], [1.0, -1.0]])


class TestLambdaZeroReducesToClassification:
    def test_the_total_is_bitwise_the_classification_loss(self) -> None:
        """The guard ablation 2 rests on."""
        objective = CombinedObjective(anatomy_weight=0.0)
        out = objective(LOGITS, LABELS, PREDICTION, TARGET)
        assert out["total"].item() == out["classification"].item()

    def test_pa_and_pb_agree_when_lambda_is_zero(self) -> None:
        """PA supplies no anatomy tensors; PB supplies them with lambda 0. Same number."""
        pa = CombinedObjective(anatomy_weight=0.0)(LOGITS, LABELS)
        pb = CombinedObjective(anatomy_weight=0.0)(LOGITS, LABELS, PREDICTION, TARGET)
        assert pa["total"].item() == pb["total"].item()

    def test_a_non_finite_anatomy_target_cannot_reach_pa(self) -> None:
        """What the branch at lambda = 0 is actually for.

        `classification + 0.0 * anatomy` is exact while anatomy is finite, so the
        bitwise test above passes with or without the branch. It only earns its place
        here: one inf in a target makes `0.0 * inf` NaN, and PA — which supplies no
        anatomy at all — would be taken down by a defect in PB's targets.
        """
        poisoned = TARGET.clone()
        poisoned[0, 0] = float("inf")
        out = CombinedObjective(anatomy_weight=0.0)(LOGITS, LABELS, PREDICTION, poisoned)
        assert torch.isfinite(out["total"]).all()
        assert out["total"].item() == out["classification"].item()

    def test_a_nonzero_lambda_does_change_the_total(self) -> None:
        """Otherwise the test above would pass on an objective that ignores anatomy."""
        weighted = CombinedObjective(anatomy_weight=0.5)(LOGITS, LABELS, PREDICTION, TARGET)
        assert weighted["total"].item() != weighted["classification"].item()


class TestBothComponentsAreVisible:
    def test_all_three_terms_are_returned(self) -> None:
        """A total that improves while classification worsens is the auxiliary term
        taking over, and that is invisible if only the total is logged."""
        out = CombinedObjective(anatomy_weight=0.1)(LOGITS, LABELS, PREDICTION, TARGET)
        assert set(out) == {"total", "classification", "anatomy"}

    def test_the_total_is_the_weighted_sum_of_its_parts(self) -> None:
        out = CombinedObjective(anatomy_weight=0.3)(LOGITS, LABELS, PREDICTION, TARGET)
        assert out["total"].item() == pytest.approx(
            out["classification"].item() + 0.3 * out["anatomy"].item()
        )

    def test_anatomy_is_zero_when_no_targets_are_supplied(self) -> None:
        out = CombinedObjective(anatomy_weight=0.3)(LOGITS, LABELS)
        assert out["anatomy"].item() == 0.0

    def test_gradients_flow_through_both_terms(self) -> None:
        logits = LOGITS.clone().requires_grad_(True)
        prediction = PREDICTION.clone().requires_grad_(True)
        CombinedObjective(anatomy_weight=0.5)(logits, LABELS, prediction, TARGET)[
            "total"
        ].backward()
        assert logits.grad is not None and logits.grad.abs().sum() > 0
        assert prediction.grad is not None and prediction.grad.abs().sum() > 0


class TestPerSequenceAnatomyMask:
    """ADR-0009: not every session ships FreeSurfer output.

    A training subject without targets still contributes classification loss and is
    excluded from the auxiliary term alone. All-or-nothing masking would make whether
    the auxiliary objective runs depend on how the sampler filled the batch.
    """

    def test_masked_rows_do_not_contribute_to_the_anatomy_term(self) -> None:
        mask = torch.tensor([True, True, False, False])
        masked = CombinedObjective(anatomy_weight=1.0)(
            LOGITS, LABELS, PREDICTION, TARGET, anatomy_mask=mask
        )
        only_real = CombinedObjective(anatomy_weight=1.0)(
            LOGITS[:2], LABELS[:2], PREDICTION[:2], TARGET[:2]
        )
        assert masked["anatomy"].item() == pytest.approx(only_real["anatomy"].item())

    def test_garbage_in_a_masked_row_cannot_reach_the_loss(self) -> None:
        mask = torch.tensor([True, True, False, False])
        poisoned = TARGET.clone()
        poisoned[2:] = 1e9
        a = CombinedObjective(anatomy_weight=1.0)(
            LOGITS, LABELS, PREDICTION, TARGET, anatomy_mask=mask
        )
        b = CombinedObjective(anatomy_weight=1.0)(
            LOGITS, LABELS, PREDICTION, poisoned, anatomy_mask=mask
        )
        assert a["anatomy"].item() == pytest.approx(b["anatomy"].item())

    def test_an_all_masked_batch_contributes_no_anatomy_loss(self) -> None:
        """Not NaN. A batch where nobody has targets is normal, not an error."""
        out = CombinedObjective(anatomy_weight=1.0)(
            LOGITS, LABELS, PREDICTION, TARGET, anatomy_mask=torch.zeros(4, dtype=torch.bool)
        )
        assert out["anatomy"].item() == 0.0
        assert out["total"].item() == out["classification"].item()


class TestClassWeighting:
    def test_pos_weight_raises_the_cost_of_a_missed_impairment(self) -> None:
        plain = CombinedObjective()(LOGITS, LABELS)["classification"].item()
        weighted = CombinedObjective(pos_weight=3.0)(LOGITS, LABELS)["classification"].item()
        assert weighted > plain

    def test_pos_weight_of_one_matches_no_weighting(self) -> None:
        plain = CombinedObjective()(LOGITS, LABELS)["classification"].item()
        unit = CombinedObjective(pos_weight=1.0)(LOGITS, LABELS)["classification"].item()
        assert unit == pytest.approx(plain)

    def test_a_negative_weight_is_refused(self) -> None:
        with pytest.raises(ValueError):
            CombinedObjective(pos_weight=-1.0)

    def test_a_negative_lambda_is_refused(self) -> None:
        with pytest.raises(ValueError):
            CombinedObjective(anatomy_weight=-0.5)


class TestAnatomyTargetScaler:
    """Fitted on the training partition only.

    Fitting on the full cohort leaks test-subject anatomy into training through the
    normalization constants — a quiet leak that no subject-level split catches, because
    no row crosses a partition boundary.
    """

    def test_training_statistics_standardize_the_training_partition(self) -> None:
        train = torch.tensor([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
        scaled = AnatomyTargetScaler().fit(train).transform(train)
        assert scaled.mean(dim=0).abs().max() < 1e-5
        assert scaled.std(dim=0, unbiased=False).sub(1.0).abs().max() < 1e-5

    def test_fitting_on_the_full_cohort_gives_different_constants(self) -> None:
        """The leak this class exists to prevent, made visible."""
        train = torch.tensor([[1.0], [2.0], [3.0]])
        everything = torch.tensor([[1.0], [2.0], [3.0], [100.0]])
        assert not torch.allclose(
            AnatomyTargetScaler().fit(train).mean, AnatomyTargetScaler().fit(everything).mean
        )

    def test_test_rows_are_transformed_with_training_constants(self) -> None:
        train = torch.tensor([[0.0], [2.0]])
        scaler = AnatomyTargetScaler().fit(train)
        # mean 1, population std 1 -> a test row of 4 standardizes to 3.
        assert scaler.transform(torch.tensor([[4.0]])).item() == pytest.approx(3.0)

    def test_transforming_before_fitting_is_an_error(self) -> None:
        """Silently standardizing with zeros would scale every target to itself."""
        with pytest.raises(RuntimeError, match="fit"):
            AnatomyTargetScaler().transform(torch.tensor([[1.0]]))

    def test_a_constant_column_does_not_produce_infinities(self) -> None:
        """A target identical for every training subject has zero variance; dividing by
        it yields inf and the auxiliary loss becomes NaN for the whole run."""
        constant = torch.tensor([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        scaled = AnatomyTargetScaler().fit(constant).transform(constant)
        assert torch.isfinite(scaled).all()

    def test_the_round_trip_recovers_the_original(self) -> None:
        train = torch.tensor([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
        scaler = AnatomyTargetScaler().fit(train)
        torch.testing.assert_close(scaler.inverse_transform(scaler.transform(train)), train)

    def test_a_width_mismatch_is_refused(self) -> None:
        scaler = AnatomyTargetScaler().fit(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        with pytest.raises(ValueError):
            scaler.transform(torch.tensor([[1.0, 2.0, 3.0]]))
