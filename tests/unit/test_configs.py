"""The configs compose, and the arms differ only where they are meant to.

The root config says it out loud: "If a comparison needs more than one override to
express, the configs are hiding an uncontrolled difference between the arms." This
module is that sentence as a test. It is cheap — no model is built, nothing is trained —
and it catches the class of mistake that produces a plausible, publishable, wrong
number.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

ABLATION_ARMS = ["baseline_single_scan", "proposed_a_longitudinal", "proposed_b_anatomy"]
"""The arms of the two core ablations. The SOTA comparison arms are out of scope here."""


@pytest.fixture(autouse=True)
def _checkpoint_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Satisfy the checkpoint interpolation without a 450 MB download.

    The config only has to *resolve*; nothing here loads weights.
    """
    monkeypatch.setenv("BRAINIAC_CHECKPOINT", str(tmp_path / "brainiac.pt"))


def _compose(*overrides: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=list(overrides))


def _encoder(cfg: DictConfig) -> dict[str, object]:
    resolved = OmegaConf.to_container(cfg.model.encoder, resolve=True)
    assert isinstance(resolved, dict)
    return {str(k): v for k, v in resolved.items()}


class TestComposition:
    @pytest.mark.parametrize("arm", [*ABLATION_ARMS, "sota_3dsctf", "sota_longformer"])
    def test_every_experiment_composes(self, arm: str) -> None:
        assert _compose(f"experiment={arm}").experiment.name == arm

    @pytest.mark.parametrize(
        "encoder", ["vit_base_brainiac", "vit_base_random", "resnet18", "vit_tiny_smoke"]
    )
    def test_every_encoder_composes(self, encoder: str) -> None:
        cfg = _compose(f"encoder={encoder}")
        assert _encoder(cfg)["name"] == encoder


class TestTheArmsDifferOnlyWhereTheyShould:
    def test_all_three_arms_share_one_encoder(self) -> None:
        """Ablation 1 varies the temporal module. If the encoder varies too, it measures
        two changes at once and neither can be attributed."""
        encoders = {arm: _encoder(_compose(f"experiment={arm}")) for arm in ABLATION_ARMS}
        baseline = encoders[ABLATION_ARMS[0]]
        for arm, encoder in encoders.items():
            assert encoder == baseline, f"{arm} does not share the baseline's encoder"

    def test_the_temporal_module_is_the_only_difference_between_b1_and_pa(self) -> None:
        """Ablation 1, stated as a diff."""
        b1 = OmegaConf.to_container(_compose("experiment=baseline_single_scan").model, resolve=True)
        pa = OmegaConf.to_container(
            _compose("experiment=proposed_a_longitudinal").model, resolve=True
        )
        assert isinstance(b1, dict) and isinstance(pa, dict)
        differing = {k for k in set(b1) | set(pa) if b1.get(k) != pa.get(k)}
        assert differing == {"name", "temporal"}, f"unexpected difference: {differing}"

    def test_the_anatomy_head_is_the_only_difference_between_pa_and_pb(self) -> None:
        """Ablation 2, stated as a diff. The anatomy head is discarded at inference
        (ADR-0007), so anything else differing here leaks into the test-time pathway."""
        pa = OmegaConf.to_container(
            _compose("experiment=proposed_a_longitudinal").model, resolve=True
        )
        pb = OmegaConf.to_container(_compose("experiment=proposed_b_anatomy").model, resolve=True)
        assert isinstance(pa, dict) and isinstance(pb, dict)
        differing = {k for k in set(pa) | set(pb) if pa.get(k) != pb.get(k)}
        # Not even `name`: PB composes from PA and adds a head, so the two arms run the
        # identical model config and are told apart by `experiment.arm`.
        assert differing == {"anatomy_head"}, f"unexpected difference: {differing}"


class TestEncoderWiring:
    def test_the_temporal_width_follows_the_encoder(self) -> None:
        """Pinned across encoders, because a hardcoded width that happens to match
        today's encoder survives a swap and silently truncates."""
        for encoder in ("vit_base_brainiac", "resnet18", "vit_tiny_smoke"):
            cfg = _compose("experiment=proposed_a_longitudinal", f"encoder={encoder}")
            assert cfg.model.temporal.embedding_dim == _encoder(cfg)["embedding_dim"]

    def test_the_encoder_volume_shape_follows_the_data_config(self) -> None:
        """A ViT is built around one input shape. This is what stops the two drifting."""
        cfg = _compose("experiment=proposed_a_longitudinal")
        assert list(_encoder(cfg)["volume_shape"]) == list(cfg.data.volume_shape)  # type: ignore[arg-type]

    def test_the_random_init_control_matches_its_pretrained_twin_everywhere_else(self) -> None:
        """The encoder-initialization question is only answerable if nothing else moves."""
        pretrained = _encoder(_compose("encoder=vit_base_brainiac"))
        random_init = _encoder(_compose("encoder=vit_base_random"))
        differing = {
            k for k in set(pretrained) | set(random_init) if pretrained.get(k) != random_init.get(k)
        }
        assert differing == {"name", "pretrained", "checkpoint_path"}

    def test_the_default_encoder_is_the_one_the_adr_chose(self) -> None:
        assert _encoder(_compose())["backbone"] == "vit_base_16"
