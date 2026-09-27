"""Small CPU checks for the masking and embedding contracts."""

import pytest
import torch

from cellvit.models import MAEConfig, MaskedAutoencoder
from cellvit.train.mae import learning_rate


def tiny_model() -> MaskedAutoencoder:
    return MaskedAutoencoder(MAEConfig(
        image_size=32, patch_size=8, in_channels=6,
        encoder_dim=48, encoder_depth=2, encoder_heads=4,
        decoder_dim=24, decoder_depth=1, decoder_heads=4,
        mask_ratio=0.75,
    ))


def test_masking_loss_and_embedding_contract() -> None:
    model = tiny_model()
    image = torch.rand(2, 6, 32, 32)
    first = model(image, generator=torch.Generator().manual_seed(17))
    second = model(image, generator=torch.Generator().manual_seed(17))

    assert first["prediction"].shape == (2, 16, 6 * 8 * 8)
    assert first["mask"].shape == (2, 16)
    assert torch.equal(first["mask"], second["mask"])
    assert torch.all(first["mask"].sum(dim=1) == 12)
    assert torch.isfinite(first["loss"])
    first["loss"].backward()
    assert model.patch_embed.weight.grad is not None
    assert torch.isfinite(model.patch_embed.weight.grad).all()
    assert model.encode(image).shape == (2, 48)


def test_invalid_geometry_and_input_fail_early() -> None:
    with pytest.raises(ValueError, match="two spatial patches"):
        MAEConfig(image_size=16, patch_size=16)
    with pytest.raises(ValueError, match="Expected image"):
        tiny_model().encode(torch.rand(1, 3, 32, 32))


def test_learning_rate_warmup_then_decay() -> None:
    assert learning_rate(0, 100, 0.01, 0.1) == pytest.approx(0.001)
    assert learning_rate(9, 100, 0.01, 0.1) == pytest.approx(0.01)
    assert learning_rate(100, 100, 0.01, 0.1) == pytest.approx(0)
