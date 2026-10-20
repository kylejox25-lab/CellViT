"""Masked CNN leakage, architecture dispatch, and channel-mask accounting."""

import pytest
import torch

from cellvit.evaluate import ReconstructionMetrics, evaluate_reconstruction
from cellvit.models import MAEConfig, build_model, model_from_checkpoint


def config(kind):
    return MAEConfig(
        model_type=kind, image_size=16, patch_size=8, in_channels=2,
        encoder_dim=16, encoder_depth=1, encoder_heads=2,
        decoder_dim=8, decoder_depth=1, decoder_heads=2, diffusion_steps=10,
    )


@pytest.mark.parametrize("kind", ["mae", "ca_mae", "diffmae", "cnn_mae"])
def test_factory_and_checkpoint_round_trip(kind):
    model = build_model(config(kind)).eval()
    state = {
        "format": "cellvit-masked-autoencoder-v2",
        "model_config": model.config.to_dict(), "model": model.state_dict(),
    }
    restored = model_from_checkpoint(state).eval()
    image = torch.rand(1, 2, 16, 16)
    with torch.no_grad():
        assert torch.equal(model.encode(image), restored.encode(image))
    if kind == "mae":
        state["format"] = "cellvit-spatial-mae-v1"
        del state["model_config"]["model_type"]
        del state["model_config"]["diffusion_steps"]
        assert model_from_checkpoint(state).config.model_type == "mae"


def test_cnn_hidden_pixels_never_enter_prediction_and_gradients_flow():
    model = build_model(config("cnn_mae"))
    image = torch.rand(2, 2, 16, 16)
    first = model(image, generator=torch.Generator().manual_seed(19))
    patches = model.patchify(image).clone()
    patches[first["mask"].bool()] = 100
    changed = model(
        model.unpatchify(patches), generator=torch.Generator().manual_seed(19),
    )
    assert torch.equal(first["prediction"], changed["prediction"])
    assert first["mask"].sum() == 6
    expected = (first["prediction"] - first["target"]).square()[first["mask"].bool()].mean()
    assert torch.allclose(first["loss"], expected)
    first["loss"].backward()
    assert torch.isfinite(model.patch_embed.weight.grad).all()
    assert model.patch_embed.weight.grad.abs().sum() > 0
    assert torch.isfinite(model.decoder_pred.weight.grad).all()
    assert model.encode(image).shape == (2, 16)
    assert not torch.allclose(model.encode(image[:1]), model.encode(image[1:]))


def test_channel_mask_metrics_weight_by_actual_hidden_pixels():
    target = torch.zeros(1, 2, 2, 1)
    prediction = torch.tensor([[[[1.0], [100.0]], [[3.0], [2.0]]]])
    mask = torch.tensor([[[1, 0], [1, 1]]])
    metrics = ReconstructionMetrics()
    metrics.update(prediction, target, mask)
    result = metrics.compute()
    assert result["masked_mse"] == pytest.approx(14 / 3)
    assert result["per_channel_mse"] == pytest.approx([5, 4])
    assert result["per_channel_masked_patches"] == [2, 1]
    assert result["mask_unit"] == "channel_patch"
    assert result["masked_patches"] == 3


def test_channel_with_no_hidden_tokens_reports_missing_channel_score():
    metrics = ReconstructionMetrics()
    metrics.update(torch.ones(1, 2, 2, 1), torch.zeros(1, 2, 2, 1),
                   torch.tensor([[[1, 0], [1, 0]]]))
    result = metrics.compute()
    assert result["per_channel_mse"] == [1, None]
    assert result["masked_mse"] == 1


def test_diffusion_evaluation_records_its_protocol():
    model = build_model(config("diffmae"))
    loader = [{"image": torch.rand(1, 2, 16, 16), "well_id": ["a"]}]
    report = evaluate_reconstruction(model, loader, torch.device("cpu"), log_every=0)
    assert report["reconstruction_protocol"] == "diffmae_sampled_t_denoising"
    report = evaluate_reconstruction(
        model, loader, torch.device("cpu"), log_every=0, diffusion_sampling_steps=2,
    )
    assert report["reconstruction_protocol"] == "diffmae_ddim_2"
