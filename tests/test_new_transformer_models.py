"""Behavioral checks for channel-aware masking and diffusion conditioning."""

import pytest
import torch

from cellvit.models import ChannelAgnosticMAE, DiffMAE, MAEConfig


def tiny_config(model_type: str, **kwargs) -> MAEConfig:
    options = dict(
        model_type=model_type, image_size=16, patch_size=8, in_channels=3,
        encoder_dim=32, encoder_depth=2, encoder_heads=4,
        decoder_dim=16, decoder_depth=2, decoder_heads=4,
        mask_ratio=0.75, diffusion_steps=20,
    )
    options.update(kwargs)
    return MAEConfig(**options)


def test_channel_mask_loss_and_all_decoder_gradients() -> None:
    model = ChannelAgnosticMAE(tiny_config("ca_mae"))
    image = torch.rand(2, 3, 16, 16)
    result = model(image, generator=torch.Generator().manual_seed(13))
    repeated = model(image, generator=torch.Generator().manual_seed(13))
    assert result["prediction"].shape == result["target"].shape == (2, 4, 3, 64)
    assert result["mask"].shape == (2, 4, 3)
    assert torch.equal(result["mask"], repeated["mask"])
    assert torch.all(result["mask"].sum((1, 2)) == 9)
    # Independent channel-patch masks must not collapse to one spatial mask.
    assert torch.any(result["mask"][:, :, 0] != result["mask"][:, :, 1])
    error = (result["prediction"] - result["target"]).square().mean(-1)
    assert torch.allclose(result["loss"], error[result["mask"].bool()].mean())
    result["loss"].backward()
    assert model.patch_embed.weight.grad is not None
    assert torch.isfinite(model.patch_embed.weight.grad).all()
    for decoder in model.channel_decoders:
        assert decoder.pred.weight.grad is not None
        assert decoder.key_value.weight.grad is not None
        assert decoder.key_value.weight.grad.abs().sum() > 0
    assert not model.encoder_pos.requires_grad


def test_channel_permutation_and_variable_channel_encoding() -> None:
    model = ChannelAgnosticMAE(tiny_config("ca_mae")).eval()
    image = torch.rand(2, 3, 16, 16)
    permutation = torch.tensor([2, 0, 1])
    with torch.no_grad():
        embedding = model.encode(image)
        permuted = model.encode(image[:, permutation])
        concat = model.encode(image, pooling="channel_concat").reshape(2, 3, 32)
        permuted_concat = model.encode(
            image[:, permutation], pooling="channel_concat"
        ).reshape(2, 3, 32)
        assert embedding.shape == (2, 32)
        assert torch.allclose(embedding, permuted, atol=2e-5, rtol=1e-5)
        assert torch.allclose(concat[:, permutation], permuted_concat, atol=2e-5, rtol=1e-5)
        assert model.encode(image[:, :1]).shape == (2, 32)
        assert model.encode(torch.rand(2, 5, 16, 16)).shape == (2, 32)
    assert torch.equal(model.unpatchify(model.patchify(image[:, :2])), image[:, :2])
    with pytest.raises(ValueError, match="pooling"):
        model.encode(image, pooling="unknown")
    with pytest.raises(ValueError, match="training channels"):
        model(image[:, :2])


def test_diffusion_randomness_target_loss_and_cross_attention_gradients() -> None:
    model = DiffMAE(tiny_config("diffmae"))
    image = torch.rand(2, 3, 16, 16)
    first = model(image, generator=torch.Generator().manual_seed(17))
    second = model(image, generator=torch.Generator().manual_seed(17))
    assert first["prediction"].shape == first["target"].shape == (2, 4, 3, 64)
    assert first["mask"].shape == (2, 4)
    assert torch.equal(first["mask"], second["mask"])
    assert torch.equal(first["timestep"], second["timestep"])
    assert torch.equal(first["noisy_masked"], second["noisy_masked"])
    assert torch.equal(first["prediction"], second["prediction"])
    assert torch.all(first["mask"].sum(1) == 3)
    assert 0 <= first["timestep"].min() <= first["timestep"].max() < 20
    error = (first["prediction"] - first["target"]).square().mean((-1, -2))
    assert torch.allclose(first["loss"], error[first["mask"].bool()].mean())
    assert torch.isfinite(first["loss"])
    first["loss"].backward()
    assert model.noise_embed.weight.grad is not None
    assert model.noise_embed.weight.grad.abs().sum() > 0
    assert model.patch_embed.weight.grad is not None
    assert model.patch_embed.weight.grad.abs().sum() > 0
    for block in model.decoder_blocks:
        assert block.key_value.weight.grad is not None
        assert torch.isfinite(block.key_value.weight.grad).all()
        assert block.key_value.weight.grad.abs().sum() > 0


def test_diffusion_decoder_uses_noise_and_clean_visible_condition() -> None:
    model = DiffMAE(tiny_config("diffmae")).eval()
    image = torch.rand(2, 3, 16, 16)
    with torch.no_grad():
        memory, ids_mask, _ = model._visible_context(image, torch.Generator().manual_seed(9))
        timestep = torch.tensor([10, 10])
        noisy = torch.randn(2, 3, model.patch_dim)
        original = model._decode(noisy, timestep, memory, ids_mask)
        changed_noise = model._decode(noisy + 0.2, timestep, memory, ids_mask)
        changed_memory = model._decode(
            noisy, timestep, [context + torch.randn_like(context) for context in memory], ids_mask
        )
        assert not torch.allclose(original, changed_noise)
        assert not torch.allclose(original, changed_memory)
        # Alter only hidden patches: the encoder must not observe their content.
        changed_patches = model.patchify(image).clone()
        changed_patches.scatter_(
            1, ids_mask[:, :, None, None].expand(-1, -1, 3, 64),
            torch.randn(2, 3, 3, 64),
        )
        changed_image = model.unpatchify(changed_patches)
        changed_context, changed_ids, _ = model._visible_context(
            changed_image, torch.Generator().manual_seed(9)
        )
        assert torch.equal(ids_mask, changed_ids)
        for before, after in zip(memory, changed_context):
            assert torch.equal(before, after)
        first_embedding = model.encode(image)
        assert torch.equal(first_embedding, model.encode(image))


def test_diffusion_sampling_starts_from_noise_and_is_reproducible() -> None:
    model = DiffMAE(tiny_config("diffmae")).eval()
    image = torch.rand(1, 3, 16, 16)
    first = model.reconstruct(image, generator=torch.Generator().manual_seed(5), sampling_steps=3)
    second = model.reconstruct(image, generator=torch.Generator().manual_seed(5), sampling_steps=3)
    assert torch.equal(first["prediction"], second["prediction"])
    assert torch.equal(first["mask"], second["mask"])
    assert torch.isfinite(first["loss"])
    assert not first["prediction"].requires_grad
    # Hidden ground truth is never an input to the inpainting sampler.
    patches = model.patchify(image).clone()
    patches[first["mask"].bool()] = 100
    changed = model.reconstruct(
        model.unpatchify(patches), generator=torch.Generator().manual_seed(5), sampling_steps=3
    )
    hidden = first["mask"].bool()
    assert torch.equal(first["prediction"][hidden], changed["prediction"][hidden])
    with pytest.raises(ValueError, match="sampling_steps"):
        model.reconstruct(image, sampling_steps=0)


@pytest.mark.parametrize("norm_pix_loss", [True, False])
def test_new_models_preserve_shared_reconstruction_target(norm_pix_loss: bool) -> None:
    image = torch.rand(1, 3, 16, 16)
    for cls, name in [(ChannelAgnosticMAE, "ca_mae"), (DiffMAE, "diffmae")]:
        model = cls(tiny_config(name, norm_pix_loss=norm_pix_loss))
        target = model.reconstruction_target(image)
        if norm_pix_loss:
            assert torch.allclose(target.mean(-1), torch.zeros_like(target[..., 0]), atol=1e-5)
        else:
            assert torch.equal(target, model.patchify(image))
