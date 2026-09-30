"""Analytic reconstruction scores and masked-region isolation."""

import pytest
import torch

from cellvit.evaluate import ReconstructionMetrics, evaluate_reconstruction
from cellvit.models import MAEConfig, MaskedAutoencoder


def score(prediction, target, mask):
    metrics = ReconstructionMetrics()
    metrics.update(prediction, target, mask)
    return metrics.compute()


def test_skill_perfect_zero_and_worse_than_baseline():
    target = torch.tensor([[[[-1.0, 1.0]], [[2.0, -2.0]]]])
    mask = torch.ones(1, 2)
    assert score(target, target, mask)["reconstruction_skill"] == pytest.approx(1.0)
    assert score(torch.zeros_like(target), target, mask)["reconstruction_skill"] == pytest.approx(0.0)
    assert score(-target, target, mask)["reconstruction_skill"] == pytest.approx(-3.0)


def test_visible_predictions_do_not_change_masked_score():
    target = torch.ones(1, 2, 1, 4)
    prediction = target.clone()
    prediction[:, 0] = 1000
    result = score(prediction, target, torch.tensor([[0, 1]]))
    assert result["masked_mse"] == 0
    assert result["masked_patches"] == 1


def test_metrics_weight_partial_batches_by_masked_patch_count():
    metrics = ReconstructionMetrics()
    metrics.update(torch.zeros(2, 1, 1, 4), torch.ones(2, 1, 1, 4), torch.ones(2, 1))
    metrics.update(torch.zeros(1, 1, 1, 4), torch.full((1, 1, 1, 4), 2.0), torch.ones(1, 1))
    assert metrics.compute()["masked_mse"] == pytest.approx(2.0)


def test_zero_energy_target_has_no_relative_score():
    result = score(torch.ones(1, 1, 1, 4), torch.zeros(1, 1, 1, 4), torch.ones(1, 1))
    assert result["masked_mse"] == 1
    assert result["reconstruction_skill"] is None
    assert result["per_channel_reconstruction_skill"] == [None]


def test_nonfinite_prediction_is_rejected():
    with pytest.raises(FloatingPointError):
        score(torch.full((1, 1, 1, 4), float("nan")), torch.zeros(1, 1, 1, 4), torch.ones(1, 1))


def test_evaluation_matches_forward_loss_and_reuses_target(capsys):
    model = MaskedAutoencoder(MAEConfig(
        image_size=16, patch_size=8, in_channels=2,
        encoder_dim=8, encoder_heads=2, encoder_depth=1,
        decoder_dim=8, decoder_heads=2, decoder_depth=1,
    ))
    image = torch.rand(2, 2, 16, 16)
    output = model(image, generator=torch.Generator().manual_seed(17))
    # Evaluation must call reconstruction_target only inside the forward pass.
    original_target = model.reconstruction_target
    calls = []

    def counted_target(image):
        calls.append(1)
        return original_target(image)

    model.reconstruction_target = counted_target
    report = evaluate_reconstruction(
        model, [{"image": image, "well_id": ["a", "b"]}], torch.device("cpu"),
        seed=17, log_every=1,
    )
    assert len(calls) == 1
    assert report["masked_mse"] == pytest.approx(float(output["loss"].detach()), rel=1e-5)
    assert report["wells"] == 2
    assert "Evaluation complete" in capsys.readouterr().err
