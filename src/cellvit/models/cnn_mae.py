"""Dense convolutional masked autoencoder, trained without external weights."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .mae import MAEConfig, MaskedAutoencoder


class ConvResidualBlock(nn.Module):
    """Bottleneck residual convolution; GroupNorm works with micro-batch one."""

    def __init__(self, width: int) -> None:
        super().__init__()
        hidden = max(4, width // 4)
        self.layers = nn.Sequential(
            nn.GroupNorm(math.gcd(width, 32), width), nn.GELU(),
            nn.Conv2d(width, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, width, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return features + self.layers(features)


class CNNMaskedAutoencoder(MaskedAutoencoder):
    """Spatial patch masking, convolutional context, and pixel reconstruction.

    The image is masked BEFORE the first non-overlapping patch convolution.
    No skip connection exposes hidden pixels to the decoder. Dense convolution
    processes the entire grid, unlike the visible-token-only ViT encoder.
    ``encoder_depth`` and ``decoder_depth`` count residual convolution blocks;
    attention-head settings do not apply to this architecture.
    """

    def __init__(self, config: MAEConfig | None = None) -> None:
        nn.Module.__init__(self)
        self.config = config or MAEConfig(model_type="cnn_mae")
        cfg = self.config
        self.num_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.patch_dim = cfg.in_channels * cfg.patch_size**2
        self.patch_embed = nn.Conv2d(
            cfg.in_channels, cfg.encoder_dim, cfg.patch_size, stride=cfg.patch_size,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, cfg.encoder_dim, 1, 1))
        self.encoder_blocks = nn.Sequential(
            *(ConvResidualBlock(cfg.encoder_dim) for _ in range(cfg.encoder_depth))
        )
        # One group preserves per-channel spatial means for global pooling,
        # including tiny test models; instance normalization would erase them.
        self.encoder_norm = nn.GroupNorm(1, cfg.encoder_dim)
        self.decoder_embed = nn.Conv2d(cfg.encoder_dim, cfg.decoder_dim, 1)
        self.decoder_blocks = nn.Sequential(
            *(ConvResidualBlock(cfg.decoder_dim) for _ in range(cfg.decoder_depth))
        )
        self.decoder_pred = nn.ConvTranspose2d(
            cfg.decoder_dim, cfg.in_channels, cfg.patch_size, stride=cfg.patch_size,
        )
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.mask_token, std=0.02)

    def encode(self, image: Tensor) -> Tensor:
        """Global average pooling of the full, clean encoder feature map."""
        self._check_image(image)
        features = self.encoder_norm(self.encoder_blocks(self.patch_embed(image)))
        return features.mean(dim=(-1, -2))

    def forward(self, image: Tensor, *, generator: torch.Generator | None = None) -> dict[str, Tensor]:
        self._check_image(image)
        cfg = self.config
        batch = image.shape[0]
        grid = cfg.image_size // cfg.patch_size
        keep = max(1, int(self.num_patches * (1 - cfg.mask_ratio)))
        order = torch.rand(batch, self.num_patches, device=image.device, generator=generator).argsort(dim=1)
        mask = torch.ones(batch, self.num_patches, device=image.device)
        mask.scatter_(1, order[:, :keep], 0)
        visible_grid = (1 - mask).reshape(batch, 1, grid, grid)
        visible_pixels = visible_grid.repeat_interleave(cfg.patch_size, -2).repeat_interleave(cfg.patch_size, -1)
        features = self.patch_embed(image * visible_pixels)
        features = features * visible_grid + self.mask_token * (1 - visible_grid)
        features = self.encoder_norm(self.encoder_blocks(features))
        decoded = self.decoder_blocks(self.decoder_embed(features))
        prediction = self.patchify(self.decoder_pred(decoded))
        target = self.reconstruction_target(image)
        per_patch = (prediction.float() - target).square().mean(dim=(-1, -2))
        loss = (per_patch * mask).sum() / mask.sum()
        return {"loss": loss, "prediction": prediction, "target": target, "mask": mask}
