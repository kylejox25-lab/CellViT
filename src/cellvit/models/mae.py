"""A spatial masked autoencoder for complete six-channel RxRx3-core wells."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cellvit.image_config import IMAGE_CHANNELS, IMAGE_SIZE, PATCH_SIZE


@dataclass(frozen=True)
class MAEConfig:
    """Architecture and reconstruction settings saved with every checkpoint."""

    image_size: int = IMAGE_SIZE
    patch_size: int = PATCH_SIZE
    in_channels: int = IMAGE_CHANNELS
    encoder_dim: int = 384
    encoder_depth: int = 12
    encoder_heads: int = 6
    decoder_dim: int = 192
    decoder_depth: int = 4
    decoder_heads: int = 6
    mlp_ratio: float = 4.0
    mask_ratio: float = 0.75
    norm_pix_loss: bool = True

    def __post_init__(self) -> None:
        if self.image_size < 1 or self.patch_size < 1 or self.image_size % self.patch_size:
            raise ValueError("image_size must be a positive multiple of patch_size")
        if (self.image_size // self.patch_size) ** 2 < 2:
            raise ValueError("MAE requires at least two spatial patches")
        if self.in_channels < 1 or not 0 < self.mask_ratio < 1:
            raise ValueError("in_channels must be positive and mask_ratio must be in (0, 1)")
        for dim, depth, heads in (
            (self.encoder_dim, self.encoder_depth, self.encoder_heads),
            (self.decoder_dim, self.decoder_depth, self.decoder_heads),
        ):
            if dim < 1 or depth < 1 or heads < 1 or dim % heads:
                raise ValueError("transformer dimensions must be positive and divisible by heads")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


class TransformerBlock(nn.Module):
    """Pre-normalized self-attention followed by a two-layer MLP."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(self.norm1(x))
        qkv = qkv.reshape(batch, tokens, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, tokens, dim))
        return x + self.mlp(self.norm2(x))


class MaskedAutoencoder(nn.Module):
    """ViT-S/16 encoder with a lighter decoder and masked-patch pixel loss.

    The encoder receives only visible spatial patches during training. At
    inference it receives every patch, and the mean of its patch tokens is the
    well embedding. No external weights are loaded or required.
    """

    def __init__(self, config: MAEConfig | None = None) -> None:
        super().__init__()
        self.config = config or MAEConfig()
        cfg = self.config
        grid = cfg.image_size // cfg.patch_size
        self.num_patches = grid * grid
        self.patch_dim = cfg.in_channels * cfg.patch_size**2

        self.patch_embed = nn.Conv2d(
            cfg.in_channels, cfg.encoder_dim, cfg.patch_size, stride=cfg.patch_size
        )
        self.encoder_pos = nn.Parameter(torch.zeros(1, self.num_patches, cfg.encoder_dim))
        self.encoder_blocks = nn.ModuleList(
            TransformerBlock(cfg.encoder_dim, cfg.encoder_heads, cfg.mlp_ratio)
            for _ in range(cfg.encoder_depth)
        )
        self.encoder_norm = nn.LayerNorm(cfg.encoder_dim)

        self.decoder_embed = nn.Linear(cfg.encoder_dim, cfg.decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.decoder_dim))
        self.decoder_pos = nn.Parameter(torch.zeros(1, self.num_patches, cfg.decoder_dim))
        self.decoder_blocks = nn.ModuleList(
            TransformerBlock(cfg.decoder_dim, cfg.decoder_heads, cfg.mlp_ratio)
            for _ in range(cfg.decoder_depth)
        )
        self.decoder_norm = nn.LayerNorm(cfg.decoder_dim)
        self.decoder_pred = nn.Linear(cfg.decoder_dim, self.patch_dim)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.xavier_uniform_(self.patch_embed.weight.flatten(1))
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.normal_(self.encoder_pos, std=0.02)
        nn.init.normal_(self.decoder_pos, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _check_image(self, image: Tensor) -> None:
        expected = (self.config.in_channels, self.config.image_size, self.config.image_size)
        if image.ndim != 4 or tuple(image.shape[1:]) != expected:
            raise ValueError(f"Expected image [batch, {expected}], got {tuple(image.shape)}")

    def patchify(self, image: Tensor) -> Tensor:
        """Return [batch, spatial_patch, channel, pixels_in_channel]."""
        self._check_image(image)
        batch, channels, _, _ = image.shape
        patch = self.config.patch_size
        grid = self.config.image_size // patch
        return (
            image.reshape(batch, channels, grid, patch, grid, patch)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch, self.num_patches, channels, patch * patch)
        )

    def _tokens(self, image: Tensor) -> Tensor:
        self._check_image(image)
        return self.patch_embed(image).flatten(2).transpose(1, 2) + self.encoder_pos

    def unpatchify(self, patches: Tensor) -> Tensor:
        """Invert patchify; input is [B, N, C, patch_size**2]."""
        cfg = self.config
        patch = cfg.patch_size
        grid = cfg.image_size // patch
        expected = (self.num_patches, cfg.in_channels, patch * patch)
        if patches.ndim != 4 or tuple(patches.shape[1:]) != expected:
            raise ValueError(f"Expected patches [batch, {expected}], got {tuple(patches.shape)}")
        return (
            patches.reshape(-1, grid, grid, cfg.in_channels, patch, patch)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(-1, cfg.in_channels, cfg.image_size, cfg.image_size)
        )

    def reconstruction_target(self, image: Tensor) -> Tensor:
        """Shared FP32 target for training, evaluation, and visualization."""
        target = self.patchify(image).float()
        if self.config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            variance = target.var(dim=-1, keepdim=True, unbiased=False)
            target = (target - mean) / (variance + 1e-6).sqrt()
        return target

    def encode(self, image: Tensor) -> Tensor:
        """Encode all patches and average them into one vector per well."""
        tokens = self._tokens(image)
        for block in self.encoder_blocks:
            tokens = block(tokens)
        return self.encoder_norm(tokens).mean(dim=1)

    def forward(self, image: Tensor, *, generator: torch.Generator | None = None) -> dict[str, Tensor]:
        """Return loss, prediction, mask, and the shared reconstruction target.

        mask=1 marks a hidden patch. Per-patch, per-channel target normalization
        prevents one bright channel from dominating the pixel loss.
        """
        cfg = self.config
        tokens = self._tokens(image)
        batch, count, width = tokens.shape
        visible_count = max(1, int(count * (1 - cfg.mask_ratio)))
        noise = torch.rand(batch, count, device=image.device, generator=generator)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :visible_count]
        visible = torch.gather(tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, width))
        for block in self.encoder_blocks:
            visible = block(visible)
        visible = self.encoder_norm(visible)

        decoded = self.decoder_embed(visible)
        missing = self.mask_token.expand(batch, count - visible_count, -1)
        decoded = torch.cat([decoded, missing], dim=1)
        decoded = torch.gather(
            decoded, 1, ids_restore.unsqueeze(-1).expand(-1, -1, cfg.decoder_dim)
        )
        decoded = decoded + self.decoder_pos
        for block in self.decoder_blocks:
            decoded = block(decoded)
        prediction = self.decoder_pred(self.decoder_norm(decoded))

        # Count masks in FP32: an FP16 sum can overflow for larger batches.
        mask = torch.ones(batch, count, device=image.device, dtype=torch.float32)
        mask[:, :visible_count] = 0
        mask = torch.gather(mask, 1, ids_restore)
        target = self.reconstruction_target(image)
        prediction_by_channel = prediction.float().reshape_as(target)
        per_patch = (prediction_by_channel - target).square().mean(dim=(-1, -2))
        loss = (per_patch * mask).sum() / mask.sum()
        return {"loss": loss, "prediction": prediction, "mask": mask, "target": target}
