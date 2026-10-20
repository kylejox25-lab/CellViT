"""Channel-agnostic masked autoencoding for multi-channel microscopy.

The encoder follows Kraus et al. (CVPR 2024): each channel is tokenized with
the same projection and spatial sine/cosine positions, without channel IDs.
Channel-specific reconstruction decoders use cross-attention to the shared
visible memory followed by spatial self-attention within their own channel.
This avoids a quadratic self-attention map over all decoded channel tokens.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .mae import MAEConfig, MaskedAutoencoder, TransformerBlock


def _sincos_2d(grid: int, dim: int) -> Tensor:
    """Fixed positions repeated identically for every input channel."""
    frequencies = torch.arange(math.ceil(dim / 4), dtype=torch.float32)
    frequencies = 1.0 / (10000 ** (frequencies / max(len(frequencies), 1)))
    yy, xx = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    x = xx.flatten().float().unsqueeze(1) * frequencies
    y = yy.flatten().float().unsqueeze(1) * frequencies
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=-1)[None, :, :dim]


class _ChannelDecoder(nn.Module):
    """One stain's spatial queries attend to the complete visible memory."""

    def __init__(self, config: MAEConfig) -> None:
        super().__init__()
        dim = config.decoder_dim
        self.heads = config.decoder_heads
        self.head_dim = dim // self.heads
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim)
        self.key_value = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList(
            TransformerBlock(dim, config.decoder_heads, config.mlp_ratio)
            for _ in range(config.decoder_depth)
        )
        self.norm = nn.LayerNorm(dim)
        self.pred = nn.Linear(dim, config.patch_size**2)

    def forward(self, queries: Tensor, memory: Tensor) -> Tensor:
        batch, count, width = queries.shape
        q = self.query(self.query_norm(queries)).reshape(
            batch, count, self.heads, self.head_dim
        ).transpose(1, 2)
        kv = self.key_value(self.memory_norm(memory)).reshape(
            batch, memory.shape[1], 2, self.heads, self.head_dim
        )
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        queries = queries + self.proj(attended.transpose(1, 2).reshape(batch, count, width))
        for block in self.blocks:
            queries = block(queries)
        return self.pred(self.norm(queries))


class ChannelAgnosticMAE(MaskedAutoencoder):
    """Shared channel-independent encoder with a decoder for each training stain.

    ``encode`` accepts any positive number of channels. ``forward`` reconstructs
    the configured training channels, whose decoders retain their stain identity.
    The mask is [B, N, C], where 1 means a hidden channel-patch token. Random
    masking acts on the combined N*C token sequence, rather than forcing the
    same spatial mask on every channel.
    """

    def __init__(self, config: MAEConfig | None = None) -> None:
        nn.Module.__init__(self)
        self.config = config or MAEConfig(model_type="ca_mae")
        cfg = self.config
        grid = cfg.image_size // cfg.patch_size
        self.num_patches = grid * grid
        self.patch_dim = cfg.in_channels * cfg.patch_size**2
        self.patch_embed = nn.Linear(cfg.patch_size**2, cfg.encoder_dim)
        self.register_buffer("encoder_pos", _sincos_2d(grid, cfg.encoder_dim))
        self.encoder_blocks = nn.ModuleList(
            TransformerBlock(cfg.encoder_dim, cfg.encoder_heads, cfg.mlp_ratio)
            for _ in range(cfg.encoder_depth)
        )
        self.encoder_norm = nn.LayerNorm(cfg.encoder_dim)
        self.decoder_embed = nn.Linear(cfg.encoder_dim, cfg.decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.decoder_dim))
        self.register_buffer("decoder_pos", _sincos_2d(grid, cfg.decoder_dim))
        self.channel_decoders = nn.ModuleList(_ChannelDecoder(cfg) for _ in range(cfg.in_channels))
        self.apply(self._init_module)
        nn.init.normal_(self.mask_token, std=0.02)

    @staticmethod
    def _init_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _check_image(self, image: Tensor) -> None:
        size = self.config.image_size
        if image.ndim != 4 or image.shape[1] < 1 or tuple(image.shape[2:]) != (size, size):
            raise ValueError(
                f"Expected image [batch, positive_channels, {size}, {size}], got {tuple(image.shape)}"
            )

    def _tokens(self, image: Tensor) -> Tensor:
        patches = self.patchify(image)
        tokens = self.patch_embed(patches) + self.encoder_pos.unsqueeze(2)
        return tokens.flatten(1, 2)

    def unpatchify(self, patches: Tensor) -> Tensor:
        """Invert patchify, including variable channel counts used at inference."""
        cfg = self.config
        patch = cfg.patch_size
        grid = cfg.image_size // patch
        if (patches.ndim != 4 or patches.shape[1] != self.num_patches
                or patches.shape[2] < 1 or patches.shape[3] != patch * patch):
            raise ValueError("Expected patches [batch, spatial_patch, positive_channels, pixels]")
        batch, _, channels, _ = patches.shape
        return (
            patches.reshape(batch, grid, grid, channels, patch, patch)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch, channels, cfg.image_size, cfg.image_size)
        )

    def encode(self, image: Tensor, pooling: str = "mean") -> Tensor:
        """Return a permutation-invariant mean, or ordered per-channel means.

        ``channel_concat`` has C*encoder_dim dimensions and follows input channel
        order. It therefore changes size if the input channel count changes.
        Neither option applies masking or stochastic diffusion at inference.
        """
        if pooling not in {"mean", "channel_concat"}:
            raise ValueError("pooling must be 'mean' or 'channel_concat'")
        tokens = self._tokens(image)
        for block in self.encoder_blocks:
            tokens = block(tokens)
        tokens = self.encoder_norm(tokens)
        if pooling == "mean":
            return tokens.mean(dim=1)
        return tokens.reshape(image.shape[0], self.num_patches, image.shape[1], -1).mean(1).flatten(1)

    def forward(self, image: Tensor, *, generator: torch.Generator | None = None) -> dict[str, Tensor]:
        self._check_image(image)
        cfg = self.config
        if image.shape[1] != cfg.in_channels:
            raise ValueError(f"Reconstruction requires {cfg.in_channels} training channels")
        tokens = self._tokens(image)
        batch, count, width = tokens.shape
        visible_count = max(1, int(count * (1 - cfg.mask_ratio)))
        ids_shuffle = torch.rand(batch, count, device=image.device, generator=generator).argsort(1)
        ids_restore = ids_shuffle.argsort(1)
        ids_keep = ids_shuffle[:, :visible_count]
        visible = tokens.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, width))
        for block in self.encoder_blocks:
            visible = block(visible)
        memory = self.decoder_embed(self.encoder_norm(visible))
        missing = self.mask_token.expand(batch, count - visible_count, -1)
        queries = torch.cat((memory, missing), dim=1).gather(
            1, ids_restore.unsqueeze(-1).expand(-1, -1, cfg.decoder_dim)
        )
        queries = queries.reshape(batch, self.num_patches, cfg.in_channels, cfg.decoder_dim)
        queries = queries + self.decoder_pos.unsqueeze(2)
        # Each decoder's self-attention operates on N spatial tokens, never N*C.
        prediction = torch.stack([
            decoder(queries[:, :, channel], memory)
            for channel, decoder in enumerate(self.channel_decoders)
        ], dim=2).float()
        mask = torch.ones(batch, count, device=image.device, dtype=torch.float32)
        mask[:, :visible_count] = 0
        mask = mask.gather(1, ids_restore).reshape(batch, self.num_patches, cfg.in_channels)
        target = self.reconstruction_target(image)
        error = (prediction - target).square().mean(-1)
        loss = (error * mask).sum() / mask.sum()
        return {"loss": loss, "prediction": prediction, "mask": mask, "target": target}
