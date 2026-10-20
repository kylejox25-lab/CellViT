"""DiffMAE with clean visible conditioning and noisy masked-patch queries.

Based on Wei et al., ICCV 2023, https://arxiv.org/abs/2304.03283. The decoder
uses cross-attention without masked-token self-attention and predicts x_0.
Project adaptations: a cosine noise schedule and, when norm_pix_loss is set,
diffusion in the same per-patch, per-channel standardized target space used by
the existing MAE. Thus reconstruction scores retain the shared target contract;
this is an adaptation of DiffMAE, rather than an exact paper configuration.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .mae import MAEConfig, MaskedAutoencoder, TransformerBlock


class _CrossAttentionBlock(nn.Module):
    """Noisy masked queries read clean visible tokens; no query self-attention."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim)
        self.key_value = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

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
        return queries + self.mlp(self.norm(queries))


class DiffMAE(MaskedAutoencoder):
    """ViT encoder and top-down cross-attention diffusion decoder.

    Training samples a timestep and noise once per image. Embedding extraction
    uses the complete clean image through the encoder, with no decoder or noise.
    ``reconstruct`` performs actual inpainting from pure Gaussian masked noise;
    its results, rather than random-timestep training predictions, are suitable
    for comparing reconstruction quality with ordinary masked autoencoders.
    """

    def __init__(self, config: MAEConfig | None = None) -> None:
        nn.Module.__init__(self)
        self.config = config or MAEConfig(model_type="diffmae")
        cfg = self.config
        self.num_patches = (cfg.image_size // cfg.patch_size) ** 2
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
        self.noise_embed = nn.Linear(self.patch_dim, cfg.decoder_dim)
        self.decoder_pos = nn.Parameter(torch.zeros(1, self.num_patches, cfg.decoder_dim))
        self.decoder_blocks = nn.ModuleList(
            _CrossAttentionBlock(cfg.decoder_dim, cfg.decoder_heads, cfg.mlp_ratio)
            for _ in range(cfg.decoder_depth)
        )
        self.time_embed = nn.Sequential(
            nn.Linear(cfg.decoder_dim, cfg.decoder_dim), nn.SiLU(),
            nn.Linear(cfg.decoder_dim, cfg.decoder_dim),
        )
        self.decoder_norm = nn.LayerNorm(cfg.decoder_dim)
        self.decoder_pred = nn.Linear(cfg.decoder_dim, self.patch_dim)
        # Improved-DDPM cosine cumulative noise schedule, clipped to beta<=.999.
        steps = torch.arange(cfg.diffusion_steps + 1, dtype=torch.float64)
        alpha_bar = torch.cos(((steps / cfg.diffusion_steps + 0.008) / 1.008) * math.pi / 2).square()
        beta = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(0, 0.999)
        self.register_buffer("alphas_cumprod", (1 - beta).cumprod(0).float())
        self._initialize_diffusion_weights()

    def _initialize_diffusion_weights(self) -> None:
        nn.init.xavier_uniform_(self.patch_embed.weight.flatten(1))
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.normal_(self.encoder_pos, std=0.02)
        nn.init.normal_(self.decoder_pos, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _visible_context(
        self, image: Tensor, generator: torch.Generator | None,
    ) -> tuple[list[Tensor], Tensor, Tensor]:
        tokens = self._tokens(image)
        batch, count, width = tokens.shape
        visible_count = max(1, int(count * (1 - self.config.mask_ratio)))
        shuffled = torch.rand(batch, count, device=image.device, generator=generator).argsort(1)
        ids_keep, ids_mask = shuffled[:, :visible_count], shuffled[:, visible_count:]
        visible = tokens.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, width))
        states = []
        for block in self.encoder_blocks:
            visible = block(visible)
            states.append(self.encoder_norm(visible))
        # Cross-attention reads progressively earlier encoder layers, as in the
        # paper's top-down connections. The first decoder reads the final layer.
        decoder_depth = self.config.decoder_depth
        encoder_depth = self.config.encoder_depth
        indices = torch.linspace(encoder_depth - 1, 0, decoder_depth).round().int().tolist()
        memory = [self.decoder_embed(states[index]) for index in indices]
        mask = torch.zeros(batch, count, device=image.device, dtype=torch.float32)
        mask.scatter_(1, ids_mask, 1)
        return memory, ids_mask, mask

    def _timestep_embedding(self, timestep: Tensor) -> Tensor:
        dim = self.config.decoder_dim
        half = math.ceil(dim / 2)
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = timestep.float().unsqueeze(1) * frequencies.unsqueeze(0)
        return torch.cat((angles.cos(), angles.sin()), dim=-1)[:, :dim]

    def _decode(
        self, noisy_masked: Tensor, timestep: Tensor, memory: list[Tensor], ids_mask: Tensor,
    ) -> Tensor:
        batch = noisy_masked.shape[0]
        positions = self.decoder_pos.expand(batch, -1, -1).gather(
            1, ids_mask.unsqueeze(-1).expand(-1, -1, self.config.decoder_dim)
        )
        time = self.time_embed(self._timestep_embedding(timestep)).unsqueeze(1)
        queries = self.noise_embed(noisy_masked) + positions + time
        for block, context in zip(self.decoder_blocks, memory):
            queries = block(queries, context)
        return self.decoder_pred(self.decoder_norm(queries)).float()

    @staticmethod
    def _result(predicted_masked: Tensor, target: Tensor, ids_mask: Tensor, mask: Tensor) -> dict[str, Tensor]:
        batch, _, channels, pixels = target.shape
        masked_target = target.flatten(2).gather(
            1, ids_mask.unsqueeze(-1).expand(-1, -1, channels * pixels)
        )
        loss = (predicted_masked - masked_target).square().mean()
        # Visible pixels are observed conditions, copied unchanged for display.
        prediction = target.flatten(2).scatter(
            1, ids_mask.unsqueeze(-1).expand(-1, -1, channels * pixels), predicted_masked
        ).reshape_as(target)
        return {"loss": loss, "prediction": prediction, "mask": mask, "target": target}

    def forward(self, image: Tensor, *, generator: torch.Generator | None = None) -> dict[str, Tensor]:
        target = self.reconstruction_target(image)
        memory, ids_mask, mask = self._visible_context(image, generator)
        masked_target = target.flatten(2).gather(
            1, ids_mask.unsqueeze(-1).expand(-1, -1, self.patch_dim)
        )
        timestep = torch.randint(
            self.config.diffusion_steps, (image.shape[0],), device=image.device, generator=generator
        )
        alpha = self.alphas_cumprod[timestep].reshape(-1, 1, 1)
        noise = torch.randn(masked_target.shape, device=image.device, generator=generator)
        noisy_masked = alpha.sqrt() * masked_target + (1 - alpha).sqrt() * noise
        predicted_masked = self._decode(noisy_masked, timestep, memory, ids_mask)
        result = self._result(predicted_masked, target, ids_mask, mask)
        result.update(timestep=timestep, noisy_masked=noisy_masked, ids_mask=ids_mask)
        return result

    @torch.no_grad()
    def reconstruct(
        self, image: Tensor, *, generator: torch.Generator | None = None, sampling_steps: int = 50,
    ) -> dict[str, Tensor]:
        """DDIM (eta=0) inpainting from noise with the training noise schedule.

        sampling_steps controls inference cost, not the training timestep count.
        Reconstruction remains stochastic through the mask and initial noise;
        supplying a seeded generator makes a reconstruction reproducible.
        """
        if not 1 <= sampling_steps <= self.config.diffusion_steps:
            raise ValueError("sampling_steps must be between 1 and diffusion_steps")
        target = self.reconstruction_target(image)
        memory, ids_mask, mask = self._visible_context(image, generator)
        sample = torch.randn(
            image.shape[0], ids_mask.shape[1], self.patch_dim,
            device=image.device, generator=generator,
        )
        schedule = torch.linspace(
            self.config.diffusion_steps - 1, 0, sampling_steps, device=image.device,
        ).round().long()
        for index, step in enumerate(schedule):
            timestep = step.expand(image.shape[0])
            prediction = self._decode(sample, timestep, memory, ids_mask)
            if index + 1 == len(schedule):
                sample = prediction
            else:
                alpha = self.alphas_cumprod[step]
                previous_alpha = self.alphas_cumprod[schedule[index + 1]]
                epsilon = (sample - alpha.sqrt() * prediction) / (1 - alpha).sqrt()
                sample = previous_alpha.sqrt() * prediction + (1 - previous_alpha).sqrt() * epsilon
        result = self._result(sample, target, ids_mask, mask)
        result["sampling_steps"] = torch.tensor(sampling_steps, device=image.device)
        return result
