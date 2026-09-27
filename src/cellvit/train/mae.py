"""Train the spatial MAE baseline on complete RxRx3-core MDS wells."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from streaming import StreamingDataLoader

from cellvit.data.streaming_dataset import make_dataloader
from cellvit.models import MAEConfig, MaskedAutoencoder


@dataclass(frozen=True)
class TrainConfig:
    mds_root: str
    output: str
    epochs: int = 1
    batch_size: int = 1
    num_workers: int = 4
    accumulation_steps: int = 64
    learning_rate: float = 1.5e-4
    weight_decay: float = 0.05
    warmup_fraction: float = 0.05
    seed: int = 2026
    checkpoint_every: int = 500
    log_every: int = 50
    amp: bool = True

    def __post_init__(self) -> None:
        if any(value < 1 for value in (
            self.epochs, self.batch_size, self.accumulation_steps,
            self.checkpoint_every, self.log_every,
        )):
            raise ValueError("epochs, batch size, accumulation, and intervals must be positive")
        if self.num_workers < 0 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid worker count, learning rate, or weight decay")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1)")


def learning_rate(step: int, total_steps: int, peak: float, warmup_fraction: float) -> float:
    """Linear warmup followed by cosine decay, indexed by optimizer step."""
    warmup = int(total_steps * warmup_fraction)
    if warmup and step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return peak * 0.5 * (1 + math.cos(math.pi * progress))


def _manifest(root: Path) -> tuple[dict, str]:
    content = (root / "manifest.json").read_bytes()
    manifest = json.loads(content)
    if manifest.get("format") != "rxrx3-core-well-mds-v1" or not manifest.get("complete"):
        raise ValueError("Training requires a complete rxrx3-core-well-mds-v1 conversion")
    if manifest["counts"]["train"] < 1 or manifest["counts"]["val"] < 1:
        raise ValueError("Training and validation splits must both be populated")
    return manifest, hashlib.sha256(content).hexdigest()


def _save_checkpoint(path: Path, state: dict) -> None:
    """Replace a checkpoint only after the new file has been fully written."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _checkpoint_state(
    model: MaskedAutoencoder,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loader: StreamingDataLoader,
    run_config: TrainConfig,
    manifest_hash: str,
    *,
    epoch: int,
    batch_in_epoch: int,
    global_step: int,
    global_batch: int,
    total_steps: int,
) -> dict:
    return {
        "format": "cellvit-spatial-mae-v1",
        "model_config": model.config.to_dict(),
        "run_config": asdict(run_config),
        "manifest_sha256": manifest_hash,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "loader": loader.state_dict(),
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "global_step": global_step,
        "global_batch": global_batch,
        "total_steps": total_steps,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


@torch.no_grad()
def validate(
    model: MaskedAutoencoder, loader: StreamingDataLoader,
    device: torch.device, seed: int, amp: bool
) -> float:
    """Monitor reconstruction only; biological retrieval remains the selection metric."""
    model.eval()
    weighted_loss = 0.0
    wells = 0
    for index, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        generator = torch.Generator(device=device.type).manual_seed(seed + index)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"
        ):
            loss = model(image, generator=generator)["loss"]
        weighted_loss += float(loss) * len(image)
        wells += len(image)
    if wells == 0:
        raise RuntimeError("Validation loader produced no wells")
    return weighted_loss / wells


def train(
    run: TrainConfig,
    model_config: MAEConfig,
    *,
    resume: Path | None = None,
    max_steps: int | None = None,
) -> Path:
    """Run or resume one single-GPU baseline; return the latest checkpoint path."""
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU; check the server PyTorch installation")
    device = torch.device("cuda")
    root = Path(run.mds_root).resolve()
    output = Path(run.output).resolve()
    if output == root or root in output.parents:
        raise ValueError("Training output must be outside the MDS directory")
    manifest, manifest_hash = _manifest(root)
    if resume is None:
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {output}")

    random.seed(run.seed)
    np.random.seed(run.seed)
    torch.manual_seed(run.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(run.seed)
        torch.cuda.reset_peak_memory_stats()
    train_loader = make_dataloader(
        mds_root=root, split="train", batch_size=run.batch_size,
        num_workers=run.num_workers, shuffle_seed=run.seed,
    )
    val_loader = make_dataloader(
        mds_root=root, split="val", batch_size=run.batch_size,
        num_workers=run.num_workers, shuffle_seed=run.seed,
    )
    batches_per_epoch = len(train_loader)
    if batches_per_epoch < 1:
        raise RuntimeError("Training loader produced no batches")
    total_steps = run.epochs * math.ceil(batches_per_epoch / run.accumulation_steps)
    model = MaskedAutoencoder(model_config).to(device)
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if parameter.ndim == 1 or name in {"encoder_pos", "decoder_pos", "mask_token"}:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": run.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=run.learning_rate,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=run.amp and device.type == "cuda")
    checkpoint_path = output / "latest.pt"
    epoch = batch_in_epoch = global_step = global_batch = 0

    if resume is not None:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state.get("format") != "cellvit-spatial-mae-v1":
            raise ValueError("Not a CellViT spatial MAE checkpoint")
        if state["run_config"] != asdict(run) or state["model_config"] != model_config.to_dict():
            raise ValueError("Resume requires the same training and model configuration")
        if state["manifest_sha256"] != manifest_hash or state["total_steps"] != total_steps:
            raise ValueError("Dataset manifest or training length changed since checkpoint")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        train_loader.load_state_dict(state["loader"])
        epoch = state["epoch"]
        batch_in_epoch = state["batch_in_epoch"]
        global_step = state["global_step"]
        global_batch = state["global_batch"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    else:
        (output / "config.json").write_text(
            json.dumps({
                "training": asdict(run), "model": model_config.to_dict(),
                "manifest_sha256": manifest_hash,
                "device": str(device), "train_wells": manifest["counts"]["train"],
                "val_wells": manifest["counts"]["val"],
            }, indent=2), encoding="utf-8",
        )

    if global_step >= total_steps or (max_steps is not None and global_step >= max_steps):
        return resume if resume is not None else checkpoint_path
    optimizer.zero_grad(set_to_none=True)
    group_wells = 0
    group_loss = 0.0
    log_path = output / "metrics.jsonl"

    for current_epoch in range(epoch, run.epochs):
        model.train()
        epoch_started = time.perf_counter()
        epoch_wells = 0
        for batch_index, batch in enumerate(train_loader, start=batch_in_epoch):
            image = batch["image"].to(device, non_blocking=True)
            generator = torch.Generator(device=device.type).manual_seed(run.seed + global_batch)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16,
                enabled=run.amp and device.type == "cuda",
            ):
                loss = model(image, generator=generator)["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at batch {global_batch}")
            scaler.scale(loss * len(image)).backward()
            group_wells += len(image)
            group_loss += float(loss.detach()) * len(image)
            epoch_wells += len(image)
            global_batch += 1
            batch_in_epoch = batch_index + 1
            if batch_in_epoch % run.accumulation_steps and batch_in_epoch < batches_per_epoch:
                continue

            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(group_wells)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0, error_if_nonfinite=True
            )
            rate = learning_rate(
                global_step, total_steps, run.learning_rate, run.warmup_fraction
            )
            for group in optimizer.param_groups:
                group["lr"] = rate
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            record = {
                "step": global_step, "epoch": current_epoch,
                "batch_in_epoch": batch_in_epoch,
                "train_loss": group_loss / group_wells,
                "learning_rate": rate,
                "wells_per_second": epoch_wells / max(1e-6, time.perf_counter() - epoch_started),
                "peak_gpu_gb": (
                    torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None
                ),
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            if global_step % run.log_every == 0:
                print(json.dumps(record), flush=True)
            group_wells = 0
            group_loss = 0.0

            stop_now = max_steps is not None and global_step >= max_steps
            if global_step % run.checkpoint_every == 0 or (
                stop_now and batch_in_epoch < batches_per_epoch
            ):
                _save_checkpoint(checkpoint_path, _checkpoint_state(
                    model, optimizer, scaler, train_loader, run, manifest_hash,
                    epoch=current_epoch, batch_in_epoch=batch_in_epoch,
                    global_step=global_step, global_batch=global_batch,
                    total_steps=total_steps,
                ))
            if stop_now and batch_in_epoch < batches_per_epoch:
                return checkpoint_path

        if batch_in_epoch != batches_per_epoch:
            raise RuntimeError("Training loader ended before its reported batch count")
        val_loss = validate(model, val_loader, device, run.seed + 10_000, run.amp)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "step": global_step, "epoch": current_epoch,
                "val_reconstruction_loss": val_loss,
            }) + "\n")
        print(f"epoch={current_epoch + 1} validation_reconstruction_loss={val_loss:.6f}", flush=True)
        batch_in_epoch = 0
        _save_checkpoint(checkpoint_path, _checkpoint_state(
            model, optimizer, scaler, train_loader, run, manifest_hash,
            epoch=current_epoch + 1, batch_in_epoch=0,
            global_step=global_step, global_batch=global_batch,
            total_steps=total_steps,
        ))
        if max_steps is not None and global_step >= max_steps:
            return checkpoint_path
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mds-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--model-config", type=Path, help="Optional JSON MAEConfig overrides")
    parser.add_argument("--resume", type=Path, help="Checkpoint from this output directory")
    parser.add_argument("--max-steps", type=int, help="Stop after this many total optimizer steps")
    args = parser.parse_args()
    run = TrainConfig(
        mds_root=str(args.mds_root.resolve()), output=str(args.output.resolve()),
        epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers,
        accumulation_steps=args.accumulation_steps, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, warmup_fraction=args.warmup_fraction,
        seed=args.seed, checkpoint_every=args.checkpoint_every,
        log_every=args.log_every, amp=not args.no_amp,
    )
    model_config = MAEConfig(**json.loads(args.model_config.read_text(encoding="utf-8"))) if args.model_config else MAEConfig()
    print(train(run, model_config, resume=args.resume, max_steps=args.max_steps))


if __name__ == "__main__":
    main()
