"""Evaluate masked pixel reconstruction in the same target space as MAE loss."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from itertools import islice
from pathlib import Path

import torch

from cellvit.models import MAEConfig, MaskedAutoencoder


class ReconstructionMetrics:
    """Accumulate hidden-patch errors; visible pixels never improve the score."""

    def __init__(self) -> None:
        self.squared_error: torch.Tensor | None = None
        self.zero_error: torch.Tensor | None = None
        self.masked_patches: torch.Tensor | None = None

    @torch.no_grad()
    def update(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
        prediction = prediction.float().reshape_as(target)
        if mask.shape != target.shape[:2]:
            raise ValueError("Mask and target patch dimensions do not match")
        finite = torch.isfinite(prediction).all() & torch.isfinite(target).all()
        if not bool(finite):
            raise FloatingPointError("Non-finite reconstruction prediction or target")
        hidden = mask.bool()
        # Dense reductions avoid dynamic boolean indexing and its GPU syncs.
        error = (prediction - target).square().mean(dim=-1)
        baseline = target.square().mean(dim=-1)
        error = torch.where(hidden.unsqueeze(-1), error, 0).sum(dim=(0, 1), dtype=torch.float64)
        baseline = torch.where(hidden.unsqueeze(-1), baseline, 0).sum(dim=(0, 1), dtype=torch.float64)
        count = hidden.sum()
        if self.squared_error is None:
            self.squared_error, self.zero_error = error, baseline
            self.masked_patches = count
        else:
            self.squared_error += error
            self.zero_error += baseline
            self.masked_patches += count

    def compute(self) -> dict:
        count = 0 if self.masked_patches is None else int(self.masked_patches)
        if not count:
            raise ValueError("No masked patches were evaluated")
        # Only copy accumulated channel totals when reporting, not every batch.
        channel_mse = self.squared_error.cpu() / count
        channel_baseline = self.zero_error.cpu() / count
        mse, baseline = float(channel_mse.mean()), float(channel_baseline.mean())
        if not math.isfinite(mse) or not math.isfinite(baseline):
            raise FloatingPointError("Non-finite reconstruction error")

        def skill(error: float, zero_error: float) -> float | None:
            # A zero-energy target has no meaningful relative improvement.
            return 1.0 - error / zero_error if zero_error > 1e-12 else None

        result = {
            "masked_mse": mse,
            "masked_rmse": math.sqrt(mse),
            "zero_baseline_mse": baseline,
            "reconstruction_skill": skill(mse, baseline),
            "masked_patches": count,
            "per_channel_mse": channel_mse.tolist(),
            "per_channel_zero_baseline_mse": channel_baseline.tolist(),
            "per_channel_reconstruction_skill": [
                skill(error, zero) for error, zero in zip(channel_mse.tolist(), channel_baseline.tolist())
            ],
        }
        return result


@torch.no_grad()
def evaluate_reconstruction(
    model: MaskedAutoencoder,
    loader,
    device: torch.device,
    *,
    seed: int = 12026,
    amp: bool = False,
    max_batches: int | None = None,
    log_every: int = 50,
) -> dict:
    """Fixed masks given the same seed, loader order, and batch size.

    max_batches=None evaluates the complete split. Partial results are marked.
    No plotting or optimizer updates take place here.
    """
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive")
    if log_every < 0:
        raise ValueError("log_every must be nonnegative (0 disables progress)")
    model.eval()
    metrics = ReconstructionMetrics()
    wells = batches = 0
    seen: set[str] = set()
    selected = loader if max_batches is None else islice(loader, max_batches)
    use_amp = amp and device.type == "cuda"
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    if max_batches is not None and total_batches is not None:
        total_batches = min(total_batches, max_batches)
    if log_every:
        print(
            f"Evaluation starting: device={device}, AMP={use_amp}, "
            f"batches={total_batches if total_batches is not None else 'unknown'}. Waiting for data...",
            file=sys.stderr, flush=True,
        )
    started = last_finished = last_report = time.perf_counter()
    loader_wait = 0.0
    for index, batch in enumerate(selected):
        loader_wait += time.perf_counter() - last_finished
        image = batch["image"].to(device, non_blocking=True)
        ids = batch["well_id"]
        if len(ids) != len(image) or len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("Evaluation contains missing or duplicate well IDs")
        seen.update(ids)
        generator = torch.Generator(device=device).manual_seed(seed + index)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            result = model(image, generator=generator)
        metrics.update(result["prediction"], result["target"], result["mask"])
        wells += len(image)
        batches += 1
        if log_every and (
            batches == 1 or batches % log_every == 0 or time.perf_counter() - last_report >= 30
        ):
            summary = metrics.compute()
            elapsed = time.perf_counter() - started
            eta = max(0, total_batches - batches) * elapsed / batches if total_batches is not None else None
            print(
                f"Evaluation {batches}/{total_batches if total_batches is not None else '?'} batches, "
                f"wells={wells}, MSE={summary['masked_mse']:.6f}, "
                f"speed={wells / max(elapsed, 1e-6):.2f} wells/s, "
                f"elapsed={elapsed:.1f}s, ETA={f'{eta:.1f}s' if eta is not None else 'unknown'}",
                file=sys.stderr, flush=True,
            )
            last_report = time.perf_counter()
        last_finished = time.perf_counter()
    summary = metrics.compute()
    elapsed = time.perf_counter() - started
    if log_every:
        print(f"Evaluation complete: {wells} wells in {elapsed:.1f}s", file=sys.stderr, flush=True)
    return {
        **summary, "wells": wells, "batches": batches,
        "elapsed_seconds": elapsed, "wells_per_second": wells / max(elapsed, 1e-6),
        "loader_wait_seconds": loader_wait,
        "mask_seed": seed, "max_batches": max_batches,
        "mask_ratio": model.config.mask_ratio,
        "target_space": "patch_channel_standardized" if model.config.norm_pix_loss else "raw_0_1",
        "precision": "amp_fp16" if use_amp else "fp32",
    }


def main() -> None:
    from cellvit.data.streaming_dataset import make_dataloader

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mds-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, help="Default: checkpoint training batch size")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, help="Default: training seed + 10000")
    parser.add_argument("--max-batches", type=int, help="Optional partial evaluation")
    parser.add_argument("--log-every", type=int, default=50, help="Progress interval; 0 disables progress")
    parser.add_argument("--no-amp", action="store_true", help="Use FP32 instead of checkpoint precision")
    args = parser.parse_args()
    root, destination = args.mds_root.resolve(), args.output.resolve()
    if destination == root or root in destination.parents:
        raise ValueError("Evaluation output must be outside the MDS directory")
    temporary = args.output.with_suffix(args.output.suffix + ".incomplete")
    if args.output.exists() or temporary.exists():
        raise FileExistsError("Evaluation output or its incomplete file already exists")
    manifest_bytes = (args.mds_root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("format") != "rxrx3-core-well-mds-v1" or not manifest.get("complete"):
        raise ValueError("Evaluation requires a complete MDS conversion")
    if (args.max_batches is not None and args.max_batches < 1) or args.log_every < 0:
        raise ValueError("Invalid max-batches or log-every")
    if args.log_every:
        print(f"Loading checkpoint: {args.checkpoint}", file=sys.stderr, flush=True)
    # Load only trusted checkpoints produced by this project.
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state.get("format") != "cellvit-spatial-mae-v1":
        raise ValueError("Not a CellViT spatial MAE checkpoint")
    if state["manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError("Dataset manifest differs from the training checkpoint")
    batch_size = args.batch_size if args.batch_size is not None else state["run_config"]["batch_size"]
    seed = args.seed if args.seed is not None else state["run_config"]["seed"] + 10_000
    if batch_size < 1 or args.num_workers < 0:
        raise ValueError("Invalid batch size or worker count")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskedAutoencoder(MAEConfig(**state["model_config"]))
    model.load_state_dict(state["model"])
    model.to(device)
    if args.log_every:
        print(f"Preparing {args.split} loader with {args.num_workers} workers", file=sys.stderr, flush=True)
    loader = make_dataloader(
        mds_root=args.mds_root, split=args.split, batch_size=batch_size,
        num_workers=args.num_workers, shuffle_seed=state["run_config"]["seed"],
    )
    report = evaluate_reconstruction(
        model, loader, device, seed=seed, amp=state["run_config"]["amp"] and not args.no_amp,
        max_batches=args.max_batches, log_every=args.log_every,
    )
    expected = manifest["counts"][args.split]
    if args.max_batches is None and report["wells"] != expected:
        raise RuntimeError(f"Expected {expected} wells, evaluated {report['wells']}")
    report.update({
        "checkpoint": str(args.checkpoint.resolve()), "split": args.split,
        "step": state["global_step"], "batch_size": batch_size,
        "expected_wells": expected, "complete_split": report["wells"] == expected,
        "manifest_sha256": state["manifest_sha256"],
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
