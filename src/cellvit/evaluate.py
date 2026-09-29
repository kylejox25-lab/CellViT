"""Evaluate masked pixel reconstruction in the same target space as MAE loss."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from itertools import islice
from pathlib import Path

import torch

from cellvit.models import MAEConfig, MaskedAutoencoder


class ReconstructionMetrics:
    """Accumulate hidden-patch errors; visible pixels never improve the score."""

    def __init__(self) -> None:
        self.squared_error: torch.Tensor | None = None
        self.zero_error: torch.Tensor | None = None
        self.masked_patches = 0

    @torch.no_grad()
    def update(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
        prediction = prediction.float().reshape_as(target)
        if mask.shape != target.shape[:2]:
            raise ValueError("Mask and target patch dimensions do not match")
        if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
            raise FloatingPointError("Non-finite reconstruction prediction or target")
        hidden = mask.bool()
        # [hidden patches, channels, pixels] -> sum of patch MSE per channel.
        error = (prediction[hidden] - target[hidden]).square().mean(dim=-1)
        baseline = target[hidden].square().mean(dim=-1)
        error = error.sum(dim=0, dtype=torch.float64).cpu()
        baseline = baseline.sum(dim=0, dtype=torch.float64).cpu()
        if self.squared_error is None:
            self.squared_error, self.zero_error = error, baseline
        else:
            self.squared_error += error
            self.zero_error += baseline
        self.masked_patches += int(hidden.sum())

    def compute(self) -> dict:
        if not self.masked_patches:
            raise ValueError("No masked patches were evaluated")
        channel_mse = self.squared_error / self.masked_patches
        channel_baseline = self.zero_error / self.masked_patches
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
            "masked_patches": int(self.masked_patches),
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
) -> dict:
    """Fixed masks given the same seed, loader order, and batch size.

    max_batches=None evaluates the complete split. Partial results are marked.
    No plotting or optimizer updates take place here.
    """
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive")
    model.eval()
    metrics = ReconstructionMetrics()
    wells = batches = 0
    seen: set[str] = set()
    selected = loader if max_batches is None else islice(loader, max_batches)
    use_amp = amp and device.type == "cuda"
    for index, batch in enumerate(selected):
        image = batch["image"].to(device, non_blocking=True)
        ids = batch["well_id"]
        if len(ids) != len(image) or len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("Evaluation contains missing or duplicate well IDs")
        seen.update(ids)
        generator = torch.Generator(device=device).manual_seed(seed + index)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            result = model(image, generator=generator)
        metrics.update(result["prediction"], model.reconstruction_target(image), result["mask"])
        wells += len(image)
        batches += 1
    return {
        **metrics.compute(), "wells": wells, "batches": batches,
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
    loader = make_dataloader(
        mds_root=args.mds_root, split=args.split, batch_size=batch_size,
        num_workers=args.num_workers,
    )
    report = evaluate_reconstruction(
        model, loader, device, seed=seed, amp=state["run_config"]["amp"] and not args.no_amp,
        max_batches=args.max_batches,
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
