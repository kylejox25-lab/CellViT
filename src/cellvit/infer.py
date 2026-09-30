"""Export one unnormalized encoder embedding for every well in an MDS split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from cellvit.data.streaming_dataset import make_dataloader
from cellvit.models import MAEConfig, MaskedAutoencoder


def export_embeddings(
    checkpoint: Path,
    mds_root: Path,
    split: str,
    output: Path,
    *,
    batch_size: int = 1,
    num_workers: int = 4,
) -> int:
    """Stream embeddings to Parquet and verify one row per expected well."""
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split: {split}")
    root, destination = mds_root.resolve(), output.resolve()
    if destination == root or root in destination.parents:
        raise ValueError("Embedding output must be outside the MDS directory")
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_suffix(output.suffix + ".incomplete")
    if temporary.exists():
        raise FileExistsError(temporary)
    manifest_bytes = (mds_root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("format") != "rxrx3-core-well-mds-v1" or not manifest.get("complete"):
        raise ValueError("Embedding export requires a complete MDS conversion")
    expected = manifest["counts"][split]
    if expected < 1:
        raise ValueError(f"No wells in split {split}")

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("format") != "cellvit-spatial-mae-v1":
        raise ValueError("Not a CellViT spatial MAE checkpoint")
    if state["manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError("Checkpoint and MDS manifest do not match")
    config = MAEConfig(**state["model_config"])
    model = MaskedAutoencoder(config)
    model.load_state_dict(state["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = state["run_config"]["amp"] and device.type == "cuda"
    model = model.to(device).eval()
    loader = make_dataloader(
        mds_root=mds_root, split=split, batch_size=batch_size,
        num_workers=num_workers,
    )
    schema = pa.schema([
        ("well_id", pa.string()),
        ("embedding", pa.list_(pa.float32(), config.encoder_dim)),
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with pq.ParquetWriter(temporary, schema=schema, compression="zstd") as writer:
        with torch.inference_mode():
            for batch in loader:
                image = batch["image"].to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16,
                    enabled=use_amp,
                ):
                    embedding = model.encode(image).float()
                if not torch.isfinite(embedding).all():
                    raise FloatingPointError("Encoder returned non-finite embeddings")
                ids = list(batch["well_id"])
                for well_id in ids:
                    if well_id in seen:
                        raise ValueError(f"Duplicate well_id in split: {well_id}")
                    seen.add(well_id)
                table = pa.Table.from_arrays([
                    pa.array(ids, type=pa.string()),
                    pa.array(embedding.cpu().tolist(), type=schema.field("embedding").type),
                ], schema=schema)
                writer.write_table(table)
    if len(seen) != expected:
        raise ValueError(f"Exported {len(seen)} wells; manifest expects {expected}")
    os.replace(temporary, output)
    return len(seen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mds-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    count = export_embeddings(
        args.checkpoint, args.mds_root, args.split, args.output,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    print(json.dumps({"split": args.split, "wells": count, "output": str(args.output)}))


if __name__ == "__main__":
    main()
