"""Read one real Mosaic Streaming batch and check its shape and metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cellvit.image_config import IMAGE_CHANNELS, IMAGE_SIZE

from .streaming_dataset import make_dataloader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mds-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    manifest = json.loads((args.mds_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["counts"][args.split] < 1:
        raise ValueError(f"No wells in {args.split}; choose a populated split")
    loader = make_dataloader(
        mds_root=args.mds_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        allow_incomplete=True,
    )
    batch = next(iter(loader))
    image = batch["image"]
    if image.ndim != 4 or tuple(image.shape[1:]) != (IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Unexpected image batch shape: {tuple(image.shape)}")
    if not image.isfinite().all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Images have non-finite or out-of-range values")
    if len(batch["well_id"]) != image.shape[0]:
        raise ValueError("well_id count does not match image batch")
    print(
        json.dumps(
            {
                "split": args.split,
                "batch_shape": list(image.shape),
                "first_well_id": batch["well_id"][0],
                "range": [float(image.min()), float(image.max())],
            }
        )
    )


if __name__ == "__main__":
    main()
