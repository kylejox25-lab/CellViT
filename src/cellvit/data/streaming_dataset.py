"""Mosaic Streaming reader for well-level RxRx3-core MDS samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from streaming import StreamingDataLoader, StreamingDataset

from .decode import decode_well


class RxRx3StreamingDataset(StreamingDataset):
    """Return four channel-aligned crops and metadata per well.

    ``batch_size`` must match StreamingDataLoader's per-device batch size so
    Streaming can partition and resume samples deterministically.
    """

    def __init__(
        self,
        *,
        local: str | Path,
        split: str,
        batch_size: int,
        remote: str | None = None,
        shuffle_seed: int = 2026,
        allow_incomplete: bool = False,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")
        if remote is None:
            manifest = json.loads((Path(local) / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("format") != "rxrx3-core-well-mds-v1":
                raise ValueError("Unexpected MDS schema version")
            if not manifest.get("complete") and not allow_incomplete:
                raise ValueError("MDS conversion is incomplete; use it only for a smoke check")
            if manifest["counts"][split] < 1:
                raise ValueError(f"MDS split {split} is empty")
        super().__init__(
            local=str(local),
            remote=remote,
            split=split,
            batch_size=batch_size,
            shuffle=(split == "train"),
            shuffle_seed=shuffle_seed,
            validate_hash="sha256",
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        return {
            "image": decode_well(sample),
            "well_id": sample["well_id"],
            "experiment_name": sample["experiment_name"],
            "plate": sample["plate"],
            "address": sample["address"],
            "perturbation_type": sample["perturbation_type"],
            "well_type_label": sample["well_type_label"],
            "gene": sample["gene"],
            "treatment": sample["treatment"],
            "concentration": sample["concentration"],
        }


def make_dataloader(
    *,
    mds_root: str | Path,
    split: str,
    batch_size: int,
    num_workers: int = 4,
    shuffle_seed: int = 2026,
    allow_incomplete: bool = False,
) -> StreamingDataLoader:
    """Build a loader whose state_dict can be saved with a training checkpoint."""
    dataset = RxRx3StreamingDataset(
        local=mds_root,
        split=split,
        batch_size=batch_size,
        shuffle_seed=shuffle_seed,
        allow_incomplete=allow_incomplete,
    )
    return StreamingDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
