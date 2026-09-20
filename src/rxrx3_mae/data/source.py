"""Read the original RxRx3-core Parquet files without changing them."""

from __future__ import annotations

import csv
import hashlib
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import pyarrow.parquet as pq


KEY_PATTERN = re.compile(
    r"^(?P<experiment>[^/]+)/Plate(?P<plate>\d+)/"
    r"(?P<address>[A-Z]{1,2}\d{2})_s(?P<site>\d+)_(?P<channel>[1-6])$"
)
CHANNELS = (1, 2, 3, 4, 5, 6)


@dataclass(frozen=True)
class ImageKey:
    well_id: str
    experiment_name: str
    plate: int
    address: str
    site: int
    channel: int


@dataclass(frozen=True)
class Well:
    key: ImageKey
    channels: tuple[bytes, ...]


def parse_image_key(value: str) -> ImageKey:
    """Map e.g. compound-001/Plate1/AA15_s1_6 to its metadata well_id."""
    match = KEY_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Unexpected RxRx3-core image key: {value!r}")
    experiment = match["experiment"]
    plate = int(match["plate"])
    address = match["address"]
    site = int(match["site"])
    if site != 1:
        raise ValueError(f"Expected one center-crop site (s1), got {value!r}")
    return ImageKey(
        well_id=f"{experiment}_{plate}_{address}",
        experiment_name=experiment,
        plate=plate,
        address=address,
        site=site,
        channel=int(match["channel"]),
    )


def load_metadata(path: Path) -> dict[str, dict[str, str]]:
    """Keep CSV values as strings; empty optional values remain empty."""
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            well_id = row.get("well_id", "")
            if not well_id or well_id in rows:
                raise ValueError(f"Missing or duplicate metadata well_id: {well_id!r}")
            rows[well_id] = {key: value or "" for key, value in row.items()}
    if not rows:
        raise ValueError(f"Metadata is empty: {path}")
    return rows


def make_plate_splits(
    metadata: Mapping[str, Mapping[str, str]], seed: int = 2026
) -> dict[tuple[str, int], str]:
    """Assign whole plates to train/val/test within each experiment."""
    plates: dict[str, set[int]] = defaultdict(set)
    for row in metadata.values():
        plates[row["experiment_name"]].add(int(row["plate"]))

    assignments: dict[tuple[str, int], str] = {}
    for experiment, values in sorted(plates.items()):
        ordered = sorted(values)
        if len(ordered) < 3:
            raise ValueError(f"Need at least three plates in {experiment}")
        digest = hashlib.sha256(f"{seed}:{experiment}".encode()).digest()
        random.Random(int.from_bytes(digest[:8], "big")).shuffle(ordered)
        n_test = max(1, round(len(ordered) * 0.1))
        n_val = max(1, round(len(ordered) * 0.1))
        if n_test + n_val >= len(ordered):
            raise ValueError(f"No training plates would remain in {experiment}")
        for plate in ordered[:n_test]:
            assignments[experiment, plate] = "test"
        for plate in ordered[n_test : n_test + n_val]:
            assignments[experiment, plate] = "val"
        for plate in ordered[n_test + n_val :]:
            assignments[experiment, plate] = "train"
    return assignments


def iter_wells(paths: Iterable[Path], batch_size: int = 1024) -> Iterator[Well]:
    """Stream six consecutive channel rows into one well, across shard boundaries.

    The source's row order is checked, not assumed silently. A broken or repeated
    channel group fails conversion instead of producing a mislabelled sample.
    """
    pending_key: ImageKey | None = None
    pending_channels: dict[int, bytes] = {}
    seen: set[str] = set()

    def finish() -> Well:
        assert pending_key is not None
        if set(pending_channels) != set(CHANNELS):
            raise ValueError(
                f"Well {pending_key.well_id} has channels {sorted(pending_channels)}, expected 1..6"
            )
        if pending_key.well_id in seen:
            raise ValueError(f"Well appears more than once: {pending_key.well_id}")
        seen.add(pending_key.well_id)
        return Well(
            key=pending_key,
            channels=tuple(pending_channels[index] for index in CHANNELS),
        )

    found_file = False
    for path in paths:
        found_file = True
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["__key__", "jp2"]):
            keys = batch.column(0).to_pylist()
            images = batch.column(1).to_pylist()
            for raw_key, image in zip(keys, images, strict=True):
                key = parse_image_key(raw_key)
                if pending_key is not None and key.well_id != pending_key.well_id:
                    yield finish()
                    pending_channels = {}
                pending_key = key
                if key.channel in pending_channels:
                    raise ValueError(f"Duplicate channel {key.channel} for {key.well_id}")
                image_bytes = image.get("bytes") if image else None
                if not image_bytes:
                    raise ValueError(f"Missing JP2 bytes for {raw_key}")
                pending_channels[key.channel] = image_bytes

    if not found_file:
        raise ValueError("No source Parquet shards were provided")
    if pending_key is not None:
        yield finish()
