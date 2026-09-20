"""Convert read-only RxRx3-core Parquet shards to well-level Mosaic MDS."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable

from .source import Well, iter_wells, load_metadata, make_plate_splits


MDS_COLUMNS = {
    "well_id": "str",
    "experiment_name": "str",
    "plate": "int",
    "address": "str",
    "perturbation_type": "str",
    "well_type_label": "str",
    "gene": "str",
    "treatment": "str",
    "concentration": "float32",
    **{f"ch{index}": "bytes" for index in range(1, 7)},
}
SPLITS = ("train", "val", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample(well: Well, row: dict[str, str]) -> dict[str, Any]:
    key = well.key
    for field, value in (
        ("experiment_name", key.experiment_name),
        ("plate", str(key.plate)),
        ("address", key.address),
    ):
        if row[field] != value:
            raise ValueError(
                f"Metadata mismatch for {key.well_id}: {field} is {row[field]!r}, expected {value!r}"
            )
    sample: dict[str, Any] = {
        "well_id": key.well_id,
        "experiment_name": key.experiment_name,
        "plate": key.plate,
        "address": key.address,
        "perturbation_type": row["perturbation_type"],
        "well_type_label": row["well_type_label"],
        "gene": row["gene"],
        "treatment": row["treatment"],
        "concentration": float(row["concentration"]) if row["concentration"] else float("nan"),
    }
    sample.update({f"ch{index}": data for index, data in enumerate(well.channels, 1)})
    return sample


def convert_dataset(
    source_root: Path,
    output_root: Path,
    *,
    seed: int = 2026,
    max_wells: int | None = None,
    writer_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Make train/val/test MDS directories; never modify source or overwrite output.

    ``max_wells`` is for a small smoke conversion only. Such output is marked
    incomplete in its manifest and must not be used for actual model training.
    """
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("MDS output must be outside the read-only source dataset")
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")
    if max_wells is not None and max_wells < 1:
        raise ValueError("max_wells must be positive")
    stage_root = output_root.with_name(output_root.name + ".incomplete")
    if stage_root.exists():
        raise FileExistsError(f"Previous incomplete conversion exists: {stage_root}")

    metadata_path = source_root / "metadata_rxrx3_core.csv"
    shards = sorted((source_root / "data").glob("train-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No image Parquet shards under {source_root / 'data'}")
    metadata = load_metadata(metadata_path)
    assignments = make_plate_splits(metadata, seed=seed)

    if writer_factory is None:
        try:
            from streaming import MDSWriter
        except ImportError as exc:
            raise RuntimeError("Install mosaicml-streaming to create MDS shards") from exc
        writer_factory = MDSWriter

    stage_root.parent.mkdir(parents=True, exist_ok=True)
    stage_root.mkdir()
    counts: Counter[str] = Counter({split: 0 for split in SPLITS})
    total = 0
    try:
        with ExitStack() as stack:
            writers: dict[str, Any] = {}
            for well in iter_wells(shards):
                row = metadata.get(well.key.well_id)
                if row is None:
                    raise ValueError(f"Image well has no metadata: {well.key.well_id}")
                split = assignments[well.key.experiment_name, well.key.plate]
                if split not in writers:
                    writers[split] = stack.enter_context(
                        writer_factory(
                            out=str(stage_root / split),
                            columns=MDS_COLUMNS,
                            compression=None,  # JP2 payloads are already compressed.
                            hashes=["sha256"],
                            size_limit="96mb",
                        )
                    )
                writers[split].write(_sample(well, row))
                counts[split] += 1
                total += 1
                if max_wells is not None and total >= max_wells:
                    break

        complete = max_wells is None
        if complete and total != len(metadata):
            raise ValueError(
                f"Converted {total} wells but metadata contains {len(metadata)}; "
                "source shards may be incomplete"
            )
        split_rows = [
            {"experiment_name": experiment, "plate": plate, "split": split}
            for (experiment, plate), split in sorted(assignments.items())
        ]
        (stage_root / "plate_splits.json").write_text(
            json.dumps(split_rows, indent=2), encoding="utf-8"
        )
        manifest = {
            "format": "rxrx3-core-well-mds-v1",
            "complete": complete,
            "seed": seed,
            "counts": dict(counts),
            "total_wells": total,
            "expected_wells": len(metadata),
            "source_root": str(source_root),
            "metadata_sha256": _sha256(metadata_path),
            "source_shards": [{"name": path.name, "bytes": path.stat().st_size} for path in shards],
            "columns": MDS_COLUMNS,
        }
        (stage_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        stage_root.rename(output_root)
        return manifest
    except Exception:
        # Keep the clearly named partial directory for inspection. It is never
        # mistaken for a finished dataset and is not deleted automatically.
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Read-only RxRx3-core root")
    parser.add_argument("--output", type=Path, required=True, help="New writable MDS root")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-wells", type=int, default=None, help="Smoke conversion only")
    args = parser.parse_args()
    manifest = convert_dataset(args.source, args.output, seed=args.seed, max_wells=args.max_wells)
    print(json.dumps({"complete": manifest["complete"], "counts": manifest["counts"]}))


if __name__ == "__main__":
    main()
