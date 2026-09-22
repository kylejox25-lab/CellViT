import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cellvit.data.convert import convert_dataset
from cellvit.data.source import iter_wells, load_metadata, make_plate_splits, parse_image_key


def write_shard(path: Path, entries: list[tuple[str, bytes]]) -> None:
    table = pa.table(
        {
            "__key__": pa.array([key for key, _ in entries]),
            "jp2": pa.array(
                [{"bytes": data, "path": key.rsplit("/", 1)[-1] + ".jp2"} for key, data in entries],
                type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
            ),
        }
    )
    pq.write_table(table, path)


def well_entries(plate: int, address: str) -> list[tuple[str, bytes]]:
    return [
        (f"gene-001/Plate{plate}/{address}_s1_{channel}", bytes([channel]))
        for channel in range(1, 7)
    ]


def test_parse_image_key_and_cross_shard_grouping(tmp_path: Path) -> None:
    assert parse_image_key("compound-001/Plate10/AA15_s1_6").well_id == "compound-001_10_AA15"
    first = tmp_path / "train-00000.parquet"
    second = tmp_path / "train-00001.parquet"
    write_shard(first, well_entries(1, "A01")[:4])
    write_shard(second, well_entries(1, "A01")[4:] + well_entries(2, "A02"))

    wells = list(iter_wells([first, second], batch_size=2))
    assert [well.key.well_id for well in wells] == ["gene-001_1_A01", "gene-001_2_A02"]
    assert wells[0].channels == tuple(bytes([channel]) for channel in range(1, 7))


def test_missing_channel_fails(tmp_path: Path) -> None:
    shard = tmp_path / "train-00000.parquet"
    write_shard(shard, well_entries(1, "A01")[:5])
    with pytest.raises(ValueError, match="channels"):
        list(iter_wells([shard]))


def test_plate_splits_are_deterministic_and_disjoint() -> None:
    metadata = {
        f"gene-001_{plate}_A01": {"experiment_name": "gene-001", "plate": str(plate)}
        for plate in range(1, 10)
    }
    first = make_plate_splits(metadata, seed=17)
    assert first == make_plate_splits(metadata, seed=17)
    assert sorted(first.values()).count("train") == 7
    assert sorted(first.values()).count("val") == 1
    assert sorted(first.values()).count("test") == 1


def test_conversion_routes_wells_by_plate_and_keeps_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    data = source / "data"
    data.mkdir(parents=True)
    entries = sum((well_entries(plate, f"A0{plate}") for plate in range(1, 4)), [])
    shard = data / "train-00000.parquet"
    write_shard(shard, entries)
    before = shard.read_bytes()
    with (source / "metadata_rxrx3_core.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "well_id", "experiment_name", "plate", "address", "gene", "treatment",
                "SMILES", "concentration", "perturbation_type", "cell_type", "well_type_label",
            ],
        )
        writer.writeheader()
        for plate in range(1, 4):
            writer.writerow(
                {
                    "well_id": f"gene-001_{plate}_A0{plate}",
                    "experiment_name": "gene-001",
                    "plate": plate,
                    "address": f"A0{plate}",
                    "gene": "GENE1",
                    "treatment": "GENE1_guide_1",
                    "perturbation_type": "CRISPR",
                    "well_type_label": "Query guides",
                }
            )

    records: dict[str, list[dict]] = {}

    class FakeWriter:
        def __init__(self, *, out: str, **_kwargs: object) -> None:
            self.split = Path(out).name
            records[self.split] = []

        def __enter__(self) -> "FakeWriter":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def write(self, sample: dict) -> None:
            records[self.split].append(sample)

    output = tmp_path / "converted"
    manifest = convert_dataset(source, output, writer_factory=FakeWriter)
    assert manifest["counts"] == {"train": 1, "val": 1, "test": 1}
    assert output.joinpath("manifest.json").exists()
    assert shard.read_bytes() == before
    assert {sample["well_id"] for samples in records.values() for sample in samples} == {
        f"gene-001_{plate}_A0{plate}" for plate in range(1, 4)
    }
    assert len(load_metadata(source / "metadata_rxrx3_core.csv")) == 3

    records.clear()
    smoke = convert_dataset(source, tmp_path / "smoke", max_wells=1, writer_factory=FakeWriter)
    assert smoke["complete"] is False
    assert smoke["total_wells"] == 1
    assert sum(len(samples) for samples in records.values()) == 1
