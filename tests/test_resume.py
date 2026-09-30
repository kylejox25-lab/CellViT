"""Real MDS cursor regressions; run on the configured server environment."""

import shutil

import pytest
from streaming import MDSWriter, StreamingDataLoader, StreamingDataset

from cellvit.train.mae import _loader_progress


def make_loader(path):
    dataset = StreamingDataset(
        local=str(path), batch_size=2, shuffle=False, num_canonical_nodes=1,
    )
    return StreamingDataLoader(dataset, batch_size=2, num_workers=0)


@pytest.mark.parametrize("at_epoch_start", [False, True])
def test_resume_mid_epoch_and_at_next_epoch_start(tmp_path, at_epoch_start):
    original, restored = tmp_path / "original", tmp_path / "restored"
    with MDSWriter(out=str(original), columns={"value": "int"}) as writer:
        for value in range(7):
            writer.write({"value": value})
    # Separate cache paths allow two live StreamingDataset instances.
    shutil.copytree(original, restored)
    loader = make_loader(original)
    iterator = iter(loader)
    assert next(iterator)["value"].tolist() == [0, 1]
    if at_epoch_start:
        list(iterator)
    saved = loader.state_dict()
    assert saved["epoch"] == 0
    assert saved["sample_in_epoch"] == (7 if at_epoch_start else 2)

    resumed = make_loader(restored)
    resumed.load_state_dict(_loader_progress(
        saved, epoch=1 if at_epoch_start else 0,
        batch_in_epoch=0 if at_epoch_start else 1,
    ))
    values = [value for batch in resumed for value in batch["value"].tolist()]
    expected = list(range(7)) if at_epoch_start else list(range(2, 7))
    assert values == expected
    # Cursor conversion must not modify the previously saved checkpoint dict.
    assert saved["epoch"] == 0
