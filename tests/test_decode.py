from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from cellvit.data.decode import decode_well


def test_decode_preserves_channel_and_crop_order() -> None:
    sample = {}
    for channel in range(1, 7):
        image = np.zeros((512, 512), dtype=np.uint8)
        image[:256, :256] = channel
        image[:256, 256:] = channel + 10
        image[256:, :256] = channel + 20
        image[256:, 256:] = channel + 30
        buffer = BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        sample[f"ch{channel}"] = buffer.getvalue()

    crops = decode_well(sample)
    assert crops.shape == (4, 6, 256, 256)
    assert crops.dtype == np.float32
    assert crops[:, 0, 0, 0] == pytest.approx([1 / 255, 11 / 255, 21 / 255, 31 / 255])
    assert crops[0, :, 0, 0] == pytest.approx([i / 255 for i in range(1, 7)])
