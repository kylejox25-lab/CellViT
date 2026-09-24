from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from cellvit.data.decode import decode_well


def test_decode_preserves_full_image_and_channel_order() -> None:
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

    decoded = decode_well(sample)
    assert decoded.shape == (6, 512, 512)
    assert decoded.dtype == np.float32
    assert decoded[:, 0, 0] == pytest.approx([i / 255 for i in range(1, 7)])
    assert decoded[0, 255, 255] == pytest.approx(1 / 255)
    assert decoded[0, 255, 256] == pytest.approx(11 / 255)
    assert decoded[0, 256, 255] == pytest.approx(21 / 255)
    assert decoded[0, 256, 256] == pytest.approx(31 / 255)
    assert decoded[5, 511, 511] == pytest.approx(36 / 255)
