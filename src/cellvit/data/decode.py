"""Decode one MDS well sample into a full six-channel image."""

from __future__ import annotations

from io import BytesIO
from typing import Mapping

import numpy as np
from PIL import Image

from cellvit.image_config import IMAGE_CHANNELS, IMAGE_SIZE


def decode_well(sample: Mapping[str, object]) -> np.ndarray:
    """Return the complete float32 image [6, 512, 512] in [0, 1]."""
    channels: list[np.ndarray] = []
    for index in range(1, IMAGE_CHANNELS + 1):
        data = sample[f"ch{index}"]
        if not isinstance(data, bytes):
            raise TypeError(f"ch{index} must be JP2 bytes")
        with Image.open(BytesIO(data)) as image:
            channel = np.asarray(image)
        if channel.shape != (IMAGE_SIZE, IMAGE_SIZE) or channel.dtype != np.uint8:
            raise ValueError(
                f"ch{index} has shape/dtype {channel.shape}/{channel.dtype}; "
                f"expected ({IMAGE_SIZE}, {IMAGE_SIZE})/uint8"
            )
        channels.append(channel)

    image = np.stack(channels, axis=0)
    return image.astype(np.float32) / 255.0
