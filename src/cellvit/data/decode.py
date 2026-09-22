"""Decode one MDS well sample into four 256×256 six-channel crops."""

from __future__ import annotations

from io import BytesIO
from typing import Mapping

import numpy as np
from PIL import Image


def decode_well(sample: Mapping[str, object]) -> np.ndarray:
    """Return float32 array [4, 6, 256, 256] in [0, 1].

    Crop order is upper-left, upper-right, lower-left, lower-right. No random
    augmentation occurs here, so resumable loaders cannot silently change crops.
    """
    channels: list[np.ndarray] = []
    for index in range(1, 7):
        data = sample[f"ch{index}"]
        if not isinstance(data, bytes):
            raise TypeError(f"ch{index} must be JP2 bytes")
        with Image.open(BytesIO(data)) as image:
            channel = np.asarray(image)
        if channel.shape != (512, 512) or channel.dtype != np.uint8:
            raise ValueError(
                f"ch{index} has shape/dtype {channel.shape}/{channel.dtype}; "
                "expected (512, 512)/uint8"
            )
        channels.append(channel)

    image = np.stack(channels, axis=0)
    crops = np.stack(
        [
            image[:, :256, :256],
            image[:, :256, 256:],
            image[:, 256:, :256],
            image[:, 256:, 256:],
        ],
        axis=0,
    )
    return crops.astype(np.float32) / 255.0
