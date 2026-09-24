"""Shared geometry for full-well images and the planned ViT encoder."""

IMAGE_CHANNELS = 6
IMAGE_SIZE = 512
PATCH_SIZE = 16
PATCHES_PER_IMAGE = (IMAGE_SIZE // PATCH_SIZE) ** 2
