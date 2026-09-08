"""Image/mask augmentations shared by ACA training datasets."""

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torch
from torchvision import transforms


_NORMALIZE = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def normalize(image: Image.Image, mask: Image.Image | None = None):
    image_tensor = _NORMALIZE(image)
    if mask is None:
        return image_tensor
    mask_tensor = torch.from_numpy(np.asarray(mask).copy()).long()
    return image_tensor, mask_tensor


def resize(
    image: Image.Image,
    mask: Image.Image,
    instance_mask: Image.Image | None = None,
    ratio_range: tuple[float, float] = (0.5, 2.0),
):
    """Randomly resize all supplied views with identical geometry."""

    lower, upper = ratio_range
    if not 0 < lower <= upper:
        raise ValueError("ratio_range must satisfy 0 < lower <= upper")
    width, height = image.size
    long_side = random.randint(int(max(height, width) * lower), int(max(height, width) * upper))
    if height > width:
        out_height = long_side
        out_width = int(width * long_side / height + 0.5)
    else:
        out_width = long_side
        out_height = int(height * long_side / width + 0.5)
    size = (max(1, out_width), max(1, out_height))
    image = image.resize(size, Image.Resampling.BILINEAR)
    mask = mask.resize(size, Image.Resampling.NEAREST)
    if instance_mask is None:
        return image, mask
    return image, mask, instance_mask.resize(size, Image.Resampling.NEAREST)


def crop(
    image: Image.Image,
    mask: Image.Image,
    size: int,
    ignore_value: int = 255,
    instance_mask: Image.Image | None = None,
):
    """Pad if necessary, then take one aligned random square crop."""

    if size <= 0:
        raise ValueError("crop size must be positive")
    width, height = image.size
    pad_width = max(0, size - width)
    pad_height = max(0, size - height)
    image = ImageOps.expand(image, border=(0, 0, pad_width, pad_height), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, pad_width, pad_height), fill=ignore_value)
    if instance_mask is not None:
        instance_mask = ImageOps.expand(instance_mask, border=(0, 0, pad_width, pad_height), fill=0)

    width, height = image.size
    left = random.randint(0, width - size)
    top = random.randint(0, height - size)
    box = (left, top, left + size, top + size)
    image = image.crop(box)
    mask = mask.crop(box)
    if instance_mask is None:
        return image, mask
    return image, mask, instance_mask.crop(box)


def hflip(
    image: Image.Image,
    mask: Image.Image,
    p: float = 0.5,
    instance_mask: Image.Image | None = None,
):
    """Horizontally flip all supplied views with probability ``p``."""

    if not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1]")
    if random.random() < p:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if instance_mask is not None:
            instance_mask = instance_mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if instance_mask is None:
        return image, mask
    return image, mask, instance_mask


def blur(image: Image.Image, p: float = 0.5) -> Image.Image:
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1]")
    if random.random() < p:
        image = image.filter(ImageFilter.GaussianBlur(radius=float(np.random.uniform(0.1, 2.0))))
    return image


def obtain_cutmix_box(
    image_size: int,
    p: float = 0.5,
    size_min: float = 0.02,
    size_max: float = 0.4,
    ratio_min: float = 0.3,
    ratio_max: float = 1 / 0.3,
) -> torch.Tensor:
    """Sample the binary CutMix rectangle used by UniMatch-V2."""

    mask = torch.zeros((image_size, image_size), dtype=torch.float32)
    if random.random() > p:
        return mask
    area = float(np.random.uniform(size_min, size_max)) * image_size * image_size
    for _ in range(100):
        ratio = float(np.random.uniform(ratio_min, ratio_max))
        width = int(np.sqrt(area / ratio))
        height = int(np.sqrt(area * ratio))
        left = int(np.random.randint(0, image_size))
        top = int(np.random.randint(0, image_size))
        if left + width <= image_size and top + height <= image_size:
            mask[top : top + height, left : left + width] = 1
            return mask
    return mask
