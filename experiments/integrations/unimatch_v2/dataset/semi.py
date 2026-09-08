"""ACA datasets used by the C-PGS UniMatch-V2 integration."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.transform import blur, crop, hflip, normalize, obtain_cutmix_box, resize


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility
        return torch.load(path, map_location="cpu")


def _read_manifest(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            fields = raw_line.strip().split()
            if not fields:
                continue
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_number}: expected '<image> <label>'")
            entries.append((fields[0].replace("\\", "/"), fields[1].replace("\\", "/")))
    if not entries:
        raise ValueError(f"manifest is empty: {path}")
    return entries


class ACASemiDataset(Dataset):
    """Labeled, unlabeled, or validation view of the ACA dataset.

    The style-prior cache is aligned with the unlabeled manifest. Historical
    caches are lists of 18-element tensors; a mapping keyed by the manifest's
    image path is also accepted for safer future preprocessing pipelines.
    """

    MODES = {"train_l", "train_u", "val"}

    def __init__(
        self,
        root: str | Path,
        manifest: str | Path,
        mode: str,
        *,
        crop_size: int | None = None,
        num_classes: int = 18,
        prior_path: str | Path | None = None,
        nsample: int | None = None,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unsupported dataset mode: {mode}")
        if mode != "val" and (crop_size is None or crop_size <= 0):
            raise ValueError("a positive crop_size is required for training")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        self.root = Path(root).expanduser().resolve()
        self.manifest = Path(manifest).expanduser().resolve()
        self.mode = mode
        self.crop_size = crop_size
        self.num_classes = num_classes
        self.entries = _read_manifest(self.manifest)
        self.priors: list[torch.Tensor] | None = None

        if mode == "train_u":
            if prior_path is None:
                self.priors = [torch.zeros(num_classes, dtype=torch.float32) for _ in self.entries]
            else:
                self.priors = self._prepare_priors(_load_torch(Path(prior_path).expanduser().resolve()))

        if nsample is not None:
            if mode == "val":
                raise ValueError("nsample is not supported for validation")
            if nsample <= 0:
                raise ValueError("nsample must be positive")
            indices = self._repeat_indices(len(self.entries), nsample)
            self.entries = [self.entries[index] for index in indices]
            if self.priors is not None:
                self.priors = [self.priors[index] for index in indices]

    @staticmethod
    def _repeat_indices(length: int, target: int) -> list[int]:
        if length <= 0:
            raise ValueError("cannot sample from an empty manifest")
        if target == length:
            return list(range(length))
        if target > length:
            return (list(range(length)) * math.ceil(target / length))[:target]
        if target == 1:
            return [0]
        return [round(index * (length - 1) / (target - 1)) for index in range(target)]

    def _prepare_priors(self, cache: Any) -> list[torch.Tensor]:
        if isinstance(cache, Mapping):
            values = []
            for image_path, _ in self.entries:
                if image_path in cache:
                    values.append(cache[image_path])
                elif image_path.replace("/", "\\") in cache:
                    values.append(cache[image_path.replace("/", "\\")])
                else:
                    raise KeyError(f"style-prior cache has no entry for {image_path}")
        elif isinstance(cache, (list, tuple)) or torch.is_tensor(cache):
            if len(cache) != len(self.entries):
                raise ValueError(
                    f"style-prior cache has {len(cache)} rows; {self.manifest} has {len(self.entries)}"
                )
            values = list(cache)
        else:
            raise TypeError("style-prior cache must be a tensor, sequence, or image-path mapping")

        priors: list[torch.Tensor] = []
        for index, value in enumerate(values):
            prior = torch.as_tensor(value, dtype=torch.float32).clone()
            if tuple(prior.shape) != (self.num_classes,):
                raise ValueError(
                    f"style-prior row {index} has shape {tuple(prior.shape)}; expected ({self.num_classes},)"
                )
            if not bool(torch.all((prior == 0) | (prior == 1))):
                raise ValueError(f"style-prior row {index} is not binary")
            priors.append(prior)
        return priors

    def __len__(self) -> int:
        return len(self.entries)

    def _path(self, relative: str) -> Path:
        return self.root / Path(relative)

    @staticmethod
    def _load_mask(path: Path) -> Image.Image:
        if path.suffix.lower() == ".npy":
            array = np.load(path, allow_pickle=False)
            if array.ndim != 2:
                raise ValueError(f"semantic label must be 2-D: {path}")
            return Image.fromarray(array.astype(np.uint8, copy=False))
        return Image.open(path)

    def __getitem__(self, index: int):
        image_relative, label_relative = self.entries[index]
        image = Image.open(self._path(image_relative)).convert("RGB")

        if self.mode == "train_u":
            mask = Image.fromarray(np.zeros((image.height, image.width), dtype=np.uint8))
        else:
            mask = self._load_mask(self._path(label_relative))

        if self.mode == "val":
            image_tensor, mask_tensor = normalize(image, mask)
            return image_tensor, mask_tensor, image_relative

        assert self.crop_size is not None
        image, mask = resize(image, mask, ratio_range=(0.5, 2.0))
        ignore_value = 254 if self.mode == "train_u" else 255
        image, mask = crop(image, mask, self.crop_size, ignore_value=ignore_value)
        image, mask = hflip(image, mask, p=0.5)

        if self.mode == "train_l":
            return normalize(image, mask)

        weak = deepcopy(image)
        strong1 = deepcopy(image)
        strong2 = deepcopy(image)
        if random.random() < 0.8:
            strong1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(strong1)
        strong1 = transforms.RandomGrayscale(p=0.2)(strong1)
        strong1 = blur(strong1, p=0.5)
        if random.random() < 0.8:
            strong2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(strong2)
        strong2 = transforms.RandomGrayscale(p=0.2)(strong2)
        strong2 = blur(strong2, p=0.5)

        cutmix1 = obtain_cutmix_box(strong1.size[0], p=0.5)
        cutmix2 = obtain_cutmix_box(strong2.size[0], p=0.5)
        ignore_mask = Image.fromarray(np.zeros((mask.height, mask.width), dtype=np.uint8))
        strong1_tensor, ignore_tensor = normalize(strong1, ignore_mask)
        strong2_tensor = normalize(strong2)
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.uint8).copy()).long()
        ignore_tensor[mask_tensor == 254] = 255

        assert self.priors is not None
        return (
            normalize(weak),
            strong1_tensor,
            strong2_tensor,
            ignore_tensor,
            cutmix1,
            cutmix2,
            self.priors[index],
        )
