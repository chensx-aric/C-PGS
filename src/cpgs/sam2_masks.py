"""Thin, path-portable helpers around the vendored SAM2 implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def build_automatic_mask_generator(
    model_config: str | Path,
    checkpoint: str | Path,
    *,
    device: str = "cuda",
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.50,
    stability_score_thresh: float = 0.65,
    min_mask_region_area: int = 512,
):
    """Build the SAM2.1 automatic generator using the paper workspace settings."""

    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found: {checkpoint_path}. "
            "Run third_party/sam2/download_ckpts.sh first."
        )
    model = build_sam2(str(model_config), str(checkpoint_path), device=device)
    return SAM2AutomaticMaskGenerator(
        model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )


def generate_instance_masks(generator: Any, image_rgb: np.ndarray) -> list[np.ndarray]:
    """Return boolean masks while preserving SAM2's candidate ordering."""

    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must have shape (height, width, 3)")
    candidates = generator.generate(image)
    return [np.asarray(candidate["segmentation"], dtype=bool) for candidate in candidates]


def category_evidence(instance_labels, num_classes: int) -> np.ndarray:
    """Aggregate predicted instance labels into Algorithm 1's binary vector z."""

    labels = np.asarray(list(instance_labels), dtype=np.int64)
    if labels.size and (labels.min() < 0 or labels.max() >= num_classes):
        raise ValueError("instance label is outside the configured class range")
    evidence = np.zeros(num_classes, dtype=np.uint8)
    if labels.size:
        evidence[np.unique(labels)] = 1
    return evidence

