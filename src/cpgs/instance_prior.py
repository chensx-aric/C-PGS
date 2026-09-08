"""SAM2 instance-geometric-prior refinement (paper Algorithm 2)."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class RefinementStep:
    """Decision made for one SAM2 candidate mask."""

    index: int
    label: int
    dominant_label: int
    dominance_ratio: float
    used_center_fallback: bool
    radius_pixels: int
    source_pixels: int
    refined_pixels: int


def _coerce_mask(candidate, shape: tuple[int, int]) -> np.ndarray:
    if isinstance(candidate, Mapping):
        if "segmentation" not in candidate:
            raise ValueError("SAM2 mask dictionaries must contain 'segmentation'")
        candidate = candidate["segmentation"]
    mask = np.asarray(candidate, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"instance mask has shape {mask.shape}; expected {shape}")
    return mask


def _center_pixel(mask: np.ndarray) -> tuple[int, int]:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise ValueError("instance masks must not be empty")
    y0, x0 = coordinates.min(axis=0)
    y1, x1 = coordinates.max(axis=0)
    center = np.asarray([(int(y0) + int(y1)) // 2, (int(x0) + int(x1)) // 2])
    cy, cx = int(center[0]), int(center[1])
    if mask[cy, cx]:
        return cy, cx

    distances = np.square(coordinates - center).sum(axis=1)
    nearest = coordinates[int(np.argmin(distances))]
    return int(nearest[0]), int(nearest[1])


def _disk(radius: int) -> np.ndarray:
    coordinates = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    return (xx * xx + yy * yy) <= radius * radius


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    try:
        from scipy.ndimage import binary_dilation

        return binary_dilation(mask, structure=_disk(radius))
    except ImportError:
        # Dependency-free fallback used only when SciPy is unavailable.
        result = np.zeros_like(mask, dtype=bool)
        height, width = mask.shape
        for dy, dx in np.argwhere(_disk(radius)) - radius:
            y_src_start = max(0, -int(dy))
            y_src_stop = min(height, height - int(dy))
            x_src_start = max(0, -int(dx))
            x_src_stop = min(width, width - int(dx))
            y_dst_start = y_src_start + int(dy)
            y_dst_stop = y_src_stop + int(dy)
            x_dst_start = x_src_start + int(dx)
            x_dst_stop = x_src_stop + int(dx)
            result[y_dst_start:y_dst_stop, x_dst_start:x_dst_stop] |= mask[
                y_src_start:y_src_stop, x_src_start:x_src_stop
            ]
        return result


def refine_with_instance_prior(
    prediction: np.ndarray,
    instance_masks: Iterable[np.ndarray | Mapping[str, object]],
    *,
    num_classes: int,
    dominance_threshold: float = 0.6,
    normalized_expansion: float = 0.6,
    radius_scale: float = 100.0,
) -> tuple[np.ndarray, tuple[RefinementStep, ...]]:
    """Refine a class map by propagating one label over each dilated instance.

    Histograms and fallback labels are always computed from the original initial
    prediction, as specified by Algorithm 2. Candidate masks are then applied in
    their provided order, so a later expanded mask deterministically wins where
    masks overlap.
    """

    initial = np.asarray(prediction)
    if initial.ndim != 2:
        raise ValueError("prediction must be a two-dimensional class map")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if not 0.0 <= dominance_threshold <= 1.0:
        raise ValueError("dominance_threshold must be in [0, 1]")
    if normalized_expansion < 0 or radius_scale < 0:
        raise ValueError("expansion values must be non-negative")
    if initial.size and (initial.min() < 0 or initial.max() >= num_classes):
        raise ValueError("prediction contains an out-of-range class index")

    radius = floor(float(radius_scale) * float(normalized_expansion))
    refined = initial.copy()
    audit: list[RefinementStep] = []

    for index, candidate in enumerate(instance_masks):
        mask = _coerce_mask(candidate, initial.shape)
        pixel_count = int(mask.sum())
        if pixel_count == 0:
            continue

        histogram = np.bincount(initial[mask].astype(np.int64), minlength=num_classes)
        dominant_label = int(np.argmax(histogram))
        ratio = float(histogram[dominant_label] / pixel_count)
        use_fallback = ratio < dominance_threshold
        if use_fallback:
            cy, cx = _center_pixel(mask)
            label = int(initial[cy, cx])
        else:
            label = dominant_label

        expanded = _dilate(mask, radius)
        refined[expanded] = label
        audit.append(
            RefinementStep(
                index=index,
                label=label,
                dominant_label=dominant_label,
                dominance_ratio=ratio,
                used_center_fallback=use_fallback,
                radius_pixels=radius,
                source_pixels=pixel_count,
                refined_pixels=int(expanded.sum()),
            )
        )

    return refined, tuple(audit)

