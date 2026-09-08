"""Instance-prior refinement used by the representative ACA evaluation.

The accepted-paper run used the Algorithm 2 defaults together with the
experiment-time area, overlap, neighbourhood, and class-policy safeguards
preserved here. Keeping this path separate from cpgs.instance_prior makes the
clean pseudocode implementation and exact evaluation protocol unambiguous.
"""

from __future__ import annotations

from math import floor
from typing import Mapping

import numpy as np


DEFAULT_CATEGORY_POLICY: Mapping[int, str] = {
    4: "force",
    6: "prediction_only",
    10: "prediction_only",
    11: "prediction_only",
    13: "prediction_only",
    16: "prediction_only",
}


def classify_instance_id_map(
    prediction: np.ndarray,
    instance_ids: np.ndarray,
    *,
    num_classes: int,
    dominance_threshold: float = 0.6,
) -> np.ndarray:
    """Assign one semantic class to every region in an instance-ID map.

    This intentionally includes instance ID 0 because the verified historical
    evaluator treated every unique ID as a candidate region.
    """

    pred = np.asarray(prediction, dtype=np.int64)
    instances = np.asarray(instance_ids)
    if pred.ndim != 2 or instances.shape != pred.shape:
        raise ValueError("prediction and instance_ids must be equally sized 2-D arrays")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if not 0.0 <= dominance_threshold <= 1.0:
        raise ValueError("dominance_threshold must be in [0, 1]")
    if pred.size and (pred.min() < 0 or pred.max() >= num_classes):
        raise ValueError("prediction contains an out-of-range class index")

    classified = np.zeros_like(pred)
    for instance_id in np.unique(instances):
        region = instances == instance_id
        region_values = pred[region]
        if region_values.size == 0:
            continue

        histogram = np.bincount(region_values, minlength=num_classes)
        dominant_label = int(np.argmax(histogram))
        dominant_ratio = float(histogram[dominant_label] / region_values.size)

        ys, xs = np.nonzero(region)
        center_y = (int(ys.min()) + int(ys.max())) // 2
        center_x = (int(xs.min()) + int(xs.max())) // 2
        center_label = int(pred[center_y, center_x])
        has_non_base_label = bool(np.any((region_values != 0) & (region_values != 1)))

        if dominant_ratio >= dominance_threshold or not has_non_base_label:
            selected_label = dominant_label
        else:
            selected_label = center_label
        classified[region] = selected_label

    return classified


def fuse_experiment_prediction(
    prediction: np.ndarray,
    classified_instances: np.ndarray,
    *,
    normalized_expansion: float = 0.6,
    radius_scale: float = 100.0,
    base_iou: float = 0.3,
    min_area_ratio: float = 0.0002,
    neighbor_consistency: float = 0.25,
    category_policy: Mapping[int, str] = DEFAULT_CATEGORY_POLICY,
) -> np.ndarray:
    """Apply the safeguards used for the paper's representative IP result."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - full environment dependency
        raise RuntimeError("OpenCV is required for representative IP evaluation") from exc

    pred = np.asarray(prediction, dtype=np.int64)
    semantic_instances = np.asarray(classified_instances, dtype=np.int64)
    if pred.ndim != 2 or semantic_instances.shape != pred.shape:
        raise ValueError(
            "prediction and classified_instances must be equally sized 2-D arrays"
        )
    if normalized_expansion < 0 or radius_scale < 0:
        raise ValueError("expansion values must be non-negative")

    expansion_pixels = floor(float(radius_scale) * float(normalized_expansion))
    fused = pred.copy()
    total_pixels = pred.size
    if total_pixels == 0:
        return fused

    for class_id_value in np.unique(semantic_instances):
        class_id = int(class_id_value)
        if class_id in (0, 1):
            continue

        policy = category_policy.get(class_id, "default")
        if policy == "prediction_only":
            continue
        if policy not in {"default", "force"}:
            raise ValueError(f"unknown category policy for class {class_id}: {policy}")

        class_mask = np.ascontiguousarray(
            semantic_instances == class_id, dtype=np.uint8
        )
        component_count, component_ids = cv2.connectedComponents(
            class_mask, connectivity=8
        )

        for component_id in range(1, component_count):
            region = np.ascontiguousarray(
                component_ids == component_id, dtype=np.uint8
            )
            area = int(region.sum())
            if area < total_pixels * min_area_ratio:
                continue

            region_bool = region.astype(bool)
            if policy == "force":
                fused[region_bool] = class_id
                continue

            predicted_class = pred == class_id
            intersection = region_bool & predicted_class
            intersection_count = int(intersection.sum())
            union_count = int((region_bool | predicted_class).sum())
            iou = intersection_count / (union_count + 1e-6)
            region_ratio = area / total_pixels
            iou_threshold = base_iou - 0.25 * np.exp(-6 * region_ratio)
            if iou < iou_threshold or intersection_count == 0:
                continue

            neighbor_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            neighbor = cv2.dilate(region, neighbor_kernel) - region
            if neighbor.sum() > 0:
                ratio = float((pred[neighbor == 1] == class_id).mean())
                if ratio < neighbor_consistency:
                    continue

            expansion_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (expansion_pixels * 2 + 1, expansion_pixels * 2 + 1),
            )
            allowed = cv2.dilate(
                np.ascontiguousarray(intersection, dtype=np.uint8),
                expansion_kernel,
                iterations=1,
            )
            fused[region_bool & allowed.astype(bool)] = class_id

    return fused


def refine_with_experiment_instance_prior(
    prediction: np.ndarray,
    instance_ids: np.ndarray,
    *,
    num_classes: int,
    dominance_threshold: float = 0.6,
    normalized_expansion: float = 0.6,
    radius_scale: float = 100.0,
) -> np.ndarray:
    """Run the complete IP path used to obtain the reported 1/2 result."""

    classified = classify_instance_id_map(
        prediction,
        instance_ids,
        num_classes=num_classes,
        dominance_threshold=dominance_threshold,
    )
    return fuse_experiment_prediction(
        prediction,
        classified,
        normalized_expansion=normalized_expansion,
        radius_scale=radius_scale,
    )
