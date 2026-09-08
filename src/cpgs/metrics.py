"""Segmentation metrics shared by clean reproduction scripts."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def confusion_matrix(
    predictions: Iterable[np.ndarray],
    targets: Iterable[np.ndarray],
    *,
    num_classes: int,
    ignore_index: int = 255,
) -> np.ndarray:
    """Accumulate a standard semantic-segmentation confusion matrix."""

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    pair_count = 0
    for prediction, target in zip(predictions, targets):
        pair_count += 1
        pred = np.asarray(prediction, dtype=np.int64)
        truth = np.asarray(target, dtype=np.int64)
        if pred.shape != truth.shape:
            raise ValueError("prediction and target shapes differ")
        valid = (truth != ignore_index) & (truth >= 0) & (truth < num_classes)
        if np.any((pred[valid] < 0) | (pred[valid] >= num_classes)):
            raise ValueError("prediction contains an out-of-range class index")
        encoded = truth[valid] * num_classes + pred[valid]
        matrix += np.bincount(encoded, minlength=num_classes * num_classes).reshape(
            num_classes, num_classes
        )
    if pair_count == 0:
        raise ValueError("at least one prediction/target pair is required")
    return matrix


def mean_iou(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    """Return mIoU and per-class IoU, excluding classes absent from the target."""

    confusion = np.asarray(matrix, dtype=np.float64)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion matrix must be square")
    intersection = np.diag(confusion)
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - intersection
    per_class = np.full(union.shape, np.nan, dtype=np.float64)
    present = union > 0
    per_class[present] = intersection[present] / union[present]
    return float(np.nanmean(per_class)), per_class

