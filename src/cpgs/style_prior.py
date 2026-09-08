"""Hypergraph-guided category-level style prior (paper Algorithm 1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Hyperedge:
    """One labeled-image category set and its building-group identifier."""

    categories: frozenset[int]
    group: Hashable
    sample_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "categories", frozenset(int(c) for c in self.categories))
        if any(c < 0 for c in self.categories):
            raise ValueError("category indices must be non-negative")


@dataclass(frozen=True)
class StylePriorResult:
    """Auditable intermediate values returned by style-prior inference."""

    prior: np.ndarray
    prototype: Hyperedge
    best_matches: tuple[Hyperedge, ...]
    overlap_score: int
    support: np.ndarray


def _as_binary_evidence(evidence: Sequence[int | bool], num_classes: int) -> np.ndarray:
    vector = np.asarray(evidence)
    if vector.ndim != 1 or vector.size != num_classes:
        raise ValueError(f"evidence must have shape ({num_classes},)")
    if not np.all(np.isin(vector, (0, 1, False, True))):
        raise ValueError("evidence must be binary")
    return vector.astype(bool, copy=False)


def infer_style_prior(
    evidence: Sequence[int | bool],
    group: Hashable,
    hyperedges: Iterable[Hyperedge],
    num_classes: int,
    *,
    support_lower: int = 1,
    support_upper: int | None = 3,
) -> StylePriorResult:
    """Infer the binary category prior for one unlabeled image.

    This follows Algorithm 1: filter reference hyperedges by BGK, maximize
    category overlap, choose the minimal-cardinality best match, and complete
    its category set with classes whose support is inside the inclusive bounds.
    The author-confirmed paper defaults are tau_l=1 and tau_h=3. Passing None
    as support_upper retains the generic unbounded behavior. Ties after
    cardinality are resolved lexicographically for reproducibility.
    """

    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if support_lower < 0:
        raise ValueError("support_lower must be non-negative")

    evidence_vector = _as_binary_evidence(evidence, num_classes)
    candidates = tuple(edge for edge in hyperedges if edge.group == group)
    if not candidates:
        raise ValueError(f"no reference hyperedges found for building group {group!r}")

    for edge in candidates:
        if any(c >= num_classes for c in edge.categories):
            raise ValueError(f"hyperedge {edge.sample_id!r} contains an out-of-range class")

    scores = np.asarray(
        [sum(bool(evidence_vector[c]) for c in edge.categories) for edge in candidates],
        dtype=np.int64,
    )
    max_score = int(scores.max())
    best_matches = tuple(edge for edge, score in zip(candidates, scores) if score == max_score)
    prototype = min(
        best_matches,
        key=lambda edge: (len(edge.categories), tuple(sorted(edge.categories)), edge.sample_id),
    )

    support = np.zeros(num_classes, dtype=np.int64)
    for edge in best_matches:
        if edge.categories:
            support[np.fromiter(sorted(edge.categories), dtype=np.int64)] += 1

    upper = len(best_matches) if support_upper is None else int(support_upper)
    if upper < support_lower:
        raise ValueError("support_upper must be greater than or equal to support_lower")

    completed = set(prototype.categories)
    completed.update(np.flatnonzero((support >= support_lower) & (support <= upper)).tolist())
    prior = np.zeros(num_classes, dtype=np.float32)
    if completed:
        prior[np.fromiter(sorted(completed), dtype=np.int64)] = 1.0

    return StylePriorResult(
        prior=prior,
        prototype=prototype,
        best_matches=best_matches,
        overlap_score=max_score,
        support=support,
    )


def inject_style_prior_numpy(
    logits: np.ndarray,
    prior: np.ndarray,
    *,
    alpha: float = 0.6,
    class_axis: int = 1,
) -> np.ndarray:
    """Apply Equation 16, ``adjusted_logits = logits + alpha * prior``."""

    array = np.asarray(logits)
    if array.ndim < 1:
        raise ValueError("logits must have at least one dimension")
    axis = class_axis % array.ndim
    prior_array = np.asarray(prior, dtype=array.dtype)

    if prior_array.ndim == 1:
        if prior_array.size != array.shape[axis]:
            raise ValueError("prior length does not match the class dimension")
        shape = [1] * array.ndim
        shape[axis] = prior_array.size
    elif prior_array.ndim == 2 and axis == 1 and array.ndim >= 2:
        if prior_array.shape != array.shape[:2]:
            raise ValueError("batched prior must have shape (batch, classes)")
        shape = list(prior_array.shape) + [1] * (array.ndim - 2)
    else:
        raise ValueError("prior must have shape (classes,) or (batch, classes)")

    return array + float(alpha) * prior_array.reshape(shape)


def inject_style_prior_torch(logits, prior, *, alpha: float = 0.6):
    """Torch equivalent of Equation 16 for ``(B,C,H,W)`` logits."""

    import torch

    if not torch.is_tensor(logits) or logits.ndim != 4:
        raise ValueError("logits must be a torch.Tensor with shape (B,C,H,W)")
    prior_tensor = torch.as_tensor(prior, dtype=logits.dtype, device=logits.device)
    if prior_tensor.ndim == 1:
        if prior_tensor.numel() != logits.shape[1]:
            raise ValueError("prior length does not match the class dimension")
        prior_tensor = prior_tensor.view(1, -1, 1, 1)
    elif prior_tensor.ndim == 2:
        if tuple(prior_tensor.shape) != tuple(logits.shape[:2]):
            raise ValueError("batched prior must have shape (batch, classes)")
        prior_tensor = prior_tensor[:, :, None, None]
    else:
        raise ValueError("prior must have shape (classes,) or (batch, classes)")
    return logits + float(alpha) * prior_tensor
