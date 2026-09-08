"""Clean implementation of the two priors described in C-PGS."""

from .instance_prior import RefinementStep, refine_with_instance_prior
from .experiment_ip import (
    classify_instance_id_map,
    fuse_experiment_prediction,
    refine_with_experiment_instance_prior,
)
from .metrics import confusion_matrix, mean_iou
from .style_prior import (
    Hyperedge,
    StylePriorResult,
    infer_style_prior,
    inject_style_prior_numpy,
)

__all__ = [
    "Hyperedge",
    "RefinementStep",
    "StylePriorResult",
    "classify_instance_id_map",
    "confusion_matrix",
    "fuse_experiment_prediction",
    "infer_style_prior",
    "inject_style_prior_numpy",
    "mean_iou",
    "refine_with_experiment_instance_prior",
    "refine_with_instance_prior",
]
