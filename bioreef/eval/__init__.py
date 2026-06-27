"""Evaluation: the Hierarchical Distance rule + the full classification metric
suite (macro acc, HD, head/medium/tail, per-level, mistake severity)."""

from .hd import hierarchical_distance, DEFAULT_LEVEL_WEIGHTS
from .metrics import (
    evaluate_classification,
    macro_accuracy,
    per_class_accuracy,
    micro_accuracy,
    group_accuracy,
    topk_accuracy,
    hierarchical_metrics,
    freq_groups,
)

__all__ = [
    "hierarchical_distance", "DEFAULT_LEVEL_WEIGHTS",
    "evaluate_classification", "macro_accuracy", "per_class_accuracy",
    "micro_accuracy", "group_accuracy", "topk_accuracy", "hierarchical_metrics",
    "freq_groups",
]
