"""
Classification metric suite for the benchmark (paper Section 5.1, item D).

Everything here is pure: it takes integer predictions/targets plus the
idx->species map and taxonomy tree, and returns the full metric panel. Build and
unit-test this BEFORE the multi-seed runs — a metric bug found after 20 runs is
the expensive mistake.

Priority order (paper):
    1. Macro / class-balanced accuracy   (headline)
    2. Hierarchical Distance (HD)         (mean over all; mistake severity = mean over errors)
    3. Head / Medium / Tail accuracy      (by train frequency)
    4. Per-level accuracy (family->genus->species) and mistake severity
    5. Top-1 / Top-5
"""

import logging
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np

from .hd import DEFAULT_LEVEL_WEIGHTS, hierarchical_distance

logger = logging.getLogger("bioreef.eval.metrics")


# --- frequency-based head/medium/tail grouping -------------------------------

def freq_groups(
    samples_per_class: Sequence[int],
    head_thresh: int = 100,
    tail_thresh: int = 20,
) -> Dict[str, List[int]]:
    """Partition class indices into head / medium / tail by TRAIN frequency.
        head:   count >  head_thresh
        medium: tail_thresh < count <= head_thresh
        tail:   count <= tail_thresh
    Thresholds are the long-tail-benchmark convention; report them in the paper.
    """
    groups = {"head": [], "medium": [], "tail": []}
    for cls, count in enumerate(samples_per_class):
        if count > head_thresh:
            groups["head"].append(cls)
        elif count > tail_thresh:
            groups["medium"].append(cls)
        else:
            groups["tail"].append(cls)
    return groups


# --- core accuracy metrics ---------------------------------------------------

def macro_accuracy(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    """Mean per-class recall (class-balanced accuracy) — the headline metric.
    Classes absent from `targets` are ignored (not counted as 0)."""
    per_class = per_class_accuracy(preds, targets, num_classes)
    present = [acc for acc in per_class if acc is not None]
    return float(np.mean(present)) if present else 0.0


def per_class_accuracy(
    preds: np.ndarray, targets: np.ndarray, num_classes: int
) -> List[Optional[float]]:
    """Per-class recall; None for classes with no support in `targets`."""
    out: List[Optional[float]] = [None] * num_classes
    for cls in range(num_classes):
        mask = targets == cls
        n = int(mask.sum())
        if n:
            out[cls] = float((preds[mask] == cls).mean())
    return out


def micro_accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    """Overall Top-1 accuracy (sample-weighted)."""
    return float((preds == targets).mean()) if len(targets) else 0.0


def group_accuracy(
    preds: np.ndarray, targets: np.ndarray, num_classes: int,
    groups: Dict[str, List[int]],
) -> Dict[str, float]:
    """Macro accuracy within each head/medium/tail group."""
    per_class = per_class_accuracy(preds, targets, num_classes)
    out = {}
    for name, classes in groups.items():
        vals = [per_class[c] for c in classes if per_class[c] is not None]
        out[name] = float(np.mean(vals)) if vals else 0.0
    return out


def topk_accuracy(scores: np.ndarray, targets: np.ndarray, k: int = 5) -> float:
    """Top-k accuracy from class scores (N, C)."""
    if not len(targets):
        return 0.0
    topk = np.argsort(scores, axis=1)[:, -k:]
    return float(np.mean([t in topk[i] for i, t in enumerate(targets)]))


# --- hierarchical metrics ----------------------------------------------------

def hierarchical_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    idx_to_species: Dict[int, str],
    taxonomy_tree: Dict[str, Dict[str, str]],
    level_weights: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """HD-based metrics: mean HD over all samples, mistake severity (mean HD over
    ERRORS only), and per-level accuracy (genus-, family-correct cumulative)."""
    weights = level_weights or DEFAULT_LEVEL_WEIGHTS
    hd_scores = np.empty(len(targets), dtype=np.float64)
    levels = Counter()
    for i, (p, t) in enumerate(zip(preds, targets)):
        score, level = hierarchical_distance(
            idx_to_species[int(p)], idx_to_species[int(t)], taxonomy_tree, weights
        )
        hd_scores[i] = score
        levels[level] += 1

    total = len(targets)
    errors = hd_scores > 0
    n_err = int(errors.sum())
    species = levels["species"]
    genus = levels["genus"]
    family = levels["family"]
    return {
        "mean_hd": float(hd_scores.mean()) if total else 0.0,
        "mistake_severity": float(hd_scores[errors].mean()) if n_err else 0.0,
        "species_accuracy": species / total if total else 0.0,        # = micro acc
        "genus_accuracy": (species + genus) / total if total else 0.0,
        "family_accuracy": (species + genus + family) / total if total else 0.0,
        "cross_family_error_rate": levels["root"] / total if total else 0.0,
    }


# --- one-call full panel -----------------------------------------------------

def evaluate_classification(
    preds: np.ndarray,
    targets: np.ndarray,
    scores: Optional[np.ndarray],
    num_classes: int,
    idx_to_species: Dict[int, str],
    taxonomy_tree: Dict[str, Dict[str, str]],
    samples_per_class: Sequence[int],
    level_weights: Optional[Dict[str, int]] = None,
) -> Dict[str, object]:
    """Compute the full paper metric panel for one set of predictions.

    preds/targets: (N,) int class indices. scores: (N, C) for Top-5 (optional).
    samples_per_class: TRAIN counts, for head/medium/tail grouping.
    """
    preds = np.asarray(preds)
    targets = np.asarray(targets)

    groups = freq_groups(samples_per_class)
    result: Dict[str, object] = {
        "macro_accuracy": macro_accuracy(preds, targets, num_classes),
        "top1_accuracy": micro_accuracy(preds, targets),
        "group_accuracy": group_accuracy(preds, targets, num_classes, groups),
        "group_sizes": {k: len(v) for k, v in groups.items()},
    }
    result.update(hierarchical_metrics(
        preds, targets, idx_to_species, taxonomy_tree, level_weights
    ))
    if scores is not None:
        result["top5_accuracy"] = topk_accuracy(np.asarray(scores), targets, k=5)
    return result
