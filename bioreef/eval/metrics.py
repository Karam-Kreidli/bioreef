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
    ERRORS only), and per-level accuracy (genus-, family-correct cumulative).

    Genus/family accuracy is derived from the taxonomy of the TOP-1 PREDICTED
    SPECIES (predicted species -> its genus/family, compared to the true genus/
    family), NOT from a marginalized parent-distribution argmax. This is uniform
    across every model — flat baselines and HSLM alike — so the panel is comparable
    (audit #83). It differs from how HSLM computes parent probabilities during
    TRAINING (marginalization); do not conflate the two in the paper."""
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
        # species_accuracy is mathematically IDENTICAL to top1/micro accuracy (both
        # = fraction with the exact species correct). Kept for the hierarchical
        # panel's internal completeness, but do NOT report it as a separate metric
        # from Top-1 in the paper (audit #84).
        "species_accuracy": species / total if total else 0.0,        # == top1/micro acc
        "genus_accuracy": (species + genus) / total if total else 0.0,
        "family_accuracy": (species + genus + family) / total if total else 0.0,
        # "root" = wrong family within the tree; "unknown" = prediction outside
        # the taxonomy entirely (e.g. an unrecognized MATANet name). Both are
        # cross-family errors — omitting unknown under-reports the rate.
        "cross_family_error_rate": (levels["root"] + levels.get("unknown", 0)) / total if total else 0.0,
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

    # Validate shapes/bounds up front: a malformed prediction array (e.g. from an
    # external model like MATANet) should raise a clear error here, not silently
    # produce a plausible-looking wrong number downstream (audit #81).
    if preds.shape != targets.shape:
        raise ValueError(
            f"preds shape {preds.shape} != targets shape {targets.shape}; "
            f"there must be exactly one prediction per target.")
    if preds.ndim != 1:
        raise ValueError(f"preds/targets must be 1-D class-index arrays, got "
                         f"{preds.ndim}-D {preds.shape}")
    if targets.size and (targets.min() < 0 or targets.max() >= num_classes):
        raise ValueError(
            f"target class index out of range [0,{num_classes}): "
            f"min {targets.min()}, max {targets.max()}")
    if scores is not None:
        scores = np.asarray(scores)
        if scores.shape[0] != preds.shape[0]:
            raise ValueError(f"scores rows {scores.shape[0]} != N {preds.shape[0]}")
        if scores.ndim != 2 or scores.shape[1] != num_classes:
            raise ValueError(f"scores must be (N,{num_classes}), got {scores.shape}")

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
