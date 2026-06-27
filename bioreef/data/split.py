"""
Leakage-safe, video/deployment-level grouped train/val/test split.

WHY GROUPED (the bug this fixes):
    The OzFish crops are frames from BRUVS stereo deployments. file_name encodes
    the deployment and camera:  A000001_L.avi.5107.png
                                ^^^^^^^ deployment   ^^^^ frame
    The same fish individual appears across consecutive frames AND in both the
    _L and _R stereo cameras of one deployment. A crop-level random shuffle
    therefore scatters near-duplicate views of the same individual across
    train/val/test, leaking identity and inflating test accuracy. The grouping
    unit must be the DEPLOYMENT (A000001), so every crop from one deployment
    lands entirely on one side.

WHAT THIS PRODUCES:
    A reproducible (seeded) ~70/15/15 split over deployments, stratified so each
    species is represented across folds as far as grouping allows. Species that
    live in too few deployments to split are placed deterministically (see
    _greedy_grouped_split) and reported — they are a documented limitation, not
    a silent leak.

The benchmark filtering (>=20 samples/species, placeholder drop) matches the
study design in the paper (Section 3).
"""

import logging
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from bioreef.data.taxonomy import is_placeholder_species

logger = logging.getLogger("bioreef.data.split")


def deployment_id(file_name: str) -> str:
    """Group key: the deployment prefix before the camera side.
    'A000001_L.avi.5107.png' -> 'A000001'. Both stereo cameras share it."""
    return file_name.split("_")[0]


def _load_rows(csv_path: str, img_dir: str, extra_img_dirs, filter_placeholders):
    """Read the metadata CSV into raw samples with resolved image paths.
    Each row -> {img_path, bbox(xywh), species, deployment}. Rows whose image is
    not found on disk are skipped (matches the production recipe)."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    search_dirs = [img_dir] + list(extra_img_dirs or [])
    raw = []
    n_missing_img = 0
    for _, row in df.iterrows():
        sp = row.get("species")
        if not isinstance(sp, str) or sp.strip() == "":
            continue
        if filter_placeholders and is_placeholder_species(sp):
            continue

        fname = row["file_name"]
        img_path = None
        for d in search_dirs:
            cand = os.path.join(d, fname)
            if os.path.exists(cand):
                img_path = cand
                break
        if img_path is None:
            n_missing_img += 1
            continue

        x0, y0, x1, y1 = int(row["x0"]), int(row["y0"]), int(row["x1"]), int(row["y1"])
        raw.append({
            "img_path": img_path,
            "bbox": [x0, y0, x1 - x0, y1 - y0],  # xyxy -> xywh for ContextHarvester
            "species": sp,
            "deployment": deployment_id(fname),
        })

    if n_missing_img:
        logger.warning("%d rows skipped (image not found on disk).", n_missing_img)
    return raw


def _greedy_grouped_split(
    samples, ratios=(0.70, 0.15, 0.15), seed=0
) -> Dict[str, str]:
    """Assign each DEPLOYMENT to one fold (train/val/test), keeping deployments
    intact while spreading every species across folds as evenly as grouping
    allows.

    Algorithm (deterministic given seed): process species rarest-first; for each,
    walk its deployments (shuffled by seed) and send each unassigned deployment
    to whichever fold is currently most short of that species relative to its
    target share. This guarantees rare species seed val/test before common
    species saturate the folds. Returns {deployment_id: fold}.
    """
    import random

    rng = random.Random(seed)
    fold_names = ["train", "val", "test"]

    # deployment -> species multiset, and species -> deployments.
    dep_species = defaultdict(Counter)
    sp_deps = defaultdict(set)
    for s in samples:
        dep_species[s["deployment"]][s["species"]] += 1
        sp_deps[s["species"]].add(s["deployment"])

    total_crops = len(samples)
    target_crops = {f: r * total_crops for f, r in zip(fold_names, ratios)}

    dep_fold: Dict[str, str] = {}
    fold_crops = {f: 0 for f in fold_names}
    fold_sp_crops = {f: Counter() for f in fold_names}        # crops per species per fold
    sp_total = Counter(s["species"] for s in samples)

    # Rarest species first (fewest deployments, then fewest crops).
    species_order = sorted(sp_deps, key=lambda sp: (len(sp_deps[sp]), sp_total[sp]))

    for sp in species_order:
        deps = [d for d in sp_deps[sp] if d not in dep_fold]
        rng.shuffle(deps)
        for dep in deps:
            dep_crop_count = sum(dep_species[dep].values())
            # Score each fold by how much it still NEEDS this species, then by
            # overall capacity. Lower score = more deserving.
            def need(f):
                sp_target = ratios[fold_names.index(f)] * sp_total[sp]
                sp_deficit = sp_target - fold_sp_crops[f][sp]          # want positive
                cap_deficit = target_crops[f] - fold_crops[f]          # want positive
                # Prioritise species deficit; break ties on overall capacity.
                return (-sp_deficit, -cap_deficit)
            best = min(fold_names, key=need)
            dep_fold[dep] = best
            fold_crops[best] += dep_crop_count
            for s2, c in dep_species[dep].items():
                fold_sp_crops[best][s2] += c

    # Any deployment with no benchmark species touched above (shouldn't happen,
    # but be safe) goes to the most under-target fold.
    for dep in dep_species:
        if dep not in dep_fold:
            best = min(fold_names, key=lambda f: fold_crops[f] - target_crops[f])
            dep_fold[dep] = best
            fold_crops[best] += sum(dep_species[dep].values())

    return dep_fold


def split_dataset(
    csv_path: str,
    img_dir: str,
    min_samples: int = 20,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 0,
    filter_placeholders: bool = True,
    extra_img_dirs: List[str] = None,
):
    """Build the leakage-safe deployment-grouped split.

    Returns (train, val, test, num_classes, class_to_species, samples_per_class),
    matching the tuple the trainer expects. Each sample dict has img_path, bbox
    (xywh), class_idx, species, deployment.
    """
    raw = _load_rows(csv_path, img_dir, extra_img_dirs, filter_placeholders)

    # >=min_samples species filter (the benchmark inclusion threshold).
    sp_counter = Counter(s["species"] for s in raw)
    kept_species = sorted(sp for sp, c in sp_counter.items() if c >= min_samples)
    species_to_class = {sp: i for i, sp in enumerate(kept_species)}
    class_to_species = {i: sp for sp, i in species_to_class.items()}

    samples = [s for s in raw if s["species"] in species_to_class]
    for s in samples:
        s["class_idx"] = species_to_class[s["species"]]

    # Deployment-level grouped, species-stratified fold assignment.
    dep_fold = _greedy_grouped_split(samples, ratios=ratios, seed=seed)

    folds = {"train": [], "val": [], "test": []}
    for s in samples:
        folds[dep_fold[s["deployment"]]].append(s)

    # samples_per_class from TRAIN ONLY (CB-Focal weights must not see val/test).
    sp_counts = [0] * len(kept_species)
    for s in folds["train"]:
        sp_counts[s["class_idx"]] += 1
    sp_counts = [max(1, c) for c in sp_counts]

    _log_split_summary(folds, dep_fold, kept_species, samples)

    return (folds["train"], folds["val"], folds["test"],
            len(kept_species), class_to_species, sp_counts)


def _log_split_summary(folds, dep_fold, kept_species, samples):
    """Report split sizes, deployment disjointness, and species missing from
    val/test (the unsplittable-rare-species limitation)."""
    n = len(samples)
    deps_per_fold = defaultdict(set)
    for s in samples:
        deps_per_fold[dep_fold[s["deployment"]]].add(s["deployment"])

    logger.info("Leakage-safe split (deployment-grouped):")
    for f in ("train", "val", "test"):
        crops = len(folds[f])
        logger.info("  %-5s: %6d crops (%.1f%%) over %4d deployments",
                    f, crops, 100 * crops / max(1, n), len(deps_per_fold[f]))

    # Sanity: deployments must be disjoint across folds.
    all_sets = [deps_per_fold[f] for f in ("train", "val", "test")]
    overlap = (all_sets[0] & all_sets[1]) | (all_sets[0] & all_sets[2]) | (all_sets[1] & all_sets[2])
    if overlap:
        logger.error("LEAK: %d deployments appear in >1 fold: %s",
                     len(overlap), sorted(overlap)[:5])
    else:
        logger.info("  deployment disjointness: OK (no deployment in >1 fold)")

    # Species coverage per fold.
    fold_species = {f: set(s["species"] for s in folds[f]) for f in folds}
    missing_test = [sp for sp in kept_species if sp not in fold_species["test"]]
    missing_val = [sp for sp in kept_species if sp not in fold_species["val"]]
    if missing_test:
        logger.warning("  %d/%d species absent from TEST (too few deployments to "
                       "split): %s", len(missing_test), len(kept_species),
                       missing_test[:5])
    if missing_val:
        logger.warning("  %d/%d species absent from VAL: %s",
                       len(missing_val), len(kept_species), missing_val[:5])
