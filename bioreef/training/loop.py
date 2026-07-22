"""
Family-agnostic train + evaluate loop.

One implementation serves every model family (dino / timm), because all models
share the interface forward(streams)->logits and the same dataset. This is what
keeps the preprocessing + eval path identical across the benchmark panel (the
fairness rule) — there is no per-family eval code to drift.

Single-GPU only (the frozen-backbone runs are light; timm baselines fit one GPU
at batch 32). DDP training stays in scripts/train.py for the heavy multi-GPU jobs.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from bioreef.data import (
    split_from_config, get_taxonomy_tree, build_taxonomy_maps, FishCropDataset,
)
from bioreef.model import (
    build_model, trainable_parameters, backbone_is_frozen, Classifier, HSLMLoss,
)
from bioreef.training import CBFocalLoss, BalancedDistributedSampler, EMA
from bioreef.eval import evaluate_classification


def probe_cache_eligible(run_cfg):
    """A run can use the frozen-feature cache iff the backbone is frozen, there is
    no context module, AND augmentation is OFF — then the ROI [CLS] is a fixed
    per-crop input, safe to compute once and reuse across epochs/seeds.

    The augment=False requirement is a HARD safety condition: with augmentation on,
    the crop (and thus its features) changes every epoch, so a cache would silently
    serve stale features. The current panel uses light augmentation on every run
    (incl. the probe), so nothing is cache-eligible by default; caching remains
    available only for a deliberately un-augmented probe (cache_features + augment
    both set false)."""
    return (
        getattr(run_cfg, "cache_features", False)
        and not getattr(run_cfg, "augment", True)     # HARD: no cache if augmenting
        and run_cfg.model_family == "dino"
        and run_cfg.context_levels == 0
        and run_cfg.unfreeze_blocks == 0
    )


def build_loss(run_cfg, sp_counts, idx_to_sp, tree, device):
    """Loss per the run config: HSLM (default) | CB-Focal | plain CE."""
    if run_cfg.hslm:
        s2g, s2f, n_gen, n_fam, n_missing = build_taxonomy_maps(idx_to_sp, tree)
        if n_missing:
            raise RuntimeError(f"[hslm] {n_missing} species have no taxonomy entry; "
                               "genus/family loss would be meaningless. Fix the CSV or disable hslm.")
        return HSLMLoss(
            sp_counts, s2g, s2f, n_gen, n_fam,
            family_weight=run_cfg.family_weight, genus_weight=run_cfg.genus_weight,
            species_weight=run_cfg.species_weight, device=device,
        )
    if run_cfg.loss == "cbfocal":
        return CBFocalLoss(sp_counts, device=device)
    return nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate(model, dl, device, num_classes, idx_to_sp, tree, sp_counts):
    model.eval()
    preds, targets, scores = [], [], []
    for batch in dl:
        streams = {k: v.to(device) for k, v in batch["streams"].items()}
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(streams)
        prob = torch.softmax(logits, dim=1).float().cpu().numpy()
        scores.append(prob)
        preds.extend(prob.argmax(1).tolist())
        targets.extend(batch["label"].tolist())
    return evaluate_classification(
        np.array(preds), np.array(targets), np.vstack(scores),
        num_classes, idx_to_sp, tree, sp_counts,
    )


def train_and_evaluate(run_cfg, bench, seed, device, batch_size=32,
                       num_workers=4, log=print):
    """Run one experiment: split -> model -> train -> return (test_metrics,
    val_metrics_of_best, model, idx_to_sp). Selects best epoch by val mean HD.

    Assumes set_seed(seed) has already been called by the caller.
    """
    train_s, val_s, test_s, num_classes, idx_to_sp, sp_counts = split_from_config(
        bench.csv_path, bench.img_dir, bench,
    )
    tree = get_taxonomy_tree(bench.csv_path)
    log(f"[data] {num_classes} species | train {len(train_s)} "
        f"val {len(val_s)} test {len(test_s)}")

    # Fast path: frozen-backbone probe trains only the head on cached features.
    if probe_cache_eligible(run_cfg):
        return _train_probe_cached(
            run_cfg, bench, seed, device, num_classes, idx_to_sp, sp_counts, tree,
            train_s, val_s, test_s, batch_size, num_workers, log,
        )

    train_ds = FishCropDataset(train_s, is_train=run_cfg.augment)  # augment off = clean crops
    val_ds = FishCropDataset(val_s, is_train=False)
    test_ds = FishCropDataset(test_s, is_train=False)

    if run_cfg.sampler == "balanced":
        sampler = BalancedDistributedSampler(train_s, num_replicas=1, rank=0, seed=seed)
        shuffle = False
    else:
        sampler, shuffle = None, True

    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                          shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)

    model = build_model(run_cfg, num_classes).to(device)
    criterion = build_loss(run_cfg, sp_counts, idx_to_sp, tree, device)
    optimizer = optim.AdamW(trainable_parameters(model), lr=run_cfg.lr, weight_decay=0.01)

    epochs, warmup = run_cfg.epochs, run_cfg.warmup_epochs
    if warmup > 0:
        warm = optim.lr_scheduler.LinearLR(optimizer, 1e-2, 1.0, warmup)
        cos = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warm, cos], [warmup])
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = EMA(model, decay=0.999)
    best_hd, best_val, best_state = float("inf"), None, None

    for epoch in range(1, epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        if backbone_is_frozen(model):
            model.backbone.eval()  # keep frozen backbone in eval (BN/dropout off)
        it = tqdm(train_dl, desc=f"{run_cfg.slug} s{seed} ep{epoch}/{epochs}")
        for batch in it:
            streams = {k: v.to(device) for k, v in batch["streams"].items()}
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(model(streams), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
        scheduler.step()

        backup = ema.apply_to(model)
        val = evaluate(model, val_dl, device, num_classes, idx_to_sp, tree, sp_counts)
        log(f"[val ep{epoch:02d}] macroAcc {val['macro_accuracy']:.4f} "
            f"HD {val['mean_hd']:.4f} top1 {val['top1_accuracy']:.4f}")
        if val["mean_hd"] < best_hd:
            best_hd, best_val = val["mean_hd"], val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        ema.restore(model, backup)

    # Load best (EMA) weights and score the held-out test set once.
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_dl, device, num_classes, idx_to_sp, tree, sp_counts)
    return test_metrics, best_val, model, idx_to_sp, num_classes


def _probe_head_forward(model, feats):
    """Run only the trainable head-side of a no-context Classifier on cached ROI
    [CLS] features: roi_only_proj -> head. (embed() would re-run the backbone.)"""
    return model.head(model.roi_only_proj(feats))


@torch.no_grad()
def _evaluate_features(model, features, labels, device, num_classes,
                       idx_to_sp, tree, sp_counts, batch_size=512):
    """Metric panel over cached features (no image I/O, no backbone)."""
    model.eval()
    preds, targets, scores = [], [], []
    for i in range(0, len(labels), batch_size):
        f = features[i:i + batch_size].to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = _probe_head_forward(model, f)
        prob = torch.softmax(logits, dim=1).float().cpu().numpy()
        scores.append(prob)
        preds.extend(prob.argmax(1).tolist())
        targets.extend(labels[i:i + batch_size].tolist())
    return evaluate_classification(
        np.array(preds), np.array(targets), np.vstack(scores),
        num_classes, idx_to_sp, tree, sp_counts,
    )


def _train_probe_cached(run_cfg, bench, seed, device, num_classes, idx_to_sp,
                        sp_counts, tree, train_s, val_s, test_s,
                        batch_size, num_workers, log):
    """Linear-probe fast path: compute frozen ROI [CLS] features once (cached,
    un-augmented, reused across seeds), then train ONLY the head on cached
    vectors. Same return tuple + best-by-val-HD selection as the full loop."""
    from bioreef.training.feature_cache import compute_or_load, TensorDataset

    # Build the real model so the trainable head-modules (roi_only_proj + head)
    # and the frozen backbone are exactly what the full path would use.
    model = build_model(run_cfg, num_classes).to(device)
    assert isinstance(model, Classifier) and model.mceam is None

    # Frozen ROI [CLS] features for every split (computed once, cached to disk).
    fx = lambda s, name: compute_or_load(
        model.backbone, s, bench, run_cfg, name, device,
        batch_size=max(batch_size, 64), num_workers=num_workers, log=log,
    )
    tr_f, tr_y = fx(train_s, "train")
    va_f, va_y = fx(val_s, "val")
    te_f, te_y = fx(test_s, "test")

    # Train only the head-side on cached features. Same optimization regime as the
    # full loop (EMA + warmup+cosine, best-by-val-HD on EMA weights) so C01 differs
    # from the panel ONLY in the intended ways (frozen probe, un-augmented).
    criterion = build_loss(run_cfg, sp_counts, idx_to_sp, tree, device)
    head_params = list(model.roi_only_proj.parameters()) + list(model.head.parameters())
    optimizer = optim.AdamW(head_params, lr=run_cfg.lr, weight_decay=0.01)
    epochs, warmup = run_cfg.epochs, run_cfg.warmup_epochs
    if warmup > 0:
        warm = optim.lr_scheduler.LinearLR(optimizer, 1e-2, 1.0, warmup)
        cos = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warm, cos], [warmup])
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ema = EMA(model, decay=0.999)   # shadows only trainable (head) params

    if run_cfg.sampler == "balanced":
        sampler = BalancedDistributedSampler(train_s, num_replicas=1, rank=0, seed=seed)
    else:
        sampler = None
    dl = DataLoader(TensorDataset(tr_f, tr_y), batch_size=batch_size,
                    sampler=sampler, shuffle=sampler is None)

    best_hd, best_val, best_state = float("inf"), None, None
    model.backbone.eval()
    for epoch in range(1, epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.roi_only_proj.train(); model.head.train()
        for f, y in dl:
            f, y = f.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(_probe_head_forward(model, f), y)
            loss.backward()
            optimizer.step()
            ema.update(model)
        scheduler.step()

        # Validate on EMA weights, like the full loop.
        backup = ema.apply_to(model)
        val = _evaluate_features(model, va_f, va_y, device, num_classes,
                                 idx_to_sp, tree, sp_counts)
        log(f"[val ep{epoch:02d}] macroAcc {val['macro_accuracy']:.4f} "
            f"HD {val['mean_hd']:.4f} top1 {val['top1_accuracy']:.4f}")
        if val["mean_hd"] < best_hd:
            best_hd, best_val = val["mean_hd"], val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        ema.restore(model, backup)

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = _evaluate_features(model, te_f, te_y, device, num_classes,
                                      idx_to_sp, tree, sp_counts)
    return test_metrics, best_val, model, idx_to_sp, num_classes
