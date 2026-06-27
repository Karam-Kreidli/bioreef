"""
Train the proposed classifier (C09) or any ablation on the OzFish benchmark.

The data split is FIXED across all runs (--split_seed, default 0) so every model
sees the same train/val/test. --seed varies model init + augmentation + sampling
only — run {0,1,2} for the paper's mean +/- std.

Single GPU:
    python scripts/train.py --csv frame_metadata.csv --img_dir <frames> --seed 0
Multi-GPU (DDP):
    torchrun --nproc_per_node=2 scripts/train.py --csv ... --img_dir ... --seed 0

Ablations are flags (paper K.2):
    --backbone dinov2            (A1)
    --context_levels 0           (A2: no MCEAM)   / 1 (A3: ROI-scale)
    --attention_depth 2          (A4)             / 4 (A4b)
    --no_hslm                    (A5: flat softmax)
    --sampler random             (A6)
    --loss ce                    (A7)
    --no_hslm --loss ce --sampler random   (A8)
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bioreef.config import BenchmarkConfig, DEFAULT_CONFIG_PATH
from bioreef.data import (
    split_from_config, get_taxonomy_tree, build_taxonomy_maps, FishCropDataset,
)
from bioreef.model import Classifier, ModelConfig, HSLMLoss
from bioreef.training import set_seed, CBFocalLoss, BalancedDistributedSampler, EMA
from bioreef.eval import evaluate_classification


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Data
    p.add_argument("--csv", required=True, help="frame_metadata.csv")
    p.add_argument("--img_dir", required=True, help="directory of crop frames")
    # Benchmark definition lives in the config; these override individual fields.
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                   help="benchmark config YAML (inclusion rules + split params)")
    p.add_argument("--min_samples", type=int, default=None, help="override config")
    p.add_argument("--min_deployments", type=int, default=None, help="override config")
    p.add_argument("--split_seed", type=int, default=None,
                   help="override config; FIXED across runs to keep the split constant")
    # Reproducibility
    p.add_argument("--seed", type=int, default=0, help="model init / aug / sampler seed")
    p.add_argument("--deterministic", action="store_true")
    # Model (ablation flags)
    p.add_argument("--backbone", default="dinov3", choices=["dinov3", "dinov2"])
    p.add_argument("--context_levels", type=int, default=3, choices=[0, 1, 3])
    p.add_argument("--attention_depth", type=int, default=1)
    p.add_argument("--unfreeze_blocks", type=int, default=0)
    # Loss / sampler (ablation flags)
    p.add_argument("--no_hslm", action="store_true", help="flat softmax instead of HSLM")
    p.add_argument("--loss", default="cbfocal", choices=["cbfocal", "ce"])
    p.add_argument("--sampler", default="balanced", choices=["balanced", "random"])
    p.add_argument("--family_weight", type=float, default=3.0)
    p.add_argument("--genus_weight", type=float, default=2.0)
    p.add_argument("--species_weight", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.9999)
    p.add_argument("--gamma", type=float, default=2.0)
    # Optimisation
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out", default="checkpoint.pt", help="best-HD checkpoint path")
    return p.parse_args()


def is_dist():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def main():
    args = parse_args()

    # DDP setup (no-op single GPU).
    if is_dist():
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        world_size = dist.get_world_size()
    else:
        local_rank, world_size = 0, 1
    is_main = local_rank == 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed, deterministic=args.deterministic)

    # Benchmark definition: config file, with any CLI flags overriding.
    bench = BenchmarkConfig.from_yaml(args.config).apply_overrides(
        min_samples=args.min_samples,
        min_deployments=args.min_deployments,
        split_seed=args.split_seed,
    )
    if is_main:
        print(f"[config] {bench}")

    # Fixed benchmark split (driven by the config).
    train_s, val_s, _test_s, num_classes, idx_to_sp, sp_counts = split_from_config(
        args.csv, args.img_dir, bench,
    )
    tree = get_taxonomy_tree(args.csv)
    if is_main:
        print(f"[data] {num_classes} species | train {len(train_s)} val {len(val_s)}")

    train_ds = FishCropDataset(train_s, is_train=True)
    val_ds = FishCropDataset(val_s, is_train=False)

    # Sampler (ablation A6/A8).
    if args.sampler == "balanced":
        train_sampler = BalancedDistributedSampler(
            train_s, num_replicas=world_size, rank=local_rank, seed=args.seed,
        )
        shuffle = False
    else:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds, shuffle=True) if is_dist() else None
        shuffle = train_sampler is None

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                          shuffle=shuffle, num_workers=args.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # Model.
    mcfg = ModelConfig(
        backbone=args.backbone, context_levels=args.context_levels,
        attention_depth=args.attention_depth, unfreeze_blocks=args.unfreeze_blocks,
    )
    model = Classifier(mcfg, num_classes).to(device)

    trainable = []
    for m in model.trainable_modules():
        trainable += list(m.parameters())
    if args.unfreeze_blocks > 0:
        trainable += [p for p in model.backbone.parameters() if p.requires_grad]

    if is_dist():
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    core = model.module if is_dist() else model

    # Loss (ablation A5/A7/A8).
    if not args.no_hslm:
        s2g, s2f, n_gen, n_fam, n_missing = build_taxonomy_maps(idx_to_sp, tree)
        if is_main and n_missing:
            print(f"[hslm] {n_missing}/{num_classes} species missing taxonomy -> __unknown__")
        criterion = HSLMLoss(
            sp_counts, s2g, s2f, n_gen, n_fam,
            family_weight=args.family_weight, genus_weight=args.genus_weight,
            species_weight=args.species_weight, beta=args.beta, gamma=args.gamma,
            device=device,
        )
    elif args.loss == "cbfocal":
        criterion = CBFocalLoss(sp_counts, beta=args.beta, gamma=args.gamma, device=device)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(trainable, lr=args.lr * world_size, weight_decay=args.weight_decay)
    if args.warmup_epochs > 0:
        warm = torch.optim.lr_scheduler.LinearLR(optimizer, 1e-2, 1.0, args.warmup_epochs)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warm, cos], [args.warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.amp.GradScaler("cuda")
    ema = EMA(core, decay=args.ema_decay)
    best_hd = float("inf")

    for epoch in range(1, args.epochs + 1):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        model.train()
        it = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}") if is_main else train_dl
        for batch in it:
            streams = {k: v.to(device) for k, v in batch["streams"].items()}
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(streams)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update(core)
        scheduler.step()

        # Validation on EMA weights.
        backup = ema.apply_to(core)
        metrics = run_eval(model, val_dl, device, num_classes, idx_to_sp, tree, sp_counts)
        ema.restore(core, backup)

        if is_main:
            print(f"[val {epoch:02d}] macroAcc {metrics['macro_accuracy']:.4f} | "
                  f"HD {metrics['mean_hd']:.4f} | top1 {metrics['top1_accuracy']:.4f} | "
                  f"top5 {metrics.get('top5_accuracy', 0):.4f}")
            if metrics["mean_hd"] < best_hd:
                best_hd = metrics["mean_hd"]
                torch.save({
                    "model": core.state_dict(),
                    "ema": ema.state_dict(),
                    "idx_to_sp": idx_to_sp,
                    "num_classes": num_classes,
                    "model_config": mcfg.__dict__,
                    "benchmark_config": bench.__dict__,   # reconstruct the exact split
                    "args": vars(args),
                }, args.out)
                print(f"  [+] best HD {best_hd:.4f} -> {args.out}")

    if is_dist():
        import torch.distributed as dist
        dist.destroy_process_group()


@torch.no_grad()
def run_eval(model, dl, device, num_classes, idx_to_sp, tree, sp_counts):
    model.eval()
    preds, targets, scores = [], [], []
    for batch in dl:
        streams = {k: v.to(device) for k, v in batch["streams"].items()}
        with torch.amp.autocast("cuda"):
            logits = model(streams)
        prob = torch.softmax(logits, dim=1).float().cpu().numpy()
        scores.append(prob)
        preds.extend(prob.argmax(1).tolist())
        targets.extend(batch["label"].tolist())
    return evaluate_classification(
        np.array(preds), np.array(targets), np.vstack(scores),
        num_classes, idx_to_sp, tree, sp_counts,
    )


if __name__ == "__main__":
    main()
