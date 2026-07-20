"""
Train any benchmark config (C01-C09, ablations) on the OzFish benchmark, on one
GPU or across several via DDP.

The data split is FIXED across all runs (--split_seed, default 0) so every model
sees the same train/val/test. --seed varies model init + augmentation + sampling
only — run {0,1,2} for the paper's mean +/- std.

--run_id builds the EXACT config run.py uses (any family, incl. the fine-tuned
timm baselines) — this is the multi-GPU path for the whole panel, so heavy
compute-bound runs (C03/C05/C07) can be split across GPUs:

    # single GPU (data paths come from configs/benchmark.yaml)
    python scripts/train.py --run_id C09 --seed 0
    # two GPUs (DDP) — halves a compute-bound fine-tune
    torchrun --nproc_per_node=2 scripts/train.py --run_id C03 --seed 0

Without --run_id the config is assembled from the ablation flags (dino one-offs):
    --backbone dinov2            (A1)
    --context_levels 0           (A2: no MCEAM)   / 1 (A3: ROI-scale)
    --attention_depth 2          (A4)             / 4 (A4b)
    --no_hslm                    (A5: flat softmax)
    --sampler random             (A6)
    --loss ce                    (A7)
    --no_hslm --loss ce          (A7; A8 retired — it duplicated A7)
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
sys.path.insert(0, os.path.dirname(__file__))   # so `from run import ...` resolves under torchrun
from bioreef.config import BenchmarkConfig, DEFAULT_CONFIG_PATH
from bioreef.data import (
    split_from_config, get_taxonomy_tree, build_taxonomy_maps, FishCropDataset,
)
from bioreef.model import build_model, trainable_parameters, ModelConfig, HSLMLoss
from bioreef.run_config import RunConfig
from bioreef.training import set_seed, CBFocalLoss, BalancedDistributedSampler, EMA
from bioreef.eval import evaluate_classification
# Reuse run.py's result writer so train.py and the campaign share one output
# contract (results/<slug>/seed<N>/) and aggregate.py sees every run.
from run import ResultWriter


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Data — default to the paths in the config; these override for a one-off.
    p.add_argument("--csv", default=None, help="override config data.csv_path")
    p.add_argument("--img_dir", default=None, help="override config data.img_dir")
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
    # Config source: a run id builds the EXACT config run.py uses (any family,
    # incl. timm) — this is how DDP covers the whole panel. Omit it to assemble a
    # config from the individual ablation flags below (one-off dino experiments).
    p.add_argument("--run_id", default=None,
                   help="build from configs/runs/<id>_*.yaml (e.g. C03, C09); "
                        "enables DDP for any family. Overrides the model flags below.")
    # Model (ablation flags — used only when --run_id is not given)
    p.add_argument("--model_family", default="dino", choices=["dino", "timm"])
    p.add_argument("--timm_name", default="resnet50", help="timm model id (timm family)")
    p.add_argument("--no_pretrained", action="store_true",
                   help="timm: train from scratch instead of ImageNet weights")
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
    p.add_argument("--epochs", type=int, default=None,
                   help="override the run config's epochs (default: use the config)")
    p.add_argument("--warmup_epochs", type=int, default=None,
                   help="override the run config's warmup (default: use the config)")
    p.add_argument("--batch_size", type=int, default=None,
                   help="override the run config's batch size (default: use the config)")
    p.add_argument("--lr", type=float, default=None,
                   help="override the run config's lr (default: use the config)")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--num_workers", type=int, default=4)
    # Output: by default write the SAME results/<slug>/seed<N>/ tree run.py uses,
    # so DDP runs are picked up by aggregate.py with no test.py hand-off. --out
    # additionally writes a standalone best-HD checkpoint for manual use.
    p.add_argument("--results_dir", default="results",
                   help="write results/<slug>/seed<N>/{metrics.json,configs,checkpoint} here")
    p.add_argument("--out", default=None,
                   help="also write a standalone best-HD checkpoint to this path")
    p.add_argument("--save_checkpoint", action="store_true",
                   help="save checkpoint.pt inside the results dir too")
    p.add_argument("--overwrite", action="store_true",
                   help="re-run even if results/<slug>/seed<N>/metrics.json exists")
    return p.parse_args()


def is_dist():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def build_run_config(args) -> RunConfig:
    """Resolve the RunConfig this invocation trains. --run_id loads the exact
    panel config (any family, incl. timm) so DDP matches run.py; without it, the
    config is assembled from the individual ablation flags (dino one-offs). The
    optimisation budget always comes from the CLI so --epochs/--batch_size/--lr
    keep working either way."""
    if args.run_id:
        rc = RunConfig.find(args.run_id)
    else:
        rc = RunConfig(
            run_id="adhoc", name="", model_family=args.model_family,
            backbone=args.backbone, context_levels=args.context_levels,
            attention_depth=args.attention_depth, unfreeze_blocks=args.unfreeze_blocks,
            timm_name=args.timm_name, pretrained=not args.no_pretrained,
            hslm=not args.no_hslm, loss=args.loss, sampler=args.sampler,
            family_weight=args.family_weight, genus_weight=args.genus_weight,
            species_weight=args.species_weight,
        )
    # CLI optimisation flags override ONLY when explicitly passed (default None).
    # Previously these were argparse defaults (epochs=30), which silently
    # overwrote the run YAML's 60 — so `train.py --run_id C09` trained a 30-epoch
    # model while run.py trained 60, i.e. the DDP path did not reproduce the
    # single-GPU panel. Explicit flags still win; absent flags keep the config.
    if args.epochs is not None:
        rc.epochs = args.epochs
    if args.warmup_epochs is not None:
        rc.warmup_epochs = args.warmup_epochs
    if args.batch_size is not None:
        rc.batch_size = args.batch_size
    if args.lr is not None:
        rc.lr = args.lr
    return rc


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
        csv_path=args.csv,
        img_dir=args.img_dir,
    )
    if not bench.csv_path:
        raise SystemExit("no dataset CSV: set data.csv_path in the config or pass --csv")

    # Fixed benchmark split (driven by the config's data paths).
    train_s, val_s, test_s, num_classes, idx_to_sp, sp_counts = split_from_config(
        bench.csv_path, bench.img_dir, bench,
    )
    tree = get_taxonomy_tree(bench.csv_path)

    # The RunConfig (from --run_id or the ablation flags) is the single source of
    # truth for model family, loss, and sampler — so DDP training matches run.py.
    run_cfg = build_run_config(args)

    # Same banner + skip-check as run.py, so DDP output reads identically.
    writer = ResultWriter(args.results_dir, run_cfg.slug, args.seed)
    if writer.already_done() and not args.overwrite:
        if is_main:
            print(f"[skip] {run_cfg.slug} seed{args.seed} already done ({writer.metrics_path})")
        if is_dist():
            import torch.distributed as dist
            dist.destroy_process_group()
        return
    if is_main:
        print(f"\n{'='*60}\n[run] {run_cfg.slug}  seed={args.seed}  "
              f"family={run_cfg.model_family}\n{'='*60}")
        print(f"[data] {num_classes} species | train {len(train_s)} "
              f"val {len(val_s)} test {len(test_s)}")

    train_ds = FishCropDataset(train_s, is_train=run_cfg.augment)  # augment off = clean crops
    val_ds = FishCropDataset(val_s, is_train=False)
    # Test set is scored once at the end, on rank 0 only.
    test_dl = None
    if is_main:
        test_dl = DataLoader(FishCropDataset(test_s, is_train=False),
                             batch_size=run_cfg.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # Sampler (ablation A6/A8).
    if run_cfg.sampler == "balanced":
        train_sampler = BalancedDistributedSampler(
            train_s, num_replicas=world_size, rank=local_rank, seed=args.seed,
        )
        shuffle = False
    else:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds, shuffle=True) if is_dist() else None
        shuffle = train_sampler is None

    train_dl = DataLoader(train_ds, batch_size=run_cfg.batch_size, sampler=train_sampler,
                          shuffle=shuffle, num_workers=args.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=run_cfg.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = build_model(run_cfg, num_classes).to(device)
    trainable = trainable_parameters(model)

    if is_dist():
        from torch.nn.parallel import DistributedDataParallel as DDP
        # find_unused_parameters=True is required for the dino family (the frozen
        # backbone params get no grad) and is always safe — kept on for every
        # family so an untested timm backbone with any unused param can't crash a
        # long unattended run. The overhead is negligible next to the backbone.
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    core = model.module if is_dist() else model

    # Loss (from run_cfg so --run_id picks the panel's loss; A5/A7/A8 flags).
    if run_cfg.hslm:
        s2g, s2f, n_gen, n_fam, n_missing = build_taxonomy_maps(idx_to_sp, tree)
        if is_main and n_missing:
            print(f"[hslm] {n_missing}/{num_classes} species missing taxonomy -> __unknown__")
        criterion = HSLMLoss(
            sp_counts, s2g, s2f, n_gen, n_fam,
            family_weight=run_cfg.family_weight, genus_weight=run_cfg.genus_weight,
            species_weight=run_cfg.species_weight, beta=args.beta, gamma=args.gamma,
            device=device,
        )
    elif run_cfg.loss == "cbfocal":
        criterion = CBFocalLoss(sp_counts, beta=args.beta, gamma=args.gamma, device=device)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(trainable, lr=run_cfg.lr * world_size, weight_decay=args.weight_decay)
    if run_cfg.warmup_epochs > 0:
        warm = torch.optim.lr_scheduler.LinearLR(optimizer, 1e-2, 1.0, run_cfg.warmup_epochs)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=run_cfg.epochs - run_cfg.warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warm, cos], [run_cfg.warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=run_cfg.epochs)

    scaler = torch.amp.GradScaler("cuda")
    ema = EMA(core, decay=args.ema_decay)
    best_hd, best_val, best_state = float("inf"), None, None

    for epoch in range(1, run_cfg.epochs + 1):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        model.train()
        desc = f"{run_cfg.slug} s{args.seed} ep{epoch}/{run_cfg.epochs}"
        it = tqdm(train_dl, desc=desc) if is_main else train_dl
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

        # Validation on EMA weights; select the best epoch by val mean HD.
        backup = ema.apply_to(core)
        metrics = run_eval(model, val_dl, device, num_classes, idx_to_sp, tree, sp_counts)
        if is_main:
            print(f"[val ep{epoch:02d}] macroAcc {metrics['macro_accuracy']:.4f} "
                  f"HD {metrics['mean_hd']:.4f} top1 {metrics['top1_accuracy']:.4f}")
            if metrics["mean_hd"] < best_hd:
                best_hd, best_val = metrics["mean_hd"], metrics
                # snapshot the EMA weights of the best epoch (CPU copy)
                best_state = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
        ema.restore(core, backup)

    # --- Final: score the held-out TEST set on the best epoch and write the same
    # results/<slug>/seed<N>/ tree run.py produces (rank 0 only) --------------
    if is_main:
        if best_state is not None:
            core.load_state_dict(best_state)
        # Evaluate on the unwrapped module (core), not the DDP wrapper.
        test_metrics = run_eval(core, test_dl, device, num_classes, idx_to_sp, tree, sp_counts)

        result = {
            "run_id": run_cfg.run_id, "slug": run_cfg.slug,
            "model_family": run_cfg.model_family, "seed": args.seed,
            "num_classes": num_classes, "test": test_metrics, "val_best": best_val,
        }
        writer.save(result, run_cfg, bench,
                    model=core if args.save_checkpoint else None, idx_to_sp=idx_to_sp)
        print(f"[done] {run_cfg.slug} seed{args.seed}: "
              f"macroAcc {test_metrics['macro_accuracy']:.4f} | "
              f"HD {test_metrics['mean_hd']:.4f} | top1 {test_metrics['top1_accuracy']:.4f}  "
              f"-> {writer.metrics_path}")

        # Optional standalone checkpoint (manual use / attention figures).
        if args.out:
            ckpt = {
                "model": core.state_dict(), "ema": ema.state_dict(),
                "idx_to_sp": idx_to_sp, "num_classes": num_classes,
                "run_config": run_cfg.__dict__, "benchmark_config": bench.__dict__,
                "args": vars(args),
            }
            if run_cfg.model_family == "dino":
                ckpt["model_config"] = ModelConfig(
                    backbone=run_cfg.backbone, context_levels=run_cfg.context_levels,
                    attention_depth=run_cfg.attention_depth,
                    unfreeze_blocks=run_cfg.unfreeze_blocks,
                ).__dict__
            torch.save(ckpt, args.out)
            print(f"  [+] standalone checkpoint -> {args.out}")

    if is_dist():
        import torch.distributed as dist
        # Non-zero ranks wait here while rank 0 runs the final test eval + write,
        # so all ranks tear down together (avoids an NCCL teardown race).
        dist.barrier()
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
