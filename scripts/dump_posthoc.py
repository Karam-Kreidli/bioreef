"""
Dump penultimate FEATURES + classifier HEAD + LOGITS for a run.py checkpoint so
the tail post-hoc sweep (tau-normalization, logit adjustment) can be computed
OFFLINE on the EXACT trained seed — no retraining. One forward pass over the
held-out TEST split.

Run this where the backbone is available (the VM/cluster the campaign trained
on) — building the DINOv3 model instantiates the backbone. The companion
`tail_posthoc_sweep.py` then runs anywhere (pure numpy, no backbone/GPU).

    python scripts/dump_posthoc.py --seed_dir results/A15_hslm_ce_unfrozen/seed0

Requires the seed dir to contain `checkpoint.pt` (i.e. the campaign run used
`--save_checkpoint`). Writes, into the same seed dir:
    posthoc_dump.npz   features/logits/targets/head_W/head_b/sp_counts/num_classes
    posthoc_meta.pkl   idx_to_sp, taxonomy_tree  (objects the metric fns need)

Sanity check printed at the end: the baseline Top-1 recomputed from the dumped
logits should match `metrics.json`'s test.top1_accuracy for this seed. If it
does, the split + weights were reconstructed correctly and the sweep is valid.
"""

import argparse
import os
import pickle
import sys
from dataclasses import fields

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bioreef.config import BenchmarkConfig
from bioreef.run_config import RunConfig
from bioreef.data import split_from_config, get_taxonomy_tree, FishCropDataset
from bioreef.model import build_model
from bioreef.training import set_seed, resolve_device


def _from_dict(cls, d):
    """Build a dataclass from a saved __dict__, ignoring any non-init keys."""
    valid = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in valid})


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed_dir", required=True,
                   help="results/<slug>/seed<N>/  (must contain checkpoint.pt)")
    p.add_argument("--csv", default=None, help="override dataset CSV path")
    p.add_argument("--img_dir", default=None, help="override frames dir")
    p.add_argument("--gpu", default=None, help="e.g. 0 | cuda:1 | cpu")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_path = os.path.join(args.seed_dir, "checkpoint.pt")
    if not os.path.exists(ckpt_path):
        raise SystemExit(
            f"no checkpoint.pt in {args.seed_dir} — the campaign run must have "
            "been launched with --save_checkpoint for this to exist.")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for k in ("model", "run_config", "benchmark_config", "idx_to_sp"):
        if k not in ckpt:
            raise SystemExit(f"{ckpt_path} missing key '{k}' — not a run.py "
                             "checkpoint (see scripts/run.py ResultWriter.save).")

    run_cfg = _from_dict(RunConfig, ckpt["run_config"])
    bench = _from_dict(BenchmarkConfig, ckpt["benchmark_config"]).apply_overrides(
        csv_path=args.csv, img_dir=args.img_dir)
    if not bench.csv_path:
        raise SystemExit("checkpoint stored no dataset CSV; pass --csv/--img_dir")

    device = resolve_device(args.gpu, bench.device)
    set_seed(0)  # eval is deterministic; the split is reconstructed from `bench`
    print(f"[config] {run_cfg.slug}  seed_dir={args.seed_dir}")
    print(f"[device] {device}")

    _tr, _va, test_s, n_split, _c2s, sp_counts = split_from_config(
        bench.csv_path, bench.img_dir, bench)
    idx_to_sp = {int(k): v for k, v in ckpt["idx_to_sp"].items()}
    num_classes = len(idx_to_sp) if idx_to_sp else n_split
    if n_split != num_classes:
        raise SystemExit(
            f"split class count {n_split} != checkpoint {num_classes} — the "
            "CSV/img_dir does not match what was trained on.")
    print(f"[test] {len(test_s)} crops, {num_classes} species")

    model = build_model(run_cfg, num_classes).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dl = DataLoader(FishCropDataset(test_s, is_train=False),
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    feats, logits, targets = [], [], []
    with torch.no_grad():
        for b in tqdm(dl, desc="dump"):
            streams = {k: v.to(device) for k, v in b["streams"].items()}
            with torch.amp.autocast("cuda"):
                f = model.embed(streams)      # (B, D) penultimate — for tau-norm
                lg = model.head(f)            # (B, C) logits — for logit-adjust
            feats.append(f.float().cpu().numpy())
            logits.append(lg.float().cpu().numpy())
            targets.extend(b["label"].tolist())

    feats = np.vstack(feats)
    logits = np.vstack(logits)
    targets = np.asarray(targets)
    W = model.head.weight.detach().cpu().numpy()                      # (C, D)
    bvec = (model.head.bias.detach().cpu().numpy()
            if model.head.bias is not None else np.zeros(W.shape[0]))  # (C,)

    out_npz = os.path.join(args.seed_dir, "posthoc_dump.npz")
    np.savez_compressed(
        out_npz, features=feats, logits=logits, targets=targets,
        head_W=W, head_b=bvec, sp_counts=np.asarray(sp_counts),
        num_classes=num_classes)
    tree = get_taxonomy_tree(bench.csv_path)
    with open(os.path.join(args.seed_dir, "posthoc_meta.pkl"), "wb") as fh:
        pickle.dump({"idx_to_sp": idx_to_sp, "taxonomy_tree": tree}, fh)

    top1 = float((logits.argmax(1) == targets).mean())
    print(f"[ok] dumped {len(targets)} test samples -> {out_npz}")
    print(f"[sanity] baseline Top-1 from dumped logits = {top1:.4f}")
    print("         compare to metrics.json test.top1_accuracy for this seed; "
          "they should match.")


if __name__ == "__main__":
    main()
