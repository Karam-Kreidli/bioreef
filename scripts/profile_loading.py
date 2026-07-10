"""
Diagnose training speed: is a run GPU-starved by CPU data loading, or bound by
the backbone forward? Times one C01-like setup (frozen DINOv3, ROI only) at a few
num_workers settings and reports data-load vs forward seconds per batch.

    python scripts/profile_loading.py --gpu 0

Run this if a frozen-backbone run crawls (<1 it/s). If data >> fwd, the fix is
more workers and/or feature caching; if fwd dominates, it's backbone compute.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from torch.utils.data import DataLoader

from bioreef.config import BenchmarkConfig, DEFAULT_CONFIG_PATH
from bioreef.data import split_from_config, FishCropDataset
from bioreef.model import Classifier, ModelConfig
from bioreef.training import resolve_device


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    p.add_argument("--csv", default=None)
    p.add_argument("--img_dir", default=None)
    p.add_argument("--gpu", default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_batches", type=int, default=5, help="batches to time per setting")
    p.add_argument("--workers", default="4,8,16", help="comma list of num_workers to try")
    p.add_argument("--context_levels", type=int, default=0,
                   help="0 = ROI-only (C01); 3 = full 4-stream harvest (C09 cost)")
    return p.parse_args()


def main():
    args = parse_args()
    bench = BenchmarkConfig.from_yaml(args.config).apply_overrides(
        csv_path=args.csv, img_dir=args.img_dir)
    if not bench.csv_path:
        raise SystemExit("no dataset CSV: set data.csv_path or pass --csv")
    device = resolve_device(args.gpu, bench.device)
    print(f"[device] {device}")

    tr, _va, _te, ncls, _i2s, _spc = split_from_config(bench.csv_path, bench.img_dir, bench)
    print(f"[data] {ncls} species, {len(tr)} train crops")

    model = Classifier(ModelConfig(context_levels=args.context_levels), ncls).to(device).eval()
    use_cuda = device.type == "cuda"

    print(f"\ncontext_levels={args.context_levels}  batch_size={args.batch_size}  "
          f"timing {args.n_batches} batches per setting:")
    for nw in [int(w) for w in args.workers.split(",")]:
        dl = DataLoader(FishCropDataset(tr, is_train=True), batch_size=args.batch_size,
                        shuffle=True, num_workers=nw, pin_memory=use_cuda)
        it = iter(dl)
        next(it)  # warm the worker pool so startup isn't counted

        t0 = time.time()
        batches = [next(it) for _ in range(args.n_batches)]
        t_load = (time.time() - t0) / args.n_batches

        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_cuda):
            for b in batches:
                streams = {k: v.to(device) for k, v in b["streams"].items()}
                _ = model(streams)
        if use_cuda:
            torch.cuda.synchronize()
        t_fwd = (time.time() - t0) / args.n_batches

        bound = "DATA-bound" if t_load > t_fwd else "COMPUTE-bound"
        print(f"  num_workers={nw:2d}: data {t_load:5.2f}s | fwd {t_fwd:5.2f}s "
              f"| ~{1/(t_load+t_fwd):4.1f} it/s  [{bound}]")

    print("\nRead-out: if data >> fwd, the GPU is starving — raise --num_workers "
          "and/or cache features. If fwd dominates, it's backbone compute.")


if __name__ == "__main__":
    main()
