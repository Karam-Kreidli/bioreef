"""
Evaluate a trained checkpoint on the held-out TEST split and print the full
paper metric panel (macro acc, HD, head/medium/tail, per-level, mistake
severity, top-1/5). Writes per-species accuracy and confused-pairs CSVs.

The model is rebuilt from the checkpoint's stored ModelConfig, and the split is
reconstructed with the checkpoint's split_seed, so test matches train exactly.

    python scripts/test.py --csv frame_metadata.csv --img_dir <frames> \
        --weights checkpoint.pt
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bioreef.config import BenchmarkConfig
from bioreef.data import split_from_config, get_taxonomy_tree, FishCropDataset
from bioreef.model import Classifier, ModelConfig
from bioreef.training import set_seed, resolve_device
from bioreef.eval import evaluate_classification, per_class_accuracy


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=None, help="override checkpoint/config data path")
    p.add_argument("--img_dir", default=None, help="override checkpoint/config data path")
    p.add_argument("--weights", required=True)
    p.add_argument("--gpu", default=None,
                   help="GPU to use, e.g. 1 or cuda:1 or cpu (overrides config 'device')")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out_dir", default="eval_out")
    p.add_argument("--use_ema", action="store_true", help="evaluate EMA weights")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    # test.py evaluates train.py checkpoints (which carry these keys). run.py
    # writes a different format and does its own eval inline — give a clear error
    # rather than a cryptic KeyError if the wrong checkpoint is passed.
    missing = [k for k in ("args", "model_config", "benchmark_config",
                           "idx_to_sp", "num_classes", "model") if k not in ckpt]
    if missing:
        raise SystemExit(
            f"{args.weights} is not a train.py checkpoint (missing {missing}). "
            "run.py checkpoints are evaluated inline by run.py, not test.py."
        )
    saved = ckpt["args"]
    mcfg = ModelConfig(**ckpt["model_config"])
    idx_to_sp = {int(k): v for k, v in ckpt["idx_to_sp"].items()}
    num_classes = ckpt["num_classes"]

    set_seed(saved.get("seed", 0))
    # Reconstruct the SAME split from the checkpoint's stored benchmark config;
    # dataset paths come from the checkpoint unless overridden on the CLI.
    bench = BenchmarkConfig(**ckpt["benchmark_config"]).apply_overrides(
        csv_path=args.csv, img_dir=args.img_dir,
    )
    if not bench.csv_path:
        raise SystemExit("no dataset CSV: the checkpoint stored none; pass --csv")
    device = resolve_device(args.gpu, bench.device)   # --gpu > config 'device' > auto
    print(f"[config] {bench}")
    print(f"[device] {device}")
    _train, _val, test_s, n_split, _c2s, sp_counts = split_from_config(
        bench.csv_path, bench.img_dir, bench,
    )
    assert n_split == num_classes, (
        f"split class count {n_split} != checkpoint {num_classes} — "
        "the CSV/img_dir does not match the one trained on."
    )
    print(f"[test] {len(test_s)} crops, {num_classes} species")

    model = Classifier(mcfg, num_classes).to(device)
    model.load_state_dict(ckpt["model"])
    if args.use_ema and "ema" in ckpt:
        # Overlay EMA shadow onto matching params.
        sd = model.state_dict()
        for k, v in ckpt["ema"].items():
            if k in sd:
                sd[k].copy_(v)
        print("[test] using EMA weights")
    model.eval()

    test_dl = DataLoader(FishCropDataset(test_s, is_train=False),
                         batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    preds, targets, scores = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="eval"):
            streams = {k: v.to(device) for k, v in batch["streams"].items()}
            with torch.amp.autocast("cuda"):
                logits = model(streams)
            prob = torch.softmax(logits, dim=1).float().cpu().numpy()
            scores.append(prob)
            preds.extend(prob.argmax(1).tolist())
            targets.extend(batch["label"].tolist())

    preds = np.array(preds)
    targets = np.array(targets)
    scores = np.vstack(scores)
    tree = get_taxonomy_tree(bench.csv_path)

    m = evaluate_classification(preds, targets, scores, num_classes, idx_to_sp, tree, sp_counts)

    print("=" * 60)
    print("TEST METRICS (held-out, deployment-grouped split)")
    print("=" * 60)
    print(f"  Macro accuracy      : {m['macro_accuracy']:.4f}   <- headline")
    print(f"  Top-1 accuracy      : {m['top1_accuracy']:.4f}")
    print(f"  Top-5 accuracy      : {m.get('top5_accuracy', 0):.4f}")
    print(f"  Mean HD             : {m['mean_hd']:.4f}")
    print(f"  Mistake severity    : {m['mistake_severity']:.4f}   (mean HD over errors)")
    print(f"  Genus accuracy      : {m['genus_accuracy']:.4f}")
    print(f"  Family accuracy     : {m['family_accuracy']:.4f}")
    print(f"  Cross-family errors : {m['cross_family_error_rate']:.4f}")
    ga, gs = m["group_accuracy"], m["group_sizes"]
    print(f"  Head/Med/Tail acc   : {ga['head']:.4f} / {ga['medium']:.4f} / {ga['tail']:.4f}"
          f"   (n={gs['head']}/{gs['medium']}/{gs['tail']})")
    print("=" * 60)

    # Persist the metric panel as JSON (one row of the benchmark table).
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)

    # Per-species accuracy CSV.
    pc = per_class_accuracy(preds, targets, num_classes)
    support = Counter(targets.tolist())
    with open(os.path.join(args.out_dir, "per_species.csv"), "w", encoding="utf-8") as f:
        f.write("species,accuracy,support\n")
        for c in range(num_classes):
            if pc[c] is not None:
                f.write(f"{idx_to_sp[c]},{pc[c]:.4f},{support[c]}\n")

    print(f"wrote {args.out_dir}/metrics.json and per_species.csv")


if __name__ == "__main__":
    main()
