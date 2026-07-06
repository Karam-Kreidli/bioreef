"""
Ingest MATANet's predictions into OUR metric harness, so C08 lands in the same
RESULTS.md table as every other run.

    python matanet/ingest_predictions.py --data_dir matanet/ozfish_data --seed 0

Reads:
    <data_dir>/dataset_test.json     the true test labels (our split)
    <data_dir>/dataset_train.json    train counts (for head/medium/tail grouping)
    <data_dir>/predictions_seed<N>.csv   MATANet's [annotation_id, concept_name]
Writes:
    results/C08_matanet/seed<N>/metrics.json   (same schema as scripts/run.py)

Note: MATANet's submission is argmax species names only (no probabilities), so
Top-5 is not computed for C08 — the priority metrics (macro accuracy, HD, mistake
severity, head/medium/tail, per-level) all come from the argmax predictions.
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bioreef.config import BenchmarkConfig, DEFAULT_CONFIG_PATH
from bioreef.data import get_taxonomy_tree
from bioreef.eval import evaluate_classification


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default="matanet/ozfish_data",
                   help="dir with dataset_{train,test}.json + predictions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                   help="benchmark config (for the CSV -> taxonomy tree)")
    p.add_argument("--csv", default=None, help="override config data.csv_path")
    p.add_argument("--results_dir", default="results")
    return p.parse_args()


def main():
    args = parse_args()
    bench = BenchmarkConfig.from_yaml(args.config).apply_overrides(csv_path=args.csv)
    if not bench.csv_path:
        raise SystemExit("no dataset CSV: set data.csv_path in the config or pass --csv")

    test = json.load(open(os.path.join(args.data_dir, "dataset_test.json")))
    train = json.load(open(os.path.join(args.data_dir, "dataset_train.json")))
    pred_csv = os.path.join(args.data_dir, f"predictions_seed{args.seed}.csv")
    if not os.path.exists(pred_csv):
        raise SystemExit(f"predictions not found: {pred_csv} — run MATANet's C1 first")

    # category_id (1-based in JSON) -> 0-based class idx; and idx -> species name.
    idx_to_sp = {c["id"] - 1: c["name"] for c in test["categories"]}
    name_to_idx = {v: k for k, v in idx_to_sp.items()}
    num_classes = len(idx_to_sp)

    # True labels per annotation id (from our test split).
    true_by_annid = {a["id"]: a["category_id"] - 1 for a in test["annotations"]}

    # Train counts per class -> head/medium/tail grouping.
    sp_counts = [0] * num_classes
    for a in train["annotations"]:
        sp_counts[a["category_id"] - 1] += 1
    sp_counts = [max(1, c) for c in sp_counts]

    # MATANet predictions: annotation_id -> predicted species name.
    import pandas as pd
    sub = pd.read_csv(pred_csv)
    # tolerate either header naming
    ann_col = "annotation_id" if "annotation_id" in sub.columns else sub.columns[0]
    pred_col = "concept_name" if "concept_name" in sub.columns else sub.columns[1]

    preds, targets, n_unknown_pred = [], [], 0
    for _, row in sub.iterrows():
        annid = int(row[ann_col])
        if annid not in true_by_annid:
            continue
        pred_name = str(row[pred_col])
        if pred_name not in name_to_idx:      # a predicted name outside our label set
            n_unknown_pred += 1
            # map to a sentinel wrong index (guaranteed != target) so it counts as error
            pred_idx = (true_by_annid[annid] + 1) % num_classes
        else:
            pred_idx = name_to_idx[pred_name]
        preds.append(pred_idx)
        targets.append(true_by_annid[annid])

    if len(targets) != len(test["annotations"]):
        print(f"[ingest] warning: {len(targets)} predictions vs "
              f"{len(test['annotations'])} test annotations (some missing)")
    if n_unknown_pred:
        print(f"[ingest] warning: {n_unknown_pred} predicted names outside the "
              "label set (counted as errors)")

    tree = get_taxonomy_tree(bench.csv_path)
    m = evaluate_classification(
        np.array(preds), np.array(targets), None,      # no scores -> no Top-5
        num_classes, idx_to_sp, tree, sp_counts,
    )

    out_dir = os.path.join(args.results_dir, "C08_matanet", f"seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    result = {
        "run_id": "C08", "slug": "C08_matanet", "model_family": "matanet",
        "seed": args.seed, "num_classes": num_classes, "test": m, "val_best": None,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"[ingest] C08 seed{args.seed}: macroAcc {m['macro_accuracy']:.4f} | "
          f"HD {m['mean_hd']:.4f} | top1 {m['top1_accuracy']:.4f}  -> {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
