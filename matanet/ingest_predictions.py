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

    # Reserved index for predictions outside our label set. It maps to a species
    # name that is deliberately ABSENT from the taxonomy tree, so
    # hierarchical_distance returns the maximum (root) penalty with level
    # 'unknown'. The previous approach mapped unknowns to (true+1) % num_classes —
    # a real neighbouring class that could share the true class's genus or family,
    # scoring HD 1 or 2 instead of the maximum 3 and UNDER-penalising the error.
    UNKNOWN_IDX = num_classes
    UNKNOWN_NAME = "__unknown_prediction__"
    idx_to_sp[UNKNOWN_IDX] = UNKNOWN_NAME     # not in the tree -> max HD by construction

    # --- REQUIRE exact one-to-one coverage before scoring anything ----------
    # Skipping unknown IDs or tolerating missing predictions would drop those
    # test items from the DENOMINATOR, silently inflating every C08 metric. A
    # benchmark comparison must score MATANet on every test annotation, so a
    # coverage gap is a hard error, not a warning.
    pred_ids = sub[ann_col].astype(int)
    if pred_ids.duplicated().any():
        dups = sorted(pred_ids[pred_ids.duplicated()].unique())[:5]
        raise SystemExit(f"[ingest] ABORT: duplicate prediction IDs (e.g. {dups}). "
                         "MATANet must emit exactly one prediction per annotation.")
    expected = set(true_by_annid)
    received = set(pred_ids)
    missing = expected - received
    extra = received - expected
    if missing or extra:
        raise SystemExit(
            f"[ingest] ABORT: prediction coverage mismatch — {len(missing)} test "
            f"annotations have no prediction, {len(extra)} predictions reference "
            f"unknown IDs. Scoring the matched subset would inflate C08 by "
            f"dropping the missing items from the denominator.\n"
            f"  missing (examples): {sorted(missing)[:5]}\n"
            f"  extra   (examples): {sorted(extra)[:5]}"
        )

    preds, targets, n_unknown_pred = [], [], 0
    for _, row in sub.iterrows():
        annid = int(row[ann_col])
        pred_name = str(row[pred_col])
        if pred_name not in name_to_idx:      # a predicted name outside our label set
            n_unknown_pred += 1
            pred_idx = UNKNOWN_IDX            # max hierarchical penalty, never a near-miss
        else:
            pred_idx = name_to_idx[pred_name]
        preds.append(pred_idx)
        targets.append(true_by_annid[annid])
    if n_unknown_pred:
        frac = n_unknown_pred / max(1, len(preds))
        print(f"[ingest] warning: {n_unknown_pred} ({frac:.1%}) predicted names "
              "outside the label set (scored at maximum hierarchical distance)")
        # A large fraction almost never means "MATANet is bad" — it means the
        # predicted name STRINGS don't match our binomial label format, so every
        # prediction is being counted wrong. That would silently produce a
        # plausible-looking ~0 score. Fail loudly instead.
        if frac > 0.05:
            unknown_names = sorted({str(r[pred_col]) for _, r in sub.iterrows()
                                    if str(r[pred_col]) not in name_to_idx})[:5]
            raise SystemExit(
                f"[ingest] ABORT: {frac:.1%} of predictions are names we do not "
                f"recognise. This is a label-FORMAT mismatch, not a bad model — "
                f"MATANet is emitting names that differ from our binomial keys.\n"
                f"  ours  (examples): {list(name_to_idx)[:3]}\n"
                f"  theirs(examples): {unknown_names}\n"
                f"Normalise the naming on one side before trusting any C08 number."
            )

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
