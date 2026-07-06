"""
Export the leakage-safe split as a released artifact (paper Section 3.6).

Writes splits/ozfish_split_seed<N>.csv with one row per crop:
    file_name, deployment, split, family, genus, species

This is the benchmark's reproducibility contribution — it pins the exact set of
crops in each fold, independent of which machine the frames live on. Runs against
the metadata CSV only (no images needed).

    python scripts/export_split.py --csv frame_metadata.csv --split_seed 0
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import bioreef.data.split as split_mod
from bioreef.config import BenchmarkConfig, DEFAULT_CONFIG_PATH
from bioreef.data.split import split_from_config
from bioreef.data.taxonomy import get_taxonomy_tree


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                   help="benchmark config YAML (inclusion rules + split params)")
    p.add_argument("--csv", default=None, help="override config data.csv_path")
    p.add_argument("--min_samples", type=int, default=None, help="override config")
    p.add_argument("--min_deployments", type=int, default=None, help="override config")
    p.add_argument("--split_seed", type=int, default=None, help="override config")
    p.add_argument("--out_dir", default="splits")
    p.add_argument("--require_images", action="store_true",
                   help="only include crops whose image is on disk (default: "
                        "include all CSV rows, since the split file is about the "
                        "label set, not local file availability)")
    p.add_argument("--img_dir", default=None, help="override config data.img_dir "
                                                    "(used with --require_images)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    bench = BenchmarkConfig.from_yaml(args.config).apply_overrides(
        min_samples=args.min_samples,
        min_deployments=args.min_deployments,
        split_seed=args.split_seed,
        csv_path=args.csv,
        img_dir=args.img_dir,
    )
    if not bench.csv_path:
        raise SystemExit("no dataset CSV: set data.csv_path in the config or pass --csv")
    print(f"[config] {bench}")
    tree = get_taxonomy_tree(bench.csv_path)

    real_exists = os.path.exists
    if not args.require_images:
        # Treat every referenced image as present so the released split reflects
        # the full labelled set, not one machine's local frames.
        split_mod.os.path.exists = lambda p: True if str(p).endswith(".png") else real_exists(p)
    try:
        # img_dir + extra_img_dirs both come from bench (split_from_config reads
        # cfg.extra_img_dirs), so multi-folder datasets are handled from config.
        train, val, test, num_classes, _c2s, _spc = split_from_config(
            bench.csv_path, bench.img_dir, bench,
        )
    finally:
        split_mod.os.path.exists = real_exists

    rows = []
    for fold, samples in (("train", train), ("val", val), ("test", test)):
        for s in samples:
            fn = os.path.basename(s["img_path"])
            tax = tree.get(s["species"], {})
            rows.append((fn, s["deployment"], fold,
                         tax.get("family", ""), tax.get("genus", ""), s["species"]))

    out_path = os.path.join(args.out_dir, f"ozfish_split_seed{bench.split_seed}.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("file_name,deployment,split,family,genus,species\n")
        for r in rows:
            f.write(",".join(r) + "\n")

    print(f"wrote {len(rows)} crops across {num_classes} species -> {out_path}")
    for fold in ("train", "val", "test"):
        n = sum(1 for r in rows if r[2] == fold)
        print(f"  {fold:5s}: {n}")


if __name__ == "__main__":
    main()
