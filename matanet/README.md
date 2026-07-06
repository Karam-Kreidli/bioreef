# C08 — MATANet on the OzFish benchmark

MATANet is the closest prior work and shares this project's multi-context
architecture, so the fair comparison is to run **their** official model on **our**
leakage-safe split — not to reimplement it. This folder is the bridge.

- **Upstream:** https://github.com/dhlee-work/fathomnet-cvpr2025-ssl (MIT license,
  "Smartsystem Lab"), pinned commit `922c2176893ef1d03de8b8701cd882b5764f9ae9`.
- **What we change:** only hardcoded FathomNet paths (via `patch_matanet.py`). The
  model, MCEAM, and hierarchical loss are untouched.
- **Backbone:** DINOv2-large, fine-tuned (as published) — this is the paper's
  stated fairness caveat vs our frozen DINOv3-base.
- **Taxonomy:** 3 levels (Family/Genus/Species), matching our HSLM exactly, so
  both methods get identical hierarchical supervision.

## What the bridge does

| Script | Role |
| --- | --- |
| `export_ozfish.py` | Our split → MATANet inputs (COCO JSONs, `hierarchical_label.csv`, distance matrix, label-encoder) **+ the OzFish config**. The split is baked into the JSONs (train = our train, test = our test), so their random-split code never runs and test is never seen in training. |
| `patch_matanet.py` | Redirects their 3 hardcoded FathomNet paths to read from config; removes a crashing debug tail in `C1`. Idempotent. |
| `ingest_predictions.py` | Their predictions → **our** metric harness → `results/C08_matanet/seed<N>/metrics.json` (same schema as `run.py`, so it lands in `RESULTS.md`). |

## Full workflow (on the VM, with a GPU)

```bash
# 0. clone MATANet at the pinned commit, next to this repo
git clone https://github.com/dhlee-work/fathomnet-cvpr2025-ssl.git ../matanet-repo
cd ../matanet-repo && git checkout 922c2176893ef1d03de8b8701cd882b5764f9ae9 && cd -
pip install -r ../matanet-repo/requirements.txt      # their deps (pl, transformers, ...)

# 1. patch their hardcoded paths (once)
python matanet/patch_matanet.py --repo ../matanet-repo

# 2. export our split into MATANet's format (per seed; writes the config too)
python matanet/export_ozfish.py --seed 0 --out_dir matanet/ozfish_data

# 3. train + test MATANet from THEIR repo, on our data
cd ../matanet-repo
python B1.BuildModel.py --config ../bioreef-classify/matanet/ozfish_data/ozfish_config_seed0.yaml
python C1.TestModel.py  --config ../bioreef-classify/matanet/ozfish_data/ozfish_config_seed0.yaml
cd -

# 4. ingest predictions into our metrics -> results/C08_matanet/seed0/
python matanet/ingest_predictions.py --data_dir matanet/ozfish_data --seed 0

# repeat 2-4 for seeds 1 and 2, then:
python scripts/aggregate.py                          # C08 now in RESULTS.md
```

## Notes

- **Metrics:** MATANet's submission is argmax species names (no probabilities), so
  **Top-5 is not reported for C08**. All priority metrics — macro accuracy, HD,
  mistake severity, head/medium/tail, per-level — come from the argmax predictions.
- **Preprocessing:** MATANet uses its own native augmentation (downsampling +
  colour jitter, as published). This is the standard way to report a baseline
  "from the official repo"; noted as a caveat rather than stripped (stripping it
  would alter their method).
- **Claims discipline:** report C08 as "MATANet, official repo, run on the OzFish
  split." Do not quote their FathomNet numbers.
- `matanet/ozfish_data/` is generated (gitignored) — regenerate with step 2.
