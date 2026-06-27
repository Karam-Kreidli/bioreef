# BioReef-Classify

A taxonomy-aware, long-tail-robust benchmark and method for **fine-grained
reef-fish classification** on OzFish.

Given a fish crop (bounding box + frame), predict its species, using frozen
vision-foundation features, multi-context attention, and a hierarchy-aware loss.
This repository accompanies the paper and releases the leakage-safe split, the
code, and the trained models.

> **Scope:** classification only. Detection and temporal modeling are out of
> scope (a future paper).

## Why this benchmark

OzFish has ~80k species-labelled crops across a rich family/genus/species
taxonomy, but no published **fine-grained classification** benchmark. The three
difficulties it poses — fine-grained (visually similar species), long-tailed
(few abundant species, a long rare tail), and underwater degradation — are
exactly what this benchmark measures.

The crucial correctness property is the **leakage-safe split**: OzFish crops come
from BRUVS stereo deployments, so the same individual appears across consecutive
frames *and* in both the left/right stereo cameras. A crop-level random split
scatters near-duplicate views of one individual across train/test and inflates
accuracy. We split at the **deployment level** instead — every crop from one
deployment goes entirely into train, val, or test.

### Benchmark definition

The inclusion rules live in [`configs/benchmark.yaml`](configs/benchmark.yaml) —
the single source of truth that every script reads, so train/test/export can
never disagree on what the benchmark is. A species is included iff it is not a
placeholder, has **>=20 crops**, and appears in **>=3 distinct deployments** (the
deployment rule guarantees it can be split across all three folds). On the full
OzFish metadata this yields **296 species / 76,395 crops**. CLI flags
(`--min_samples`, `--min_deployments`, `--split_seed`) override individual config
fields for one-off experiments.

## Repository layout

```
bioreef/
  data/        leakage-safe deployment-grouped split, taxonomy maps,
               context-stream crops, marine augmentation, dataset
  model/       frozen DINOv3 ViT-B/16 backbone -> MCEAM context fusion ->
               classifier head; HSLM marginalization loss; model builder
  training/    unified seeding, CB-Focal loss, balanced sampler, EMA, DDP infra
  eval/        Hierarchical Distance rule + full metric suite
scripts/
  train.py     train the proposed model or any ablation (single-GPU or DDP)
  test.py      evaluate a checkpoint on the held-out test split
  export_split.py  write the released split file from the config
configs/       benchmark.yaml — inclusion rules + split params (source of truth)
tests/         metric, split-leakage, and config unit tests
splits/        released split files (crop -> split, deployment, taxonomy)
```

## Method

- **Backbone.** Frozen **DINOv3 ViT-B/16** (`facebook/dinov3-vitb16-pretrain-lvd1689m`).
  Frozen for parameter efficiency and stability on a small, long-tailed set; an
  optional last-N-block adaptation path is available (`--unfreeze_blocks`).
- **MCEAM.** The ROI `[CLS]` token cross-attends to the patch embeddings of each
  context stream (ROI / social / habitat / full-frame), fused by a gated FFN. A
  lighter single-block variant inspired by MATANet's multi-context attention,
  with a learned gate as the addition.
- **HSLM loss.** The species distribution is *marginalized* up the taxonomy to
  genus/family; per-level NLL with weights family=3 / genus=2 / species=1 trades
  a little species Top-1 for better Hierarchical Distance. This is a
  marginalization mechanism — distinct from MATANet's parallel per-level heads.
- **Long-tail.** CB-Focal species loss + a class-balanced sampler.

## Reproducibility

Every run is seeded through one utility (`bioreef.training.set_seed`) covering
Python, NumPy (augmentation), and PyTorch CPU+CUDA init; the sampler takes the
same seed. The **data split is fixed** (`--split_seed`, default 0) across all
runs — `--seed` varies only model init / augmentation / sampling. Report mean ±
std over seeds {0, 1, 2}.

## Usage

```bash
pip install -r requirements.txt

# Train the proposed model (C09), single GPU
python scripts/train.py --csv frame_metadata.csv --img_dir <frames> --seed 0

# Multi-GPU
torchrun --nproc_per_node=2 scripts/train.py --csv ... --img_dir ... --seed 0

# Evaluate on the held-out test split
python scripts/test.py --csv frame_metadata.csv --img_dir <frames> --weights checkpoint.pt
```

### Ablations (one flag each — see paper Table K.2)

| Flag | Ablation |
| --- | --- |
| `--backbone dinov2` | A1 — DINOv3 → DINOv2 |
| `--context_levels 0` | A2 — no MCEAM (head on pooled ROI) |
| `--context_levels 1` | A3 — ROI-scale only |
| `--attention_depth 2` (or 4) | A4 / A4b — attention depth |
| `--no_hslm` | A5 — flat softmax |
| `--sampler random` | A6 — no balanced sampler |
| `--loss ce` | A7 — plain CE species loss |
| `--no_hslm --loss ce --sampler random` | A8 — all long-tail handling off |

## Metrics

`scripts/test.py` reports the full panel: **macro / class-balanced accuracy**
(headline), **Hierarchical Distance** (mean, plus mistake severity = mean HD over
errors), **head / medium / tail** accuracy (by train frequency), per-level
(family → genus → species) accuracy, and Top-1 / Top-5.

## Tests

```bash
python -m pytest tests/ -v
```

`tests/test_metrics.py` pins every metric to a hand-computed value;
`tests/test_split.py` asserts deployment disjointness and species stratification
on the real metadata.
