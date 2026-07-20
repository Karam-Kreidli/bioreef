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
deployment rule guarantees it can be split across all three folds). A species is
keyed by its full binomial (genus + epithet), not the bare epithet — 49 epithets
in OzFish (e.g. *niger*) are shared across multiple genera and would otherwise
collapse distinct species into one class. On the full OzFish metadata this yields
**321 species / 76,083 crops** (39 families, 127 genera). CLI flags
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
- **Long-tail.** CB-Focal species loss. The reference C09 uses the **random**
  sampler; the class-balanced sampler is ablation A7 (it lost on all three
  metrics). Note the two also differ in optimizer steps per epoch — see A7.

## Reproducibility

Every run is seeded through one utility (`bioreef.training.set_seed`) covering
Python, NumPy (augmentation), and PyTorch CPU+CUDA init; the sampler takes the
same seed. The **data split is fixed** (`--split_seed`, default 0) across all
runs — `--seed` varies only model init / augmentation / sampling. Report mean ±
std over seeds {0, 1, 2}.

## Usage

```bash
pip install -r requirements.txt

# Train the proposed model (C09), single GPU. The --run_id is REQUIRED: without
# it train.py builds an ad-hoc config (balanced sampler, 30 epochs) that is NOT
# C09 (random sampler, 60 epochs). Prefer run.py — it is the reference path and
# writes the full results/ tree with provenance.
python scripts/run.py C09 --seed 0

# Multi-GPU
torchrun --nproc_per_node=2 scripts/train.py --csv ... --img_dir ... --seed 0

# Evaluate on the held-out test split
python scripts/test.py --csv frame_metadata.csv --img_dir <frames> --weights checkpoint.pt
```

### The run campaign (recommended path)

The whole benchmark is driven by config files so ~50 runs stay auditable. Each
panel/ablation config is one file in [`configs/runs/`](configs/runs/); run it by
id and it saves its own result + provenance:

```bash
# dataset paths come from configs/benchmark.yaml (data.csv_path / data.img_dir)
python scripts/run.py C09 --seed 0            # one config, one seed
python scripts/run.py C09                     # all seeds in the config (0,1,2)
python scripts/run.py C09 --seed 0 --gpu 1    # pin to GPU 1 (see below)
python scripts/aggregate.py                   # -> RESULTS.md (the benchmark table)
```

**Choosing a GPU:** `--gpu N` (or `--gpu cpu`) on `run.py`/`test.py` overrides the
`device` field in `configs/benchmark.yaml` (empty = auto `cuda:0`). On a
multi-GPU box you can run different configs in parallel from separate shells:
`run.py C09 --gpu 0` and `run.py C03 --gpu 1`.

Each run writes `results/<slug>/seed<N>/` with `metrics.json` (the metric panel),
`run_config.yaml`, and `benchmark_config.yaml` — so a reviewer can open any table
number and see exactly what produced it, then reproduce it with one command.
[`MANIFEST.md`](MANIFEST.md) is the campaign ledger.

**Model families** (dispatched by the config's `model_family`, so config selects
architecture without code edits):
- `dino` — frozen ViT + optional MCEAM. Config-only: C01, C09, and every A* ablation.
- `timm` — a fine-tuned `timm` backbone on the ROI crop: C03/C05/C07.
- `matanet` — run from the [official repo](https://github.com/dhlee-work/fathomnet-cvpr2025-ssl)
  on our split (shared multi-context architecture; not reimplemented here): C08.

`scripts/train.py` / `test.py` remain available for manual single runs and heavy
multi-GPU (DDP) training.

### Ablations (one flag each — see paper Table K.2)

| Flag | Ablation |
| --- | --- |
| `--backbone dinov2` | A1 — DINOv3 → DINOv2 |
| `--context_levels 0` | A2 — no MCEAM (head on pooled ROI) |
| `--context_levels 1` | A3 — single context stream (social only) |
| `--attention_depth 2` (or 4) | A4 / A4b — attention depth |
| `--no_hslm` | A5 — flat softmax |
| `--sampler random` | A6 — no balanced sampler |
| `--loss ce` | A7 — plain CE species loss |
| `--no_hslm --loss ce --sampler random` | A8 — all long-tail handling off |

## Attention figures (qualitative)

`scripts/visualize.py` renders the paper's attention panel from any saved
checkpoint — a **two-part figure per crop**:

- **Backbone saliency** (left): attention rollout over the frozen ViT on the ROI
  crop — *does the backbone localize the fish?*
- **MCEAM cross-attention** (right, one column per context stream): where the
  fish `[CLS]` query attends across social / habitat / full-frame — *what
  environmental context does MCEAM pool?*

```bash
python scripts/visualize.py --weights results/C09_proposed/seed0/checkpoint.pt \
    --n 8 --out_dir figures/attention
```

This is a visualization tool only — it never touches the benchmark numbers and
needs no re-runs; generate figures after training. The two maps are distinct:
rollout is backbone self-attention (pixels that make the fish); the MCEAM maps
are the module's cross-attention (which context patches it fused). The A2
ablation (`context_levels=0`) has no MCEAM, so only the backbone column renders.

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
