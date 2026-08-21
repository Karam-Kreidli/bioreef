# Run Manifest

The full training campaign. Each config lives in `configs/runs/` and is run with
`python scripts/run.py <id> --seed <N>` (single-GPU). Every paper config runs at
seeds {0,1,2}; the final table is `RESULTS.md` (built by `scripts/aggregate.py`).

**Proposed model = A15** (unfrozen DINOv3 + MCEAM + HSLM + plain CE). C09 is the
**frozen baseline** and the start of the adaptation-depth frontier, not the
proposed model.

Status legend: `[ ]` not started · `[~]` partial · `[x]` all 3 seeds done

## Benchmark panel — baselines + references

| Status | Run | Model | Family | Backbone | Context | Hierarchy | Loss + Sampler |
|---|---|---|---|---|---|---|---|
| [x] | C01 | linear_probe | dino | dinov3 frozen | none | flat | CE + random |
| [x] | C03 | resnet50 | timm | resnet50 (FT) | n/a | flat | CE + random |
| [x] | C04 | efficientnetv2_s | timm | effnetv2-s (FT) | n/a | flat | CE + random |
| [x] | C05 | convnext_tiny | timm | convnext-t (FT) | n/a | flat | CE + random |
| [x] | C06 | vit_base | timm | vit-b/16 (FT) | n/a | flat | CE + random |
| [x] | C07 | swin_base | timm | swin-b (FT) | n/a | flat | CE + random |
| [~] | C08 | matanet | matanet | DINOv2-large (FT, their repo) | native | per-level | native |
| [x] | C09 | proposed (FROZEN REF) | dino | dinov3 frozen | MCEAM 3sc/1blk | HSLM | CB-Focal + random |

C08 = MATANet, run from the official repo on our split (see `matanet/README.md`);
seed 0 done, seeds 1/2 running one at a time on the L40S. All others 3-seed complete.

## Proposed model + ablations

Config-only ablations branch off the **frozen reference C09** (one field changed).
The unfrozen loss chain (A13/A14/A15/A12) and the adaptation-depth frontier
(C09→A9→A10→A11) build toward the proposed unfrozen model **A15**.

| Status | Run | One-factor change / role | Priority |
|---|---|---|---|
| [x] | A1 | C09 backbone DINOv3 → frozen DINOv2-base (frozen backbone-generation comparison) | core |
| [x] | A2 | context off: MCEAM removed (head on pooled ROI) | core |
| [x] | A3 | single context stream (social only) vs all three | core |
| [x] | A4 | attention depth 1 → 2 blocks | core |
| [x] | A5 | attention depth 1 → 4 blocks | optional |
| [x] | A6 | hierarchy off: HSLM → flat softmax (species-only), frozen | core |
| [x] | A7 | sampler random → balanced (frozen) | core |
| [x] | A8 | all long-tail handling off: (CB-Focal) → plain CE, frozen | core |
| [x] | A9 | frozen → last 2 blocks unfrozen (depth frontier) | core |
| [x] | A10 | frozen → last 4 blocks unfrozen | optional |
| [x] | A11 | frozen → FULL fine-tune (all blocks + embeddings), lr 1e-4 | optional |
| [x] | A12 | A11 with lr 1e-4 → 1e-5 (HSLM + CB-Focal, unfrozen) | optional |
| [x] | A13 | unfrozen HSLM off: flat CB-Focal | core |
| [x] | A14 | unfrozen flat plain CE (loss-chain endpoint) | core |
| [x] | **A15** | **PROPOSED: unfrozen HSLM + plain CE (lr 1e-5)** | core |
| [x] | A16 | A15 backbone DINOv3 → DINOv2-base (unfrozen backbone comparison) | optional |

## Deployment (NOT paper-benchmark runs)

| Status | Run | Role | Priority |
|---|---|---|---|
| [~] | D1 | deployment config, CB-Focal (Junior 35-species transfer) — **not a paper run** | optional |
| [~] | D2 | deployment config, plain CE — **not a paper run** | optional |

D1/D2 are the deployment through-line, not part of the benchmark table. The
aggregator should exclude them from the paper table (see AUDIT.md #14 — add a
`--campaign` filter).

## Notes

- **C08 MATANet** runs from the official repo, not `run.py` — see
  `configs/runs/C08_matanet.yaml` and `matanet/README.md`.
- The attention-mass / context-stream analysis is computed from A15's test
  inferences — no separate run.
- Every result JSON records its resolved config + seed; see AUDIT.md for the
  provenance/checksum work still pending before release.
