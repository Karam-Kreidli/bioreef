# Run Manifest

The full training campaign. Each config lives in `configs/runs/` and is run
with `python scripts/run.py <id> --seed <N>`. Every config runs at seeds
{0,1,2}; the final table is `RESULTS.md` (built by `scripts/aggregate.py`).

Status legend: `[ ]` not started · `[~]` partial · `[x]` all seeds done

## K.1 — Benchmark panel (standalone models)

| Status | Run | Model | Family | Backbone | Context | Hierarchy | Loss+Sampler |
|---|---|---|---|---|---|---|---|
| [ ] | C01 | linear_probe | dino | dinov3 frozen | none | flat | CE+ran |
| [ ] | C03 | resnet50 | timm | resnet50 | n/a | flat | — |
| [ ] | C05 | convnext_tiny | timm | convnext_tiny | n/a | flat | — |
| [ ] | C07 | swin_base | timm | swin_base_patch4_window7_224 | n/a | flat | — |
| [ ] | C08 | matanet | matanet | DINOv2 (their repo) | n/a | per-level | — |
| [ ] | C09 | proposed | dino | dinov3 frozen | MCEAM 3sc/1blk | HSLM | CB-Focal+bal |

## K.2 — Ablations (one-factor changes to C09)

| Status | Run | Changes vs C09 | Priority |
|---|---|---|---|
| [ ] | A1 | C09 with the backbone swapped to frozen DINOv2-base. Disentangles backbone generation from architecture in the MATANet comparison. | core |
| [ ] | A2 | context off: MCEAM removed (head on pooled ROI). Keeps HSLM+CB-Focal+balanced. | core |
| [ ] | A3 | single context stream (social only) vs all three. | core |
| [ ] | A4 | attention depth 1 -> 2 blocks. | core |
| [ ] | A4b | attention depth 1 -> 4 blocks. | optional |
| [ ] | A5 | hierarchy off: HSLM -> flat softmax (species-only). | core |
| [ ] | A6 | sampler balanced -> random. | core |
| [ ] | A7 | species loss CB-Focal -> plain CE (HSLM off so the species term is CE). | core |
| [ ] | A8 | all long-tail handling off: (CB-Focal+balanced) -> (CE+random). | core |
| [ ] | A9 | stream contribution: all context -> ROI-only anchor (context_levels 0). | core |

## Notes

- **A2 and A9 are the same model** (context off / ROI-only anchor); both ids are
  kept for traceability to the paper's K.2 table but produce identical results.
- **C08 MATANet** runs from the official repo, not `run.py` — see
  `configs/runs/C08_matanet.yaml` and `bioreef/model/factory.py::build_matanet`.
- The attention-mass readout (context-stream analysis) is computed from C09's
  test inferences — no separate run.
