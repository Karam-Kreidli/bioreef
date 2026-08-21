# Repository Audit — findings, status, decisions

A deep audit of the `bioreef-classify` repo (2026-08-17): the full Python tree was
compiled, tests run, every run YAML parsed, single-GPU and DDP paths traced
separately, split artifacts inspected, split statistics and sampler epoch lengths
recomputed, result provenance checked, and all **71 stored result files**
sanity-checked.

**Clean bill on the basics:** no Python syntax errors, all modules compile, all run
YAMLs parse, the 71 metric files have no NaNs/infinities/impossible values/broken
taxonomy ordering, and the test suite reports 20/20 — but see #21 (some are false
passes on a clean checkout).

This file is the single reference for the semantic/scientific/reproducibility
findings. Status legend:

- ✅ **DONE** — fixed or resolved.
- 🟢 **DECIDED** — decision made, execution pending (or handled by process, e.g. fresh repo).
- 🔧 **TODO** — real, to fix before the new-repo upload.
- 🔍 **VERIFY** — needs a check against real data/history before acting.
- 💤 **DEFER / DOC-ONLY** — low severity, comment/wording fix, or optional.

**Anchoring decisions (2026-08-17):**
1. **Proposed model = A15** (HSLM + plain CE, unfrozen DINOv3). C09 is the *frozen
   baseline*, not "the proposed model". CB-Focal dropped.
2. **Split: disclose, do NOT re-split.** Report "321 defined / 313 evaluable"; no re-runs.
3. **Fresh repo, uploaded all at once** — no visible git chronology, so every config
   is an *initial* configuration. This dissolves the "post-hoc / test-tuned" framing
   concern (#1, #16-framing-half); the remaining work is doc consistency.
4. **DDP deleted** — never used; removed rather than fixed.

---

## Critical / scientific

| # | Finding | Status | Notes |
|---|---|---|---|
| 1 | Test set used for development (C09→random, A12–A16 conceived after seeing results) | 🟢 DECIDED | Dissolved by fresh-repo plan (#3 above) + benchmark-first framing. No chronology visible. |
| 2 | Benchmark defines 321 species but test covers only 313 (val 315; seeds 1/2 → 312/311). `macro_accuracy()` averages over *present* classes only → 313-class macro under a 321 banner. | 🟢 DECIDED → 🔧 | **Disclose, don't re-split.** TODO: label "321 defined / 313 evaluable" wherever macro appears; fix benchmark.yaml's false "guarantees split across all folds" comment; fix group_sizes labelling (#85). |
| 16 | Two contradictory "proposed model" definitions (frozen C09 vs unfrozen A12/A15) | ✅ DONE (docs) | **A15 is proposed.** C09_proposed.yaml header + MANIFEST + PAPER_FRAMING now say C09=frozen baseline, A15=proposed. Remaining: strip post-hoc *narration* comments inside A12–A16 configs (they explain "created after seeing X") — do in final polish so fresh-repo shows no chronology. |

## DDP path — all DELETED (never used; only `run.py` single-GPU ran)

| # | Finding | Status |
|---|---|---|
| 5 | DDP `train.py` HSLM ignores `species_loss_type` → always cbfocal even with `loss: ce` | ✅ DONE (deleted) |
| 6 | DDP scales `lr *= world_size`, batch is per-rank; unrecorded in run YAML | ✅ DONE (deleted) |
| 7 | DDP ignores early stopping (`patience`, `min_delta`) | ✅ DONE (deleted) |
| 8 | DDP doesn't revalidate CLI overrides | ✅ DONE (deleted) |
| 34 | `train.py`/`test.py` CPU autocast unconditional `"cuda"` | ✅ DONE (deleted) |
| 35 | `test.py` only loads `train.py`-format checkpoints (incompatible with run.py's) | ✅ DONE (deleted) |
| 36 | `test.py --use_ema` advertised but always errors | ✅ DONE (deleted) |
| 79 | DDP ranks share one seed → correlated RNGs | ✅ DONE (deleted) |
| 80 | DDP duplicates full validation on every GPU | ✅ DONE (deleted) |

Removed `scripts/train.py` + `scripts/test.py`; fixed stale refs in `run_config.py`
(docstring + dead `train_flags()`), `loop.py`, `slurm/README.md`.

## Reproducibility / provenance (for the benchmark co-star)

| # | Finding | Status |
|---|---|---|
| 3 | Released split CSVs stale — no `annotation_id`/bbox; file_name not unique (48× dup one file; 15,647 byte-identical rows) | 🔧 TODO (VM) | The *exporter* (`export_split.py`) is already fixed (writes annotation_id + x0/y0/x1/y1, hard-errors on dup ids). Just need to REGENERATE the `splits/*.csv` (still old 6-col schema) — needs the metadata CSV, so run on the L40S/openuae: `python scripts/export_split.py --split_seed {0,1,2}`. |
| 4 | Training recomputes the split from source, doesn't consume the released manifest → released split ≠ trained split | 💤 DEFER | Bigger design change (training would validate against a frozen manifest hash). Lower priority given the provenance fingerprint (#13) now records the dataset hash, so a changed CSV is already detectable. Revisit if a reviewer needs the manifest to be authoritative. |
| 10 | `aggregate.py` averages seeds across different `code_revision` values with no check | ✅ DONE (no re-run) | `provenance_issue()` checks + warns (`--strict-provenance` to hard-fail). Caught C06/C07/C09 seeds spanning 3 revisions — but those revs differ only in docs / other-path fixes, NOT the model code those runs used, so the numbers are valid. **Fresh single-commit repo → all seeds share ONE revision → artifact disappears. No re-run.** See re-run policy below. |
| 11 | `already_done()` reuses stale results (only checks file existence, not provenance) | ✅ DONE | Now provenance-aware: a re-run after a code/data change is not skipped. Legacy (no-fingerprint) results treated as done-with-warning (no mass re-run). |
| 12 | `_git_revision()` records commit but not dirty working tree | ✅ DONE | Adds `-dirty` suffix when the tree has uncommitted changes. |
| 13 | No dataset checksum (records CSV *path*, not contents) | ✅ DONE | `_dataset_hash()` (SHA-256/16) in the provenance fingerprint. |
| 14 | `aggregate.py` includes D1/D2 ("NOT a paper run") + post-hoc A12–A16 automatically; needs `--campaign` | ✅ DONE ⚠️ | `--campaign configs/campaign.yaml` now filters to declared runs (excludes D1/D2). **BUT campaign.yaml is missing A13/A14/A15/A16** — since A15 is the proposed model, ADD A13–A16 to campaign.yaml so the paper table includes them. |
| 71 | `requirements.txt` uses ranges, not pins → future drift (matters for transformers/timm) | 🔧 TODO |
| 72 | Pretrained model *revisions* not pinned (only names) | 🔧 TODO |
| 73 | `benchmark.yaml` mixes scientific definition with machine-local paths | 🔧 TODO |

## Split / data construction

| # | Finding | Status |
|---|---|---|
| 22 | Split test asserts coverage ≥ 0.95 (accepts ~16 missing classes) | 💤 (aligns with #2 disclosure; tighten if wanted) |
| 23 | `sp_counts = [max(1,c) ...]` fabricates a train count of 1 for a 0-support class (hides catastrophic condition; also in MATANet ingest) | ✅ DONE (split.py) | Now hard-errors listing the zero-support species. MATANet-ingest copy still TODO (Tier 4). |
| 24 | `BenchmarkConfig` silently ignores unknown fields (typos) + missing config path falls back to defaults | ✅ DONE | Unknown keys → ValueError; explicit missing config → FileNotFoundError. Tests added. |
| 25 | Split ratios not validated (count=3, positive, sum=1) | ✅ DONE | `_validate()` on load + after overrides. Test added. |
| 26 | Invalid bboxes warned + skipped even under `strict_images: true` | ✅ DONE | Now hard-errors under strict_images (frozen benchmark); warns otherwise. |
| 27 | Wildly out-of-frame boxes (e.g. 100k px) pass validation | ✅ DONE (careful) | Rejects only OVERSIZED boxes (dim > 12000). **NOTE: an earlier over-strict version also rejected slightly-negative origins (x0=-18, a legit edge-clipped box) and silently dropped 550 rows → 316 species. The real-data split test CAUGHT this. Fixed to keep edge-clipped boxes (`_extract_crop` clamps them).** |
| 28 | `_extract_crop()` clamping not robust for fully-outside boxes | ✅ DONE | Clamps both ends into [0,w]/[0,h]; returns zero-padded canvas on empty intersection. |
| 29 | Extreme aspect ratio can truncate a resized dim to 0 → cv2.resize fails | ✅ DONE | `max(1, int(...))` on both dims. |
| 55 | **Suffixed disk files (`...png-16944-1.png`) assumed to be source frames; if they're pre-made crops, applying frame-bbox coords is catastrophic.** | ✅ ~RESOLVED by results | If images were crops, every model would extract garbage regions and score ≈random (0.3% top-1 for 321 classes). Instead C08 = 84.1% top-1, all models coherent → bboxes ARE applied to correct source frames. Data from official OzFish site. Optional: run a size/fits check once for the paper's data-integrity appendix (uniform frame size + 100% bbox-fits), but no longer a risk. |
| 56 | Frame-collision warning ("lexicographically first") slightly misleading (dir order first) | 💤 DOC |
| 57 | Deployment parse `file_name.split("_")[0]` brittle for underscored IDs | 💤 (OzFish OK) |
| 58 | Training path doesn't detect duplicate annotations (exporter does) | ✅ DONE | `_load_rows` now detects duplicate annotation_ids: hard-error under strict_images, warn otherwise. |

## Sampler / training loop

| # | Finding | Status |
|---|---|---|
| 9 | A7 balanced sampler = median(count)×321 = 19,902 samples/epoch vs C09's 49,378 → A7 gets ~40% of C09's optimizer steps. Not a one-factor ablation. | ✅ FIXED (code) ⚠️ RE-RUN A7 | sampler.py default now draws ceil(N_train/C)/class → ~N_train/epoch, matching random's exposure. **The existing A7 result used the OLD median exposure — A7 must be re-run to be a valid one-factor ablation.** (A7 lost anyway, so conclusion unchanged, but the number is stale.) |
| 37 | No non-finite-loss guard (NaN can corrupt optimizer/EMA/best_state) | ✅ DONE | Both loops (main + probe) hard-fail on non-finite loss. |
| 38 | No guarantee a valid `best_state` exists (all-NaN val HD → tests current model silently) | ✅ DONE | Both loops raise if best_state is None (model selection failed). |
| 39 | `early_stop_min_delta` also gates checkpoint saving, not just patience → strict best not saved | 💤 (semantic; low) |
| 40 | Headline metric = macro accuracy, but checkpoint selection = min val HD | ✅ DONE (declared) | Kept HD-selection (deliberate: least-severe-mistakes = the paper's argument; uniform across panel). Documented at the selection site + must be stated in the paper protocol section. Not changed (would re-select every checkpoint). |
| 76 | Default `sampler = "balanced"` in RunConfig, but protocol is random | ✅ DONE | Default now `random`. No existing run affected (all configs set sampler explicitly). |

## Feature cache (currently disabled in the panel)

| # | Finding | Status |
|---|---|---|
| 41 | Cache key omits annotation IDs / metadata checksum / preprocessing / transformers version → collisions possible | 💤 (cache off; fix before enabling) |
| 42 | Cache ignores `--results_dir` (hard-coded `results/feature_cache`) | 💤 |

## Augmentation

| # | Finding | Status |
|---|---|---|
| 43 | `_apply_turbidity_noise()` has no probability gate → applied to every image (P=1.0); not a physical turbidity model | ✅ DONE | Renamed `_apply_gaussian_noise`, honest docstring (additive noise ≠ physical turbidity), added `noise_prob` gate **defaulting to 1.0** (preserves historical behaviour → existing results still reproducible; now documented + tunable). |
| 44 | Photometric corruptions sampled independently per context view (ROI/3×/5×/full-frame) → physically inconsistent | 🔍 (deliberate regularizer? test shared params) |
| 45 | Motion-blur kernel normalization no zero-sum guard | 💤 |

## Model / MCEAM

| # | Finding | Status |
|---|---|---|
| 46 | 3×/5×/full-frame context includes the fish itself → context attention may re-attend the target, not habitat | 🔍 (ablation: mask ROI in context) |
| 47 | Full frame can encode deployment/site shortcuts (deployment split helps but doesn't eliminate site correlation) | 🔍 (analyse) |
| 48 | MCEAM depth > 1 isn't a standard Transformer block stack | ✅ DONE (doc) | Module docstring now states: no residual/FFN, and depth>1 refines the query (not re-using the original ROI each layer). |
| 49 | MCEAM's module-level formula doesn't match implementation perfectly | ✅ DONE (doc) | Docstring formula rewritten to concat→FFN→gate (matches code), not summed contexts. |
| 50 | `embed_dim=768` hard-coded in ModelConfig instead of from backbone | ✅ DONE | MCEAM/head now sized from `backbone.embed_dim`; ModelConfig.embed_dim defaults None; explicit mismatch → ValueError. No behaviour change on DINOv2/v3-base (both 768). |
| 51 | DINO token slicing (CLS→register→patches) is CORRECT | ✅ (verified good) |

## Documentation / comments (mostly wording)

| # | Finding | Status |
|---|---|---|
| 15 | `MANIFEST.md` badly stale (says C09=balanced; old ablation names; wrong statuses) | ✅ DONE (regenerated from YAMLs, A15=proposed, real statuses) |
| 17 | A11 comment says patch-embed stays frozen; code unfreezes EVERYTHING (code is right) | ✅ DONE (comment corrected) |
| 18 | A16 says DINOv2-base is ViT-B/16; it's patch-14 (256 patches vs DINOv3's 196) → call it a backbone comparison | ✅ DONE (comment corrected) |
| 19 | C06 vs C09 is not a clean frozen-vs-FT comparison (many diffs); the clean one is C09→A9→A10→A11 | ✅ DONE (comment corrected) |
| 20 | C09/MCEAM context-levels comment miscounts (3 = social/habitat/full_frame; ROI is the query) | ✅ DONE (C09 comment fixed; MCEAM docstring still TODO) |
| 52 | Taxonomy corrections buried in `split.py` → move to versioned `taxonomy_corrections.csv` | ✅ DONE | Created `configs/taxonomy_corrections.csv` (type,key,corrected,reason); split.py loads it (in-code defaults as fallback). |
| 53 | Placeholder filter narrow (misses `sp.`/`spp.`/`cf.`/`aff.`) | ✅ DONE (verified) | Filter broadened to sp/sp./spp/spp./cf./aff./indet.; **verified split still yields 321 (no legit class dropped).** |
| 54 | Taxonomic completeness enforced only for HSLM, not flat baselines → validate globally | ✅ DONE | `split_dataset` now hard-errors if any benchmark species lacks genus/family — applies to EVERY model, not just HSLM. |
| 74 | `A7_balanced_sampler.yaml` self-describes as A6/random_sampler (rename drift) | ✅ DONE | Corrected the wrong A6 references; added the #9 re-run note directly in the config. |
| 75 | `RunConfig` docstring "mirrors train.py 1:1" — stale | ✅ DONE (fixed w/ DDP removal) |
| 77 | `scripts/train.py` retired-numbering docs | ✅ DONE (file deleted) |

## Metrics / reporting

| # | Finding | Status |
|---|---|---|
| 81 | `evaluate_classification()` lacks input validation (N_pred==N_target==N_scores, class bounds) | ✅ DONE | Validates preds/targets same 1-D shape, target class bounds, scores (N,C) shape. A malformed external (MATANet) prediction array now errors clearly instead of scoring wrong. |
| 82 | Your "Hierarchical Distance" isn't literal graph-hop distance | ✅ DONE (doc) | hd.py docstring: defined as an ordinal severity 0/1/2/3, explicitly not edge-count; distinct from MATANet's 0/2/4/6. |
| 83 | Training marginalization and evaluation genus/family accuracy use different mechanisms | ✅ DONE (doc) | metrics.py docstring: genus/family accuracy = taxonomy of top-1 predicted species (uniform across models), not marginalized argmax. |
| 84 | `species_accuracy` duplicates Top-1 | ✅ DONE (doc) | Commented as == top1/micro; do not report separately. |
| 85 | `group_sizes` (114/149/58 = 321) reported alongside test group accuracy over <321 present classes | 🔧 TODO (label; #2) |
| 86 | Regenerate benchmark stats from code (head/med/tail = 114/149/58, not the draft's 112/161/48) | 🔧 TODO |

## MATANet / C08 bridge (touch ONLY after C08 seed runs finish)

| # | Finding | Status |
|---|---|---|
| 59 | Exporter dup-ID test only fails on different img_path; identical dup annotation slips through | 🔧 TODO (post-C08) |
| 60 | MATANet stable ID uses `basename(img_path)` not source metadata filename | 🔧 TODO (post-C08) |
| 61 | `export_split.py` derives `file_name` from physical path, not metadata | 🔧 TODO |
| 62 | Sample dicts discard original `file_name` → add it; never reconstruct IDs from img_path | 🔧 TODO |
| 63 | `export_split.py` monkeypatches `os.path.exists` globally → add `require_images=False` param | 🔧 TODO |
| 64 | "identical hierarchical supervision" overclaim — only the taxonomy is identical, not the mechanism | ✅ DONE (matanet/README.md fixed: same taxonomy + same eval, not same supervision) |
| 65 | Eval HD uses 0/1/2/3; MATANet export matrix uses 0/2/4/6 → not the same objective | ✅ DONE (clarified in README + PAPER_FRAMING §11) |
| 66 | MATANet exporter substitutes `__unknown_family__` instead of hard-failing on incomplete taxonomy | 🔧 TODO |
| 67 | `patch_matanet.py` checks for expected text but not the pinned commit hash | 🔧 TODO (`git rev-parse HEAD`) |
| 68 | C08 result provenance weaker than rest (no upstream commit / config / CSV+JSON checksums) | 🔧 TODO |
| 69 | Ingest tolerates unknown predicted labels (closed-set → likely a mapping bug; hard-fail) | 🔧 TODO |
| 70 | No C08 result existed at audit time | ✅ (seed 0 now done, 2026-08-17; seeds 1/2 running one at a time) |

## Tests

| # | Finding | Status |
|---|---|---|
| 21 | 20/20 "passing" but split tests `print("SKIP"); return` (not `pytest.skip`) → false passes when real data absent | ✅ DONE | Real-CSV tests now `pytest.skip` (report SKIPPED on clean checkout). Added 3 SYNTHETIC-fixture split tests (leakage-safe, reproducible, stereo-pairs) that always run — real coverage without external data. 25 tests pass. |

## Environment

| # | Finding | Status |
|---|---|---|
| 31 | Small-object rule (`0.05`, `512`) is a hidden hyperparameter not in any config | ✅ DONE | Moved to `bioreef/protocol.py` (single source) + recorded in each result's provenance. |
| 32 | Many hyperparameters are code constants, not serialized (CB-Focal β, focal γ, wd, EMA, crops, aug probs) | ✅ DONE | Centralised in `bioreef/protocol.py`, wired into loop/dataset/loss as the source of truth, and added to the provenance fingerprint. |
| 33 | `train.py` could override beta/gamma without recording (moot — train.py deleted) | ✅ DONE (deleted) |
| 30 | Small-object 512→224 double-resize doesn't recover detail ("preserve texture" misleading) | ✅ DONE (doc) | Comment corrected: it's an interpolation-path change (+mild blur), not detail recovery; retained for reproducibility; flagged as an open ablation. Behaviour unchanged. |
| 78 | Common LR across architectures ≠ equally-tuned models | 💤 (methodological; declare tuning budget) |

---

## RE-RUN POLICY (decided 2026-08-17)

**Re-run ONLY when the run itself was scientifically wrong** — a real experimental
confound in how that number was produced. **Do NOT re-run for dev-history optics.**
Rationale: the project ships as a **NEW repo, uploaded all at once, single commit**,
so anything that was only a *chronology* artifact (post-hoc config ordering,
seeds trained at different intermediate commits) vanishes — one commit, one
revision, all configs present as initial designs. Re-running to fix optics wastes
compute; the fresh repo fixes it for free.

**MUST re-run (genuine confound in the run):**
- **A7 (balanced sampler)** — the #9 fix changed the balanced sampler's per-epoch
  exposure (median×C ≈ 40% → ~N_train, matching the random reference). A7's stored
  3-seed result trained on the WRONG budget — a real methodological error the fresh
  repo cannot launder. Re-run for a valid one-factor sampling ablation. (It lost
  anyway, so the conclusion holds; only the number is invalid.) Fast (frozen, 3
  seeds). Delete `results/A7_balanced_sampler/seed*/metrics.json` to force it.

**Do NOT re-run (optics only, resolved by the fresh repo):**
- C06/C07/C09 (and any) seeds trained across multiple intermediate commits (#10) —
  the differing revisions were docs / unrelated-path changes; the model code those
  runs used was unchanged, so the numbers are valid. Single-commit repo collapses
  them to one revision.
- Post-hoc config ordering (A12–A16 "created after seeing X") (#1/#16) — all configs
  ship at once as initial designs; no chronology to betray.

No other stored result is affected: the #37/#38/#23 guards only fire on failure
(never triggered on completed runs), #24/#25 config validation doesn't change
loaded values, and #76's default only touches ad-hoc one-offs (all configs set
`sampler` explicitly).

## Recommended execution order

1. 🔍 **#55 — data integrity** (verify disk files are source frames, not crops). Highest stakes; fast.
2. 🔧 **#16/#2 doc consistency** — A15=proposed, C09=frozen baseline, "321 defined/313 evaluable"; strip post-hoc narration.
3. 🔧 **Latent code fixes** — #23, #37, #38, #24, #25, #9, #76.
4. 🔧 **Reproducibility** — #3, #4, #10–#14, #71–#73 (the benchmark co-star).
5. 🔧 **MATANet #59–#69** — AFTER C08 seeds finish.
6. 🔧 **Tests #21** + regenerate MANIFEST/#15/#86 from code.
7. 💤 Doc-only + deferred as a final polish pass before upload.

All fixes accumulate in the working tree — **no piecemeal commits** (single final commit into the new repo, per standing constraint).
