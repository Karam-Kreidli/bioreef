#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_c08.sh — end-to-end MATANet (C08) on the OzFish split, on a RunPod A40.
#
# Runs entirely from files already on the persistent volume (repo + 106 GB
# frames + frame_metadata.csv were rsync'd from the VM). C08 uses DINOv2-large
# (facebook/dinov2-large) which is NOT gated, so NO HuggingFace token is needed
# (unlike C09/DINOv3). Idempotent: safe to re-run; each stage skips if its
# output already exists, so a crashed seed resumes without redoing finished work.
#
#   bash matanet/run_c08.sh            # all 3 seeds, then aggregate
#   SEEDS="0" bash matanet/run_c08.sh  # just seed 0 (smoke)
#
# Watch progress (three ways):
#   * TensorBoard : https://<podid>-6006.proxy.runpod.net   (launched below)
#   * live log    : tail -f $LOGS/c08_seed<N>.log
#   * from Claude : stream-pod-logs
# ---------------------------------------------------------------------------
set -euo pipefail

# ===================== paths on the volume =================================
ROOT="${ROOT:-/workspace}"
REPO="${REPO:-$ROOT/bioreef-classify}"
MATANET="${MATANET:-$ROOT/matanet-repo}"
OUT="${OUT:-$REPO/matanet/ozfish_data}"
LOGS="${LOGS:-$ROOT/logs}"
PIN="922c2176893ef1d03de8b8701cd882b5764f9ae9"   # pinned MATANet commit
SEEDS="${SEEDS:-0 1 2}"
mkdir -p "$LOGS"
cd "$REPO"

say()  { printf "\n\033[1;36m[c08] %s\033[0m\n" "$*"; }
die()  { printf "\n\033[1;31m[c08] ERROR: %s\033[0m\n" "$*" >&2; exit 1; }

# --- 0. sanity: GPU + data present (fail fast, before any GPU-hours) --------
say "GPU check"
command -v nvidia-smi >/dev/null || die "no nvidia-smi — pod has no GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

say "Data check (the crops must have made it across from the VM)"
# auto-detect wherever the transfer put them (repo-root OR bioreef/data/...)
CSV="${CSV:-$(find "$REPO" -maxdepth 4 -name frame_metadata.csv 2>/dev/null | head -1)}"
FRAMES="${FRAMES:-$(find "$REPO" -maxdepth 4 -type d -name frames 2>/dev/null | head -1)}"
[ -n "$CSV" ] && [ -f "$CSV" ]       || die "no frame_metadata.csv found under $REPO — transfer incomplete"
[ -n "$FRAMES" ] && [ -d "$FRAMES" ] || die "no frames/ dir found under $REPO — transfer incomplete"
NPNG=$(find "$FRAMES" -maxdepth 1 -name '*.png' | wc -l)
NROW=$(($(wc -l < "$CSV") - 1))
say "frames on disk: $NPNG  |  metadata rows: $NROW (rows >> crops is normal: metadata includes filtered/non-crop annotations)"
# NOTE: do NOT gate on NPNG vs metadata rows — the CSV has many more rows than
# extracted crops. The authoritative completeness check is export_ozfish.py
# below: with strict_images:true it hard-errors (during CPU export, before any
# GPU time) if the SPLIT references a crop that is missing on disk.
[ "$NPNG" -gt 0 ] || die "no crops found under $FRAMES — transfer did not land"

# --- 1. point benchmark.yaml at the volume paths ---------------------------
say "Set benchmark.yaml -> volume paths"
python - "$CSV" "$FRAMES" <<'PY'
import sys, yaml
csv, frames = sys.argv[1], sys.argv[2]
p = "configs/benchmark.yaml"
c = yaml.safe_load(open(p))
c.setdefault("data", {})
c["data"]["csv_path"] = csv
c["data"]["img_dir"]  = frames
yaml.safe_dump(c, open(p, "w"), sort_keys=False)
print("[c08] benchmark.yaml ->", csv, "|", frames)
PY

# --- 2. python env (reuse base CUDA torch; add deps on top) ----------------
say "Python deps"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || die "base image torch has no CUDA — pick a PyTorch/CUDA pod template"
pip install --quiet -r requirements.txt
pip install --quiet tensorboard
# The RunPod base image sets HF_HUB_ENABLE_HF_TRANSFER=1 but omits the hf_transfer
# package, which makes every HuggingFace download raise. Satisfy it so the DINOv2
# weight/processor fetch works (DINOv2-large is ungated — no token needed).
pip install --quiet hf_transfer
# NOTE: MATANet's own requirements (pytorch-lightning, etc.) are installed in
# step 3 AFTER the repo is cloned — installing them here would be a no-op because
# $MATANET/requirements.txt does not exist yet.

# --- 3. clone + pin + patch MATANet (idempotent) ---------------------------
if [ ! -d "$MATANET/.git" ]; then
  say "Clone MATANet @ $PIN"
  git clone https://github.com/dhlee-work/fathomnet-cvpr2025-ssl.git "$MATANET"
  git -C "$MATANET" checkout "$PIN"
else
  say "MATANet repo present ($(git -C "$MATANET" rev-parse --short HEAD))"
fi
say "Patch MATANet paths (idempotent)"
python matanet/patch_matanet.py --repo "$MATANET"

# --- 4. TensorBoard (background; Lightning logs land in $MATANET/logs) ------
if ! pgrep -f "tensorboard.*--port 6006" >/dev/null; then
  say "Launch TensorBoard on :6006"
  mkdir -p "$MATANET/logs"
  nohup tensorboard --logdir "$MATANET/logs" --port 6006 --host 0.0.0.0 \
        >"$LOGS/tensorboard.log" 2>&1 &
  sleep 2
fi

# --- 5. per-seed: export -> train (B1) -> test (C1) -> ingest ---------------
for S in $SEEDS; do
  LOG="$LOGS/c08_seed${S}.log"
  CFG="$OUT/ozfish_config_seed${S}.yaml"
  METRICS="$REPO/results/C08_matanet/seed${S}/metrics.json"

  if [ -f "$METRICS" ]; then
    say "seed $S already has metrics.json — skipping (delete it to force re-run)"
    continue
  fi

  {
    say "seed $S :: 1/3 export our split -> MATANet inputs"
    # A40 (44GB) OOMs MATANet's 4x DINOv2-large at physical batch 16, so split the
    # published effective batch 16 into 8x2 and enable gradient checkpointing
    # (recompute = numerically identical to batch-16, ~30% slower). The export
    # enforces batch_size*accum==16, so this stays a faithful official-repo baseline.
    python matanet/export_ozfish.py --seed "$S" --out_dir "$OUT" \
        --batch_size 8 --accumulate_grad_batches 2 --gradient_checkpointing

    say "seed $S :: 2/3 train (B1.BuildModel) — DINOv2-large fine-tune"
    ( cd "$MATANET" && python B1.BuildModel.py --config "$CFG" )

    say "seed $S :: 2b test (C1.TestModel) -> predictions"
    ( cd "$MATANET" && python C1.TestModel.py  --config "$CFG" )

    say "seed $S :: 3/3 ingest predictions -> our metrics"
    python matanet/ingest_predictions.py --data_dir "$OUT" --seed "$S"
  } 2>&1 | tee "$LOG"

  [ -f "$METRICS" ] || die "seed $S finished but no metrics.json at $METRICS"
  say "seed $S DONE -> $METRICS"
done

# --- 6. aggregate into RESULTS.md ------------------------------------------
say "Aggregate (C08 lands in RESULTS.md)"
python scripts/aggregate.py 2>&1 | tee "$LOGS/aggregate.log"

say "ALL DONE. C08 metrics in results/C08_matanet/seed*/ and RESULTS.md."
say "Tear-down reminder: stop/delete the pod AND delete the volume to stop billing."
