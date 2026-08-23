#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/Desktop/Senior_Group}"
REPO="${REPO:-$ROOT/bioreef}"
MATANET="${MATANET:-$ROOT/matanet-repo}"
OUT="${OUT:-$REPO/matanet/ozfish_data}"
LOGS="${LOGS:-$ROOT/logs}"
PIN="922c2176893ef1d03de8b8701cd882b5764f9ae9"
SEEDS="${SEEDS:-0 1 2}"
BATCH="${BATCH:-16}"
ACCUM="${ACCUM:-1}"
GRAD_CKPT="${GRAD_CKPT:-0}"
mkdir -p "$LOGS"
cd "$REPO"

say()  { printf "\n\033[1;36m[c08] %s\033[0m\n" "$*"; }
die()  { printf "\n\033[1;31m[c08] ERROR: %s\033[0m\n" "$*" >&2; exit 1; }

say "GPU check"
command -v nvidia-smi >/dev/null || die "no nvidia-smi"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

say "Data check"
CSV="${CSV:-$(find "$REPO" -maxdepth 4 -name frame_metadata.csv 2>/dev/null | head -1)}"
FRAMES="${FRAMES:-$(find "$REPO" -maxdepth 4 -type d -name frames 2>/dev/null | head -1)}"
[ -n "$CSV" ] && [ -f "$CSV" ]       || die "no frame_metadata.csv under $REPO"
[ -n "$FRAMES" ] && [ -d "$FRAMES" ] || die "no frames/ dir under $REPO"
NPNG=$(find "$FRAMES" -maxdepth 1 -name '*.png' | wc -l)
NROW=$(($(wc -l < "$CSV") - 1))
say "frames: $NPNG | rows: $NROW"
[ "$NPNG" -gt 0 ] || die "no crops under $FRAMES"

say "Set benchmark.yaml -> on-disk paths"
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

say "Python deps"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || die "torch has no CUDA — activate the bioreef conda env first"
pip install --quiet tensorboard
pip install --quiet hf_transfer

if [ ! -d "$MATANET/.git" ]; then
  say "Clone MATANet @ $PIN"
  git clone https://github.com/dhlee-work/fathomnet-cvpr2025-ssl.git "$MATANET"
  git -C "$MATANET" checkout "$PIN"
else
  say "MATANet repo present ($(git -C "$MATANET" rev-parse --short HEAD))"
fi
say "Install MATANet requirements"
pip install --quiet -r "$MATANET/requirements.txt"
say "Patch MATANet paths"
python matanet/patch_matanet.py --repo "$MATANET"

if ! pgrep -f "tensorboard.*--port 6006" >/dev/null; then
  say "Launch TensorBoard on :6006"
  mkdir -p "$MATANET/logs"
  nohup tensorboard --logdir "$MATANET/logs" --port 6006 --host 0.0.0.0 \
        >"$LOGS/tensorboard.log" 2>&1 &
  sleep 2
fi

BATCH_FLAGS="--batch_size $BATCH --accumulate_grad_batches $ACCUM"
[ "$GRAD_CKPT" = "1" ] && BATCH_FLAGS="$BATCH_FLAGS --gradient_checkpointing"
say "batch config: $BATCH_FLAGS"

for S in $SEEDS; do
  LOG="$LOGS/c08_seed${S}.log"
  CFG="$OUT/ozfish_config_seed${S}.yaml"
  METRICS="$REPO/results/C08_matanet/seed${S}/metrics.json"
  if [ -f "$METRICS" ]; then
    say "seed $S already has metrics.json — skipping"
    continue
  fi
  {
    say "seed $S :: 1/3 export"
    python matanet/export_ozfish.py --seed "$S" --out_dir "$OUT" $BATCH_FLAGS
    say "seed $S :: 2/3 train (B1.BuildModel)"
    ( cd "$MATANET" && python B1.BuildModel.py --config "$CFG" )
    say "seed $S :: 2b test (C1.TestModel)"
    ( cd "$MATANET" && python C1.TestModel.py  --config "$CFG" )
    say "seed $S :: 3/3 ingest"
    python matanet/ingest_predictions.py --data_dir "$OUT" --seed "$S"
  } 2>&1 | tee "$LOG"
  [ -f "$METRICS" ] || die "seed $S finished but no metrics.json at $METRICS"
  say "seed $S DONE -> $METRICS"
done

say "Aggregate"
python scripts/aggregate.py 2>&1 | tee "$LOGS/aggregate.log"
say "ALL DONE. C08 metrics in results/C08_matanet/seed*/ and RESULTS.md."
