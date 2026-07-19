#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_cloud.sh — bootstrap bioreef-classify on a fresh cloud GPU box for the
# heavy runs (A12 full fine-tune, C08 MATANet) that the 8 GB Quadros can't fit.
#
# Target box: ONE GPU with >=24 GB VRAM (A10 / L4 / A5000 / RTX 4090/3090).
# C08 (MATANet, DINOv2-large ~300M, multi-context) is the sizing constraint;
# A12 (full FT of ViT-B + MCEAM) fits in 16 GB but 24 gives batch-32 headroom.
#
# Run once on the fresh box:  bash setup_cloud.sh
# It is idempotent-ish: safe to re-run; skips steps whose output already exists.
#
# ── WHAT YOU MUST PROVIDE (edit the CONFIG block below) ─────────────────────
#   HF_TOKEN     : a HuggingFace token WITH access to the gated DINOv3 repo
#                  (facebook/dinov3-vitb16-pretrain-lvd1689m). Without it the
#                  backbone download 401s — this repo has hit that exact error.
#   DATA_URL     : a URL to a tarball of your dataset artifacts (see below).
#
# ── THE DATA (important) ────────────────────────────────────────────────────
# The benchmark needs TWO things that are NOT in this git repo:
#   1. frame_metadata.csv     (lives in the pipeline repo, Junior-Project/)
#   2. the ~76k OzFish crop images
# OzFish source frames are public (AIMS), but the crops are YOUR extraction, so
# the reliable path is to package what you already have and fetch it here, NOT to
# re-extract from raw OzFish on the box. Make a tarball ONCE on your VM:
#
#   tar czf ozfish_bench.tar.gz frame_metadata.csv frames/     # from where they live
#   # upload it somewhere the box can curl: an S3/GCS presigned URL, a
#   # transfer.sh / file host, or `rclone` to your Drive. Put that URL in DATA_URL.
#
# The tarball must expand to:  ./frame_metadata.csv  and  ./frames/<crop>.png
# ---------------------------------------------------------------------------
set -euo pipefail

# ===================== CONFIG — EDIT THESE ==================================
HF_TOKEN="${HF_TOKEN:-PUT_YOUR_HF_TOKEN_HERE}"
DATA_URL="${DATA_URL:-PUT_YOUR_DATA_TARBALL_URL_HERE}"
REPO_URL="${REPO_URL:-https://github.com/Karam-Kreidli/bioreef.git}"
WORKDIR="${WORKDIR:-$HOME/bioreef-classify}"
# ===========================================================================

say() { printf "\n\033[1;36m[setup] %s\033[0m\n" "$*"; }
die() { printf "\n\033[1;31m[setup] ERROR: %s\033[0m\n" "$*" >&2; exit 1; }

# --- 0. sanity: GPU present ------------------------------------------------
say "GPU check"
command -v nvidia-smi >/dev/null || die "no nvidia-smi — this box has no GPU driver"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
[ "$VRAM_MB" -ge 22000 ] || say "WARNING: <24 GB VRAM ($VRAM_MB MB). A12 may fit; C08 (DINOv2-large) likely won't at batch>=16."

# --- 1. system deps + python ----------------------------------------------
say "System deps"
if command -v apt-get >/dev/null; then
  sudo apt-get update -qq && sudo apt-get install -y -qq git wget curl python3-venv python3-pip libgl1 libglib2.0-0
fi
PY=$(command -v python3 || command -v python) || die "no python"
"$PY" --version

# --- 2. clone the repo -----------------------------------------------------
say "Repo: $REPO_URL -> $WORKDIR"
if [ ! -d "$WORKDIR/.git" ]; then
  git clone "$REPO_URL" "$WORKDIR"
else
  git -C "$WORKDIR" pull --ff-only
fi
cd "$WORKDIR"

# --- 3. venv + python deps -------------------------------------------------
say "Python env"
[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
# Install torch matched to the box's CUDA FIRST (the requirements torch>=2.1 will
# otherwise pull a default build). If the box's base image already has torch, skip.
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null \
  && say "torch already present with CUDA" \
  || { say "installing torch (CUDA 12.1 wheels)"; pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121; }
pip install --quiet -r requirements.txt
python -c "import transformers, timm; print('transformers', transformers.__version__, '| timm', timm.__version__)"

# --- 4. HuggingFace auth for gated DINOv3 ----------------------------------
say "HuggingFace auth (DINOv3 is a gated repo)"
[ "$HF_TOKEN" != "PUT_YOUR_HF_TOKEN_HERE" ] || die "set HF_TOKEN (needs access to facebook/dinov3-vitb16-pretrain-lvd1689m)"
python -c "from huggingface_hub import login; login('$HF_TOKEN')"
# fail fast if the token lacks DINOv3 access, rather than 6 min into a run
python - <<'PY' || die "DINOv3 not accessible with this token — request access on the model page"
from huggingface_hub import model_info
model_info("facebook/dinov3-vitb16-pretrain-lvd1689m")
print("[setup] DINOv3 access OK")
PY

# --- 5. dataset ------------------------------------------------------------
say "Dataset"
if [ ! -f frame_metadata.csv ] || [ ! -d frames ]; then
  [ "$DATA_URL" != "PUT_YOUR_DATA_TARBALL_URL_HERE" ] || die "set DATA_URL to your ozfish_bench.tar.gz (see header)"
  say "downloading dataset tarball"
  curl -fL "$DATA_URL" -o /tmp/ozfish_bench.tar.gz
  tar xzf /tmp/ozfish_bench.tar.gz -C "$WORKDIR"
fi
[ -f frame_metadata.csv ] || die "frame_metadata.csv missing after extract"
[ -d frames ]             || die "frames/ dir missing after extract"

# point benchmark.yaml at the box's paths (absolute)
python - <<PY
import yaml, os
p = "configs/benchmark.yaml"
c = yaml.safe_load(open(p))
c.setdefault("data", {})
c["data"]["csv_path"] = os.path.abspath("frame_metadata.csv")
c["data"]["img_dir"]  = os.path.abspath("frames")
yaml.safe_dump(c, open(p, "w"), sort_keys=False)
print("[setup] benchmark.yaml ->", c["data"]["csv_path"], "|", c["data"]["img_dir"])
PY

# --- 6. verify: split regenerates to 321 species ---------------------------
say "Verify split (expect 321 species)"
python scripts/export_split.py --split_seed 0 || die "export_split failed — check the CSV/paths"

# --- 7. smoke run: 1 epoch of C09 to prove the whole stack works -----------
say "Smoke run: C09 for 1 epoch (proves data + DINOv3 + train loop)"
python scripts/run.py C09 --seed 0 --gpu 0 --epochs 1 --num_workers 8 \
  --results_dir results_smoke || die "smoke run failed"

say "DONE. The box is ready. Now run the heavy jobs, e.g.:"
echo "    python scripts/run.py A12 --gpu 0 --num_workers 8 --save_checkpoint   # full FT (batch 32 fits on 24 GB)"
echo "    # C08 (MATANet): follow matanet/README.md — clone their repo, patch, export, run, ingest"
echo "  Remember: A12 keeps batch 32 for panel parity; C08 keeps MATANet's own batch (do not bump it)."
