#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_slurm.sh — one-time setup on a Slurm cluster LOGIN NODE.
#
# This is the Slurm counterpart to setup_cloud.sh. They are NOT interchangeable:
# setup_cloud.sh assumes a box you own (sudo, a GPU on the machine you ssh into,
# a venv). On a cluster all three are false:
#   * no sudo            -> no apt-get; use the provided `module` + conda
#   * no GPU on login    -> nvidia-smi here shows nothing; that is NORMAL
#   * jobs run elsewhere -> anything you set up must live on a SHARED filesystem
#                           ($HOME), not /tmp, or the compute node won't see it
#
# Run this ONCE on the login node:   bash slurm/setup_slurm.sh
# Then submit work with:             sbatch slurm/job_bioreef.sh C09
#
# ── WHAT YOU MUST PROVIDE ───────────────────────────────────────────────────
#   HF_TOKEN : HuggingFace token WITH access to the gated DINOv3 repo
#              (facebook/dinov3-vitb16-pretrain-lvd1689m).
#   the data : frame_metadata.csv + frames/ — scp'd from your VM (see below).
#
# ── GETTING THE DATA THERE ──────────────────────────────────────────────────
# From the machine that HAS the data (your VM), not from here:
#     tar czf ozfish_bench.tar.gz frame_metadata.csv frames/
#     scp ozfish_bench.tar.gz oelmutasim@44.210.222.21:/home/oelmutasim/
# ~76k crops is large; if scp keeps dropping, use `rsync -avP --partial` which
# resumes instead of restarting.
# ---------------------------------------------------------------------------
set -euo pipefail

# ===================== CONFIG — EDIT THESE ==================================
HF_TOKEN="${HF_TOKEN:-PUT_YOUR_HF_TOKEN_HERE}"
REPO_URL="${REPO_URL:-https://github.com/Karam-Kreidli/bioreef.git}"
WORKDIR="${WORKDIR:-$HOME/bioreef-classify}"
ENVNAME="${ENVNAME:-bioreef}"
TARBALL="${TARBALL:-$HOME/ozfish_bench.tar.gz}"
# ===========================================================================

say() { printf "\n\033[1;36m[setup] %s\033[0m\n" "$*"; }
die() { printf "\n\033[1;31m[setup] ERROR: %s\033[0m\n" "$*" >&2; exit 1; }

# --- 0. we should be on the login node, and that means NO GPU --------------
say "Environment"
hostname
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  say "note: a GPU is visible here. Still do not train on the login node —"
  say "      submit with sbatch. Long jobs on a login node get killed."
else
  say "no GPU on this node — expected on a login node. Jobs get GPUs via sbatch."
fi
command -v sbatch >/dev/null || die "no sbatch found — is this really the Slurm cluster?"

# --- 1. conda --------------------------------------------------------------
say "Conda"
module load anaconda3 2>/dev/null || module load anaconda3/3.11 2>/dev/null || \
  say "could not 'module load anaconda3' — continuing if conda is already on PATH"
command -v conda >/dev/null || die "conda not on PATH after module load"

# Create with an EXPLICIT python. `conda create -n env` with no package pins
# gives an env with no python at all, and the first pip call then silently
# installs into the base environment.
if ! conda env list | grep -qE "^${ENVNAME}\s"; then
  say "creating env '$ENVNAME' (python 3.11)"
  conda create -y -n "$ENVNAME" python=3.11
else
  say "env '$ENVNAME' already exists"
fi
# `conda activate` needs the shell hook; `source activate` is the older form the
# cluster docs use. Use the hook when we can, fall back to what they documented.
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate "$ENVNAME" 2>/dev/null || source activate "$ENVNAME"
python --version
[ "$(basename "$CONDA_PREFIX")" = "$ENVNAME" ] || die "env did not activate (CONDA_PREFIX=$CONDA_PREFIX)"

# --- 2. repo ---------------------------------------------------------------
say "Repo -> $WORKDIR"
if [ ! -d "$WORKDIR/.git" ]; then
  git clone "$REPO_URL" "$WORKDIR"
else
  git -C "$WORKDIR" pull --ff-only
fi
cd "$WORKDIR"

# --- 3. python deps --------------------------------------------------------
say "Python deps"
pip install --quiet --upgrade pip
python -c "import torch,sys; sys.exit(0 if torch.version.cuda else 1)" 2>/dev/null \
  && say "torch with CUDA already present" \
  || { say "installing torch (CUDA 12.1 wheels)"
       pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121; }
pip install --quiet -r requirements.txt
python -c "import transformers, timm; print('transformers', transformers.__version__, '| timm', timm.__version__)"
# NOTE: torch.cuda.is_available() is False here and that is fine — no GPU on the
# login node. The job script re-checks it on the compute node, where it matters.

# --- 4. HuggingFace: download weights NOW, on the login node ---------------
# Compute nodes frequently have no outbound internet. If the first thing a job
# does is fetch a 350 MB gated checkpoint, it dies 20 minutes into the queue.
# Pre-populating the shared HF cache makes the job run offline.
say "HuggingFace auth + weight pre-fetch"
[ "$HF_TOKEN" != "PUT_YOUR_HF_TOKEN_HERE" ] || die "set HF_TOKEN (needs gated DINOv3 access)"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
python - <<PY || die "DINOv3 not accessible with this token — request access on the model page"
from huggingface_hub import login, model_info
login("$HF_TOKEN", add_to_git_credential=False)
model_info("facebook/dinov3-vitb16-pretrain-lvd1689m")
print("[setup] DINOv3 access OK")
PY
python - <<'PY'
# Pull every backbone the campaign uses into the cache while we still have net.
from transformers import AutoModel
for mid in ("facebook/dinov3-vitb16-pretrain-lvd1689m", "facebook/dinov2-base"):
    AutoModel.from_pretrained(mid)
    print("[setup] cached", mid)
import timm
for m in ("resnet50", "convnext_tiny", "swin_base_patch4_window7_224"):
    timm.create_model(m, pretrained=True)
    print("[setup] cached", m)
PY

# --- 5. dataset ------------------------------------------------------------
say "Dataset"
if [ ! -f frame_metadata.csv ] || [ ! -d frames ]; then
  [ -f "$TARBALL" ] || die "no data. scp your tarball first:
    (on the VM)  tar czf ozfish_bench.tar.gz frame_metadata.csv frames/
                 scp ozfish_bench.tar.gz oelmutasim@44.210.222.21:/home/oelmutasim/
  then re-run this script (or set TARBALL=/path/to/it)."
  say "extracting $TARBALL"
  tar xzf "$TARBALL" -C "$WORKDIR"
fi
[ -f frame_metadata.csv ] || die "frame_metadata.csv missing after extract"
[ -d frames ]             || die "frames/ missing after extract"
say "frames on disk: $(find frames -type f | wc -l)"

python - <<PY
import yaml, os
p = "configs/benchmark.yaml"
c = yaml.safe_load(open(p))
c.setdefault("data", {})
c["data"]["csv_path"] = os.path.abspath("frame_metadata.csv")
c["data"]["img_dir"]  = os.path.abspath("frames")
yaml.safe_dump(c, open(p, "w"), sort_keys=False)
print("[setup] benchmark.yaml ->", c["data"]["csv_path"])
PY

# --- 6. verify the split ---------------------------------------------------
# strict_images is true, so this FAILS LOUDLY if the transfer dropped crops.
# That is the point: a partial frames/ dir would otherwise silently redefine the
# benchmark (different species count, class indices and split sizes).
say "Verify split (expect 321 species; strict_images=true will catch a partial transfer)"
python scripts/export_split.py --split_seed 0 \
  || die "split failed. If it reports missing images, your scp was incomplete —
  re-sync with: rsync -avP frames/ oelmutasim@44.210.222.21:$WORKDIR/frames/"

say "DONE. Submit work with:"
echo "    cd $WORKDIR"
echo "    sbatch slurm/job_bioreef.sh C09          # one run, seed 0"
echo "    sbatch slurm/job_bioreef.sh A11 1        # run A11, seed 1"
echo "    squeue -u \$USER                          # check status"
echo "    tail -f logs/<jobid>.out                 # watch it"
