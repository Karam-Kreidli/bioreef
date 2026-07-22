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
#   the data : bioreef/data/{metadata,frames}/ — rsync'd from your VM (below).
#
# ── GETTING THE DATA THERE ──────────────────────────────────────────────────
# From the repo root on the machine that HAS the data (your VM), not from here.
# rsync needs the target's PARENT to exist and creates only the last component,
# so make the dirs on the cluster first:
#     ssh oelmutasim@44.210.222.21 'mkdir -p ~/bioreef-classify/bioreef/data/{frames,metadata}'
#     rsync -avP bioreef/data/metadata/ oelmutasim@...:~/bioreef-classify/bioreef/data/metadata/
#     rsync -avP bioreef/data/frames/   oelmutasim@...:~/bioreef-classify/bioreef/data/frames/
# The trailing slash on the SOURCE matters: "frames/" copies the contents,
# "frames" would nest it as frames/frames/. rsync -P resumes on a dropped link.
# ---------------------------------------------------------------------------
set -euo pipefail

# ===================== CONFIG — EDIT THESE ==================================
# Pass this in the ENVIRONMENT, never by editing this line:
#     HF_TOKEN=hf_xxxxx bash slurm/setup_slurm.sh
# The ${VAR:-default} form already lets the env var win, so there is no reason
# to paste a real token here — and a token committed to git stays in the history
# even after the line is edited.
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
# Match on the env's PATH, not the name column: `conda env list` pads names and
# can render the active one with a '*', so "^name\s" misses an existing env and
# silently recreates it on every re-run.
if ! conda env list | awk '{print $NF}' | grep -qx ".*/envs/${ENVNAME}"; then
  say "creating env '$ENVNAME' (python 3.11 + conda-built numpy)"
  # numpy comes from CONDA, not pip. This box is RHEL7-era (glibc 2.17, GCC
  # 7.3.1); current numpy wheels need a newer glibc, so pip falls back to
  # building from source and dies on "NumPy requires GCC >= 9.3". Conda ships
  # a numpy compiled against its own toolchain, sidestepping the system GCC.
  conda create -y -n "$ENVNAME" python=3.11 "numpy>=1.24,<2"
else
  say "env '$ENVNAME' already exists"
  # An env created by an earlier version of this script has no conda numpy, so
  # pip would try (and fail) to build it from source. Install it via conda here
  # so re-running this script repairs that env instead of requiring a delete.
  conda install -y -n "$ENVNAME" "numpy>=1.24,<2" >/dev/null 2>&1 \
    && say "ensured conda-built numpy in '$ENVNAME'" \
    || say "note: could not conda-install numpy (may already be present)"
fi
# `conda activate` needs the shell hook; `source activate` is the older form the
# cluster docs use. Use the hook when we can, fall back to what they documented.
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate "$ENVNAME" 2>/dev/null || source activate "$ENVNAME"
python --version
[ "$(basename "$CONDA_PREFIX")" = "$ENVNAME" ] || die "env did not activate (CONDA_PREFIX=$CONDA_PREFIX)"

# Ignore the user site-packages (~/.local). This box has a stale ~/.local from
# another project (ultralytics, an older torch) that pip and python both pick up
# ahead of the conda env — that is why torch landed in ~/.local and imported a
# build needing libcusparseLt.so.0, which this box does not have. With this set,
# every python/pip below sees ONLY the env.
export PYTHONNOUSERSITE=1

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
# Always the ENV's pip, never a user/base one: `python -m pip` binds to the
# interpreter we just activated. --no-user forbids installing into ~/.local even
# if some global pip config re-enables it.
PIP="python -m pip install --quiet --no-user"
$PIP --upgrade pip

# A previous run leaked torch into ~/.local (before PYTHONNOUSERSITE was set).
# --no-user stops NEW installs going there, but pip still SEES that torch as
# "already installed" and can skip reinstalling into the env; worse, python
# imports it ahead of the env copy. Remove the leaked user copy — only ~/.local.
if [ -d "$HOME/.local/lib/python3.11/site-packages/torch" ]; then
  say "removing leaked torch from ~/.local (was shadowing the env)"
  rm -rf "$HOME/.local/lib/python3.11/site-packages/torch" \
         "$HOME/.local/lib/python3.11/site-packages/torchvision" 2>/dev/null || true
fi
# --only-binary=:all: for every pip call below. On this old toolchain a missing
# wheel silently becomes a source build that fails deep in a compiler error;
# this turns that into an immediate, readable "no wheel available" instead.
export PIP_ONLY_BINARY=":all:"

# Torch: pin 2.5.1 (cu121). This is a two-sided constraint:
#   * transformers 4.56.x (needed for DINOv3) imports
#     torch.nn.attention.flex_attention, which does NOT exist before torch 2.5
#     -> ModuleNotFoundError deep in transformers.masking_utils. So torch must
#     be >= 2.5.
#   * unpinned/newer torch resolves to a wheel that dlopens libcusparseLt.so.0,
#     absent on this box -> ImportError on `from torch._C import *`. 2.5.1 cu121
#     imports cleanly here (verified) and does not need it.
# 2.5.1 is the version that satisfies both. --extra-index-url (not --index-url):
# --index-url REPLACES PyPI, sending pip to pytorch.org for numpy's build deps.
NEED_TORCH="2.5.1"
if python -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('$NEED_TORCH') else 1)" 2>/dev/null; then
  say "torch $NEED_TORCH already present"
else
  say "installing torch $NEED_TORCH (cu121)"
  $PIP torch==2.5.1 torchvision==0.20.1 \
    --extra-index-url https://download.pytorch.org/whl/cu121
fi
# requirements.txt says numpy>=1.24, which would let pip pull 2.x over conda's
# build and re-trigger the source compile. Keep whatever conda installed.
# (The code itself is numpy-2 clean — this pin is a platform workaround for the
# old glibc/GCC on this cluster, not a code constraint.)
$PIP -r requirements.txt "numpy<2"
# transformers 4.56.2 is the verified-working version with torch 2.5.1: has
# DINOv3, and its flex_attention / DTensor imports resolve against 2.5. Pin it
# exactly so a re-run cannot drift to an untested release.
$PIP --force-reinstall --no-deps "transformers==4.56.2"
python -c "import numpy; print('numpy', numpy.__version__)"
# Prove torch actually LOADS (its C extension), not just that it imports a name.
python -c "import torch; print('torch', torch.__version__, '| built for CUDA', torch.version.cuda)"
python -c "import transformers, timm; print('transformers', transformers.__version__, '| timm', timm.__version__)"
# Prove DINOv3 modeling code actually IMPORTS against this torch — this is the
# exact path that failed on a DTensor import, so surface it here, not mid-run.
python -c "from transformers.models.dinov3_vit import modeling_dinov3_vit; print('dinov3 modeling imports OK')"
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
# Canonical layout (matches the VM): the data sits INSIDE the bioreef/ package
# dir, alongside the .py sources — bioreef/data/{frames,metadata}/.
CSV_PATH="$WORKDIR/bioreef/data/metadata/frame_metadata.csv"
IMG_PATH="$WORKDIR/bioreef/data/frames"
mkdir -p "$(dirname "$CSV_PATH")" "$IMG_PATH"

if [ ! -f "$CSV_PATH" ] || [ -z "$(ls -A "$IMG_PATH" 2>/dev/null)" ]; then
  # A tarball is OPTIONAL. If one was transferred, use it; otherwise tell the
  # user how to rsync the directory straight across. Packing 76k already-
  # compressed PNGs into a tar.gz needs a second copy's worth of free disk on
  # the source machine and saves almost nothing, so rsync is the better default.
  if [ -f "$TARBALL" ]; then
    say "extracting $TARBALL"
    tar xzf "$TARBALL" -C "$WORKDIR"
  else
    die "no data yet — but the target directories now EXIST (created above), so
the transfer below will work. From the machine that HAS the data (your VM):

  A) rsync the directories across — no extra disk needed on the VM, and it
     RESUMES if the connection drops (best for ~76k files). Run FROM the repo
     root on the VM:
       rsync -avP bioreef/data/frames/ \\
         oelmutasim@44.210.222.21:$IMG_PATH/
       rsync -avP bioreef/data/metadata/ \\
         oelmutasim@44.210.222.21:$(dirname "$CSV_PATH")/

  B) stream a tar over ssh — no temp file on the VM either, but it restarts
     from zero if it breaks:
       tar cf - bioreef/data/frames bioreef/data/metadata | \\
         ssh oelmutasim@44.210.222.21 'tar xf - -C $WORKDIR'

  C) if you already made a tarball, put it at \$TARBALL ($TARBALL) and re-run.
     It must expand to bioreef/data/{frames,metadata}/ under the repo root.

then re-run this script."
  fi
fi
[ -f "$CSV_PATH" ] || die "metadata CSV missing: $CSV_PATH"
say "frames on disk: $(find "$IMG_PATH" -type f | wc -l)"

CSV_PATH="$CSV_PATH" IMG_PATH="$IMG_PATH" python - <<'PY'
import yaml, os
p = "configs/benchmark.yaml"
c = yaml.safe_load(open(p))
c.setdefault("data", {})
c["data"]["csv_path"] = os.environ["CSV_PATH"]
c["data"]["img_dir"]  = os.environ["IMG_PATH"]
yaml.safe_dump(c, open(p, "w"), sort_keys=False)
print("[setup] benchmark.yaml csv ->", c["data"]["csv_path"])
print("[setup] benchmark.yaml img ->", c["data"]["img_dir"])
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
