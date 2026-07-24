#!/bin/bash
# ---------------------------------------------------------------------------
# job_matanet.sh — train + test MATANet (C08) on ONE GPU via Slurm.
#
# This is step 5 of matanet/README.md. It runs THEIR repo's B1/C1 scripts in the
# separate `matanet` conda env, on the split you exported with export_ozfish.py.
# The export (step 4, bioreef env) and ingest (step 6, bioreef env) run
# separately on the login node — they are fast and need no GPU.
#
#   sbatch slurm/job_matanet.sh 0     # seed 0
#   sbatch slurm/job_matanet.sh 1     # seed 1
#
# Prereqs (README steps 0-4 done once): matanet-repo cloned + patched, `matanet`
# env created with their requirements, DINOv2-large pre-fetched, split exported.
# ---------------------------------------------------------------------------
#SBATCH --job-name=matanet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=dcv-1gpu-g5-ond
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=YOUR_EMAIL@sharjah.ac.ae
#SBATCH --mail-type=END,FAIL

set -euo pipefail

SEED="${1:-0}"
ENVNAME="${ENVNAME:-matanet}"                       # THEIR env, not bioreef
MATANET_REPO="${MATANET_REPO:-$HOME/matanet-repo}"
CONFIG="$HOME/bioreef-classify/matanet/ozfish_data/ozfish_config_seed${SEED}.yaml"

echo "=== $(date) | job $SLURM_JOB_ID | MATANet seed $SEED ==="
hostname
nvidia-smi

# --- activate conda (same hardened pattern as job_bioreef.sh) --------------
# set +eu: conda's own hook scripts reference unset vars (CONDA_MKL_...), which
# under set -u abort the job; and a failing activate under set -e kills it
# silently. Source conda.sh directly, verify, re-enable.
set +eu
module load anaconda3 2>/dev/null || module load anaconda3/3.11 2>/dev/null
for CSH in "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" \
           "$HOME/.conda/etc/profile.d/conda.sh" \
           "/opt/conda/etc/profile.d/conda.sh"; do
  [ -f "$CSH" ] && { . "$CSH"; break; }
done
conda activate "$ENVNAME" || source activate "$ENVNAME"
set -eu
if [ "$(basename "${CONDA_PREFIX:-none}")" != "$ENVNAME" ]; then
  echo "FATAL: could not activate conda env '$ENVNAME' (CONDA_PREFIX=${CONDA_PREFIX:-unset})." >&2
  echo "  envs: $(conda env list 2>&1 | tr '\n' ';')" >&2
  exit 1
fi
echo "[env] active: $CONDA_PREFIX | python $(python --version 2>&1)"
export PYTHONNOUSERSITE=1
# Their DINOv2-large was pre-fetched on the login node; force offline so a cache
# miss fails fast instead of hanging on the (usually absent) compute-node net.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Fine-tuned DINOv2-large at batch 16 fills the 22 GB A10G almost exactly: the
# first attempt died allocating 12 MiB with 1.3 GiB reserved-but-unallocated,
# i.e. lost to fragmentation. expandable_segments reclaims that margin WITHOUT
# touching batch size / LR / augmentation, so published-method parity is intact.
# (If it still OOMs, the parity-preserving next step is batch_size 8 +
# accumulate_grad_batches 2 in the config — effective batch stays 16.)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[ -f "$CONFIG" ] || { echo "FATAL: config not found: $CONFIG" >&2
  echo "  run export first: python matanet/export_ozfish.py --seed $SEED" >&2; exit 1; }

python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; \
print('GPU:', torch.cuda.get_device_name(0), \
round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')"

# --- train then test, from THEIR repo, on OUR config -----------------------
# Do NOT change their batch size / augmentation — parity with the published
# method is the whole point of an "official repo" baseline.
cd "$MATANET_REPO"
echo "[matanet] B1.BuildModel (train) seed $SEED"
python B1.BuildModel.py --config "$CONFIG"
echo "[matanet] C1.TestModel (predict) seed $SEED"
python C1.TestModel.py  --config "$CONFIG"

echo "=== $(date) | done. Now ingest (bioreef env, login node): ==="
echo "  conda activate bioreef"
echo "  python ~/bioreef-classify/matanet/ingest_predictions.py --data_dir ~/bioreef-classify/matanet/ozfish_data --seed $SEED"
