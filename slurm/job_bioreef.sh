#!/bin/bash
# ---------------------------------------------------------------------------
# job_bioreef.sh — run ONE bioreef config on ONE GPU via Slurm.
#
#   sbatch slurm/job_bioreef.sh C09        # run C09, seed 0
#   sbatch slurm/job_bioreef.sh A11 1      # run A11, seed 1
#   sbatch slurm/job_bioreef.sh C05 2      # run C05, seed 2
#
# Check on it:   squeue -u $USER
# Watch it:      tail -f logs/<jobid>.out
# Kill it:       scancel <jobid>
# ---------------------------------------------------------------------------
#SBATCH --job-name=bioreef
#SBATCH --nodes=1
#SBATCH --ntasks=1
# cpus-per-task drives the dataloader workers. This pipeline is DATA-LOADING
# BOUND (76k JPEG decodes + 4 context crops each), so CPU count matters as much
# as the GPU. The cluster template used --ntasks=24 --cpus-per-task=1: that asks
# for 24 separate processes, which is wrong for a single-process training job.
# One task with many CPUs is what you want.
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=dcv-1gpu-g5-ond
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=YOUR_EMAIL@sharjah.ac.ae
#SBATCH --mail-type=END,FAIL

set -euo pipefail

RUN_ID="${1:?usage: sbatch job_bioreef.sh <RUN_ID> [SEED]   e.g. C09, A11}"
SEED="${2:-0}"
WORKDIR="${WORKDIR:-$HOME/bioreef-classify}"
ENVNAME="${ENVNAME:-bioreef}"

echo "=== $(date) | job $SLURM_JOB_ID | $RUN_ID seed $SEED ==="
hostname
nvidia-smi

# --- activate conda WITHOUT letting set -e kill us silently ---------------
# The previous version died right after nvidia-smi with empty .err: under
# `set -e`, a failing `$(conda shell.bash hook)` command-substitution aborts the
# script before the `|| true` is even applied, and `conda activate` returning
# non-zero does the same. Disable errexit for the whole activation, source
# conda's profile script directly (the canonical non-interactive init), then
# VERIFY and re-enable. Nothing here can abort without printing why.
set +e
module load anaconda3 2>/dev/null || module load anaconda3/3.11 2>/dev/null
# Prefer sourcing conda.sh over the shell hook — it works in a bare batch shell.
for CSH in "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" \
           "$HOME/.conda/etc/profile.d/conda.sh" \
           "/opt/conda/etc/profile.d/conda.sh"; do
  [ -f "$CSH" ] && { . "$CSH"; break; }
done
conda activate "$ENVNAME" || source activate "$ENVNAME"
ACT_RC=$?
set -e
if [ "$(basename "${CONDA_PREFIX:-none}")" != "$ENVNAME" ]; then
  echo "FATAL: could not activate conda env '$ENVNAME' (rc=$ACT_RC, CONDA_PREFIX=${CONDA_PREFIX:-unset})." >&2
  echo "  conda base : $(conda info --base 2>&1)" >&2
  echo "  conda.sh   : ${CSH:-none tried}" >&2
  echo "  envs       : $(conda env list 2>&1 | tr '\n' ';')" >&2
  exit 1
fi
echo "[env] active: $CONDA_PREFIX | python $(python --version 2>&1)"

# Ignore ~/.local: a stale user-site there (old torch/ultralytics) would
# otherwise shadow the conda env and import a torch needing libcusparseLt.so.0.
export PYTHONNOUSERSITE=1

cd "$WORKDIR"

# Compute nodes often have no outbound internet. setup_slurm.sh pre-cached every
# backbone, so force offline mode: a cache miss then fails immediately with a
# clear message instead of hanging on a network timeout inside the training loop.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Single-GPU. The DDP env vars in the cluster's sample script (WORLD_SIZE/RANK/
# MASTER_ADDR/...) are NOT set here on purpose: run.py is a single-process
# entry point, and exporting WORLD_SIZE=1 with RANK=0 can push torch into
# distributed init for a job that has nothing to distribute.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python -c "import torch; assert torch.cuda.is_available(), 'no CUDA on compute node'; \
print('GPU:', torch.cuda.get_device_name(0), \
round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')"

# --- stage frames to node-local disk -------------------------------------
# $HOME/data/frames is on a NETWORK filesystem (df shows 127.0.0.1:/, an
# exabyte-scale distributed FS). Training reads ~49k small PNGs every epoch;
# over NFS the cold first pass alone took >12 min. Copy them ONCE to the
# compute node's local scratch and read from there — later epochs then hit
# local disk, not the network. If no local scratch exists or the copy fails,
# fall back to the network path so the job still runs.
SRC_FRAMES="$WORKDIR/data/frames"
LOCAL_BASE="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/bioreef_$SLURM_JOB_ID"
LOCAL_FRAMES="$LOCAL_BASE/frames"
IMG_DIR="$SRC_FRAMES"
# Report what we have to work with, and DON'T hide cp's error (an earlier run
# swallowed it and we could not tell space-exhaustion from a permission fault).
NEED_KB=$(du -sk "$SRC_FRAMES" 2>/dev/null | cut -f1)
AVAIL_KB=$(df -Pk "$(dirname "$LOCAL_BASE")" 2>/dev/null | awk 'NR==2{print $4}')
echo "[stage] source ${NEED_KB:-?} KB | scratch $(dirname "$LOCAL_BASE") avail ${AVAIL_KB:-?} KB"
if [ -n "$NEED_KB" ] && [ -n "$AVAIL_KB" ] && [ "$AVAIL_KB" -lt "$NEED_KB" ]; then
  echo "[stage] scratch too small for frames; using network path"
elif mkdir -p "$LOCAL_FRAMES" 2>/dev/null; then
  echo "[stage] copying frames -> $LOCAL_FRAMES"
  if cp -r "$SRC_FRAMES/." "$LOCAL_FRAMES/"; then     # stderr now visible in .err
    n=$(find "$LOCAL_FRAMES" -type f | wc -l)
    echo "[stage] staged $n files to node-local disk"
    IMG_DIR="$LOCAL_FRAMES"
  else
    echo "[stage] copy failed (see .err); using network path $SRC_FRAMES"
  fi
else
  echo "[stage] could not create $LOCAL_FRAMES; using network path"
fi
# Clean up the local copy on exit (scratch is usually auto-wiped, but be tidy).
trap 'rm -rf "$LOCAL_BASE" 2>/dev/null || true' EXIT

# Plain `python`, NOT `srun python`. This is a single-node, single-task job, so
# it runs directly under the sbatch allocation. Wrapping it in srun launches a
# job STEP, which on this cluster failed with "Task launch ... Unspecified
# error" before Python even started. srun buys nothing for a 1-task job.
#
# --num_workers matches the CPUs we actually reserved. Asking for more workers
# than allocated CPUs makes them contend and run SLOWER, not faster.
# --img_dir points at the staged copy; benchmark.yaml is left untouched so the
# benchmark DEFINITION does not change, only where the pixels are read from.
python scripts/run.py "$RUN_ID" \
  --seed "$SEED" \
  --gpu 0 \
  --img_dir "$IMG_DIR" \
  --num_workers "$SLURM_CPUS_PER_TASK" \
  --save_checkpoint

echo "=== $(date) | done ==="
