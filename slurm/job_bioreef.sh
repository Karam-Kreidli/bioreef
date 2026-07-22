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

module load anaconda3 2>/dev/null || module load anaconda3/3.11 2>/dev/null || true
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate "$ENVNAME" 2>/dev/null || source activate "$ENVNAME"

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

# --num_workers matches the CPUs we actually reserved. Asking for more workers
# than allocated CPUs makes them contend and run SLOWER, not faster.
srun python scripts/run.py "$RUN_ID" \
  --seed "$SEED" \
  --gpu 0 \
  --num_workers "$SLURM_CPUS_PER_TASK" \
  --save_checkpoint

echo "=== $(date) | done ==="
