#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# submit_campaign.sh — queue the whole panel as independent one-GPU jobs.
#
#   bash slurm/submit_campaign.sh              # every run in campaign.yaml, seeds 0 1 2
#   bash slurm/submit_campaign.sh 0            # seed 0 only
#   bash slurm/submit_campaign.sh 0 "C09 A9"   # specific runs, seed 0
#   DRY=1 bash slurm/submit_campaign.sh        # print what WOULD be submitted
#
# One job per (run, seed) rather than one big job: the scheduler backfills small
# single-GPU jobs far sooner than a long multi-GPU one, a crash costs you one
# cell instead of the campaign, and run.py already skips a (run, seed) whose
# metrics.json exists — so re-running this after failures resumes cleanly.
# ---------------------------------------------------------------------------
set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/bioreef-classify}"
cd "$WORKDIR"

SEEDS="${1:-0 1 2}"
RUNS="${2:-}"

if [ -z "$RUNS" ]; then
  # Read the campaign list from the YAML rather than hardcoding ids here — the
  # ids were renumbered once already and a stale duplicate list would silently
  # queue the wrong experiments.
  RUNS=$(python - <<'PY'
import yaml
c = yaml.safe_load(open("configs/campaign.yaml"))
print(" ".join(c["runs"]))
PY
)
fi

echo "runs : $RUNS"
echo "seeds: $SEEDS"
echo

mkdir -p logs
n=0
for run in $RUNS; do
  for seed in $SEEDS; do
    if [ "${DRY:-0}" = "1" ]; then
      echo "would submit: $run seed $seed"
    else
      jid=$(sbatch --parsable --job-name="br_${run}_s${seed}" \
                   slurm/job_bioreef.sh "$run" "$seed")
      echo "submitted $run seed $seed -> job $jid"
    fi
    n=$((n+1))
  done
done

echo
echo "$n job(s). Watch with:  squeue -u $USER"
echo "Results land in:        results/<slug>/seed<N>/metrics.json"
echo "Aggregate when done:    python scripts/aggregate.py"
