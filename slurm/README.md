# Running on the AWS Slurm cluster

Three scripts. Run them in this order.

```bash
# --- 1. clone FIRST, on the cluster LOGIN node --------------------------
# (the repo must exist before you rsync data into it)
ssh -X oelmutasim@44.210.222.21
module load anaconda3 && source ~/.bashrc
git clone https://github.com/Karam-Kreidli/bioreef.git ~/bioreef-classify

# --- 2. send the data, FROM YOUR VM (a separate terminal) ---------------
# rsync, not tar+scp: needs no second copy on the VM's disk (the crops are
# already-compressed PNGs, so tar.gz saves almost nothing) and it RESUMES if
# the connection drops — which matters at ~76k files.
# Paths mirror the VM layout: data/ at the repo ROOT (not inside bioreef/,
# which is the Python package). Trailing slash on the source matters —
# "frames/" copies the CONTENTS; "frames" would nest it as frames/frames/.
cd ~/bioreef-classify        # or wherever the repo is on the VM
rsync -avP data/frames/ \
  oelmutasim@44.210.222.21:~/bioreef-classify/data/frames/
rsync -avP data/metadata/ \
  oelmutasim@44.210.222.21:~/bioreef-classify/data/metadata/

# --- 3. back on the login node ------------------------------------------
cd ~/bioreef-classify
HF_TOKEN=hf_xxxxx bash slurm/setup_slurm.sh   # pass it here, don't edit the script

# --- then, per experiment ----------------------------------------------
sbatch slurm/job_bioreef.sh C09 0        # one run, one seed
bash   slurm/submit_campaign.sh 0        # whole panel, seed 0
bash   slurm/submit_campaign.sh          # whole panel, seeds 0 1 2  (54 jobs)

squeue -u $USER                          # status
tail -f logs/<jobid>.out                 # watch
scancel <jobid>                          # kill
```

## Where this differs from the cluster's sample job script

The sample script in the cluster docs is a generic template. Four changes matter
for this pipeline:

**`--ntasks=24 --cpus-per-task=1` → `--ntasks=1 --cpus-per-task=8`.** The sample
asks for 24 independent processes. `run.py` is a single training process that
wants many *threads* for the dataloader, not 24 copies of itself. Twenty-four
tasks would launch twenty-four full training runs on one GPU and OOM instantly.

**No DDP environment variables.** The sample exports `WORLD_SIZE`/`RANK`/
`MASTER_ADDR`/`MASTER_PORT`. `run.py` is single-process; setting those can push
torch into distributed init for a job with nothing to distribute. They're needed
only for the multi-GPU `train.py` path.

**`--gres=gpu:1`, not `gpu:4`.** One GPU per job, many jobs. The scheduler
backfills small jobs much sooner than a 4-GPU one, and a crash costs one cell of
the results table instead of the campaign.

**Offline HuggingFace.** Compute nodes usually have no outbound internet, and
DINOv3 is a *gated* 350 MB download. `setup_slurm.sh` pre-caches every backbone
on the login node; the job then sets `HF_HUB_OFFLINE=1` so a cache miss fails in
seconds with a clear error rather than hanging on a network timeout mid-run.

## Things that will bite you

**`nvidia-smi` shows nothing on the login node.** That's normal — no GPU there.
Don't conclude the cluster is broken. And don't train on the login node; long
processes there get killed.

**`conda create -n myenv6` with no packages creates an env with no Python.** The
first `pip install` then quietly installs into `base`. `setup_slurm.sh` pins
`python=3.11` for this reason.

**`strict_images: true` is on.** If the scp dropped crops, `export_split.py` in
setup fails loudly instead of quietly training on a smaller, different benchmark.
That is the desired behaviour. Re-sync with `rsync -avP` (which resumes) rather
than restarting scp.

**Check the partition name.** The docs mention `gpu-g5-ond` (4-GPU) and
`dcv-1gpu-g5-ond` (1-GPU). The job script uses the 1-GPU one. Confirm with
`sinfo -s` — a wrong partition means the job sits `PD` forever.

**`--time=24:00:00` is a guess.** If jobs get killed at the limit, raise it; if
they never start, lower it (shorter jobs backfill sooner). Check the max with
`scontrol show partition dcv-1gpu-g5-ond`.

## Memory

A11 (full fine-tune, all 12 blocks + embeddings across four context streams) is
the tightest fit on 24 GB. If it OOMs, the honest fix is gradient accumulation
rather than a smaller batch — the panel keeps batch 32 for comparability, and
changing it for one run makes that row incomparable to the rest.

Smoke-test one config before queueing 54:

```bash
sbatch slurm/job_bioreef.sh C09 0     # let it finish an epoch, check logs/<jid>.out
```
