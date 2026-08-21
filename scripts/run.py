"""
Run ONE benchmark config at ONE seed, end to end, and save the result.

    python scripts/run.py C09 --seed 0     # one config, one seed
    python scripts/run.py C09              # one config, all its seeds (0,1,2)
    python scripts/run.py --campaign       # every run in configs/campaign.yaml
    python scripts/run.py --campaign my.yaml   # a specific campaign list

Resolves configs/runs/<id>_*.yaml, trains + evaluates on the fixed benchmark
split, and writes:
    results/<slug>/seed<N>/metrics.json          <- the metric panel (one table row)
    results/<slug>/seed<N>/run_config.yaml       <- exactly what was run
    results/<slug>/seed<N>/benchmark_config.yaml <- the data/split definition
    results/<slug>/seed<N>/checkpoint.pt         <- best-HD weights (optional)

Dataset paths come from configs/benchmark.yaml (data.csv_path / data.img_dir);
no per-run --csv/--img_dir needed. This is the reviewer-facing unit: one command
regenerates one number, and the two saved YAMLs say precisely how.

Structure (single responsibility each):
    resolve_benchmark()  config file + CLI overrides -> BenchmarkConfig
    ResultWriter         owns the result directory and every save action
    execute_run()        skip-check -> train -> persist -> report
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bioreef.config import BenchmarkConfig, DEFAULT_CONFIG_PATH
from bioreef.run_config import RunConfig
from bioreef.training import set_seed, resolve_device
from bioreef.training.loop import train_and_evaluate

DEFAULT_CAMPAIGN_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "campaign.yaml"
)


def _git_revision() -> str:
    """Short git commit the run was produced at (with a '-dirty' suffix if the
    working tree had uncommitted changes), or 'unknown'. A bare commit hash lies
    when the code was edited but not committed — the run then records the previous
    commit as though the code were unchanged (audit #12), so flag dirtiness."""
    import subprocess
    cwd = os.path.dirname(os.path.abspath(__file__))
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, stderr=subprocess.DEVNULL,
        ).decode().strip()
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return rev


def _dataset_hash(csv_path: str) -> str:
    """SHA-256 (first 16 hex) of the metadata CSV *contents* — not its path. Two
    runs that point at the same path but a MODIFIED csv would otherwise look
    identical in provenance (audit #13). 'missing' if the file is absent."""
    import hashlib
    if not csv_path or not os.path.exists(csv_path):
        return "missing"
    h = hashlib.sha256()
    with open(csv_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _provenance(run_cfg, bench) -> dict:
    """The scientific fingerprint of a run: everything that must match for two
    seeds/results to be COMPARABLE. aggregate.py refuses to average across
    differing fingerprints; already_done() refuses to reuse a stale one.

    Includes the fixed PROTOCOL constants (weight decay, EMA, CB-Focal beta/gamma,
    crop resolution, small-object rule, augmentation) so the saved result fully
    describes the experiment even though these live in code, not the run YAML
    (audit #31/#32)."""
    from bioreef import protocol
    return {
        "code_revision": _git_revision(),
        "dataset_sha256": _dataset_hash(bench.csv_path),
        "benchmark": {
            "min_samples": bench.min_samples,
            "min_deployments": bench.min_deployments,
            "filter_placeholders": bench.filter_placeholders,
            "ratios": list(bench.ratios),
            "split_seed": bench.split_seed,
        },
        "run_config": run_cfg.to_serializable_dict(),
        "protocol": protocol.as_dict(),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_id", nargs="?", default=None,
                   help="run id, e.g. C09 or A1 (resolves configs/runs/<id>_*.yaml). "
                        "Omit with --campaign to run a whole list.")
    p.add_argument("--campaign", nargs="?", const=DEFAULT_CAMPAIGN_PATH, default=None,
                   help="run every id listed in a campaign YAML "
                        f"(default: {DEFAULT_CAMPAIGN_PATH})")
    p.add_argument("--seed", type=int, default=None,
                   help="single seed; omit to run every seed in the config")
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="benchmark config YAML")
    p.add_argument("--csv", default=None, help="override data.csv_path")
    p.add_argument("--img_dir", default=None, help="override data.img_dir")
    p.add_argument("--gpu", default=None,
                   help="GPU to use, e.g. 1 or cuda:1 or cpu (overrides config 'device')")
    p.add_argument("--batch_size", type=int, default=None,
                   help="override the config's batch size; omit to use each "
                        "run's configured value (recorded in run_config.yaml)")
    p.add_argument("--epochs", type=int, default=None,
                   help="override the config's epoch count (e.g. a length sweep); "
                        "omit to use each run's configured epochs")
    p.add_argument("--no_augment", action="store_true",
                   help="disable marine augmentation for this run (clean crops); "
                        "diagnostic / augmentation ablation on frozen backbones")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--save_checkpoint", action="store_true")
    p.add_argument("--overwrite", action="store_true",
                   help="re-run even if metrics.json already exists")
    return p.parse_args()


def resolve_benchmark(args) -> BenchmarkConfig:
    """Benchmark config from file, with CLI data-path overrides."""
    bench = BenchmarkConfig.from_yaml(args.config).apply_overrides(
        csv_path=args.csv, img_dir=args.img_dir,
    )
    if not bench.csv_path:
        raise SystemExit("no dataset CSV: set data.csv_path in the config or pass --csv")
    return bench


class ResultWriter:
    """Owns one run's result directory and all its save actions."""

    def __init__(self, results_dir, slug, seed):
        self.dir = os.path.join(results_dir, slug, f"seed{seed}")
        self.metrics_path = os.path.join(self.dir, "metrics.json")

    def already_done(self, provenance=None) -> bool:
        """A run counts as done only if metrics.json exists AND (when a current
        provenance fingerprint is supplied) the STORED fingerprint matches it. This
        stops a fixed-bug re-run from being silently skipped in favour of the stale
        result produced by the old code (audit #11). Legacy results with no stored
        provenance are treated as done (nothing to compare) but warned about."""
        if not os.path.exists(self.metrics_path):
            return False
        if provenance is None:
            return True
        try:
            with open(self.metrics_path, encoding="utf-8") as f:
                stored = json.load(f).get("provenance")
        except Exception:
            return True
        if stored is None:
            print(f"[warn] {self.metrics_path} has no provenance fingerprint "
                  "(pre-provenance result) — cannot verify it matches the current "
                  "code/data; treating as done. Use --overwrite to force a re-run.")
            return True
        if stored != provenance:
            print(f"[stale] {self.metrics_path} provenance differs from the current "
                  "code/data — re-running (the stored result is out of date).")
            return False
        return True

    def _yaml(self, obj_dict, name):
        import yaml
        self._atomic(os.path.join(self.dir, name),
                     lambda f: yaml.safe_dump(obj_dict, f, sort_keys=False))

    @staticmethod
    def _atomic(path, write_fn):
        """Write to <path>.tmp then rename onto <path>. rename is atomic on POSIX,
        so a reader/skip-check never sees a half-written file."""
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            write_fn(f)
        os.replace(tmp, path)

    def save(self, result, run_cfg, bench, model=None, idx_to_sp=None):
        os.makedirs(self.dir, exist_ok=True)
        # Provenance + checkpoint FIRST, metrics.json LAST. already_done() keys
        # on metrics.json, so writing it last makes its existence mean "the whole
        # result directory is complete" — a crash mid-save leaves no metrics.json
        # and the run re-executes instead of being skipped half-written.
        self._yaml(run_cfg.to_serializable_dict(), "run_config.yaml")
        self._yaml(bench.__dict__, "benchmark_config.yaml")
        if model is not None:
            torch.save({"model": model.state_dict(), "idx_to_sp": idx_to_sp,
                        "run_config": run_cfg.__dict__, "benchmark_config": bench.__dict__},
                       os.path.join(self.dir, "checkpoint.pt"))
        # Stamp the full provenance fingerprint (code rev + dataset hash + benchmark
        # def + resolved run config). already_done() and aggregate.py compare it, so
        # a result from stale code/data is detected instead of silently reused/averaged.
        prov = _provenance(run_cfg, bench)
        result = dict(result, code_revision=prov["code_revision"], provenance=prov)
        self._atomic(self.metrics_path, lambda f: json.dump(result, f, indent=2))


def execute_run(run_cfg, bench, seed, args, device):
    writer = ResultWriter(args.results_dir, run_cfg.slug, seed)
    # Skip only if a MATCHING result exists — provenance-aware, so a re-run after a
    # code/data change is not silently skipped in favour of the stale result.
    if writer.already_done(_provenance(run_cfg, bench)) and not args.overwrite:
        print(f"[skip] {run_cfg.slug} seed{seed} already done ({writer.metrics_path})")
        return

    if args.epochs is not None:
        run_cfg.epochs = args.epochs   # length sweep / diagnostic override
    if args.batch_size is not None:
        # Write the override BACK into run_cfg before it is serialized, so
        # run_config.yaml records the batch size that actually trained rather
        # than the one the YAML happened to declare.
        run_cfg.batch_size = args.batch_size
    if args.no_augment:
        run_cfg.augment = False        # augmentation ablation / diagnostic

    # Re-validate AFTER overrides: --epochs can drop below warmup_epochs, which
    # the YAML-load validation could not have caught. The resolved config is
    # what actually trains, so it is what must be valid.
    run_cfg._validate("<resolved runtime config>")

    print(f"\n{'='*60}\n[run] {run_cfg.slug}  seed={seed}  family={run_cfg.model_family}\n{'='*60}")
    set_seed(seed)
    test_metrics, val_metrics, model, idx_to_sp, num_classes = train_and_evaluate(
        run_cfg, bench, seed, device,
        batch_size=run_cfg.batch_size, num_workers=args.num_workers,
    )

    result = {
        "run_id": run_cfg.run_id, "slug": run_cfg.slug,
        "model_family": run_cfg.model_family, "seed": seed,
        "num_classes": num_classes, "test": test_metrics, "val_best": val_metrics,
    }
    writer.save(result, run_cfg, bench,
                model=model if args.save_checkpoint else None, idx_to_sp=idx_to_sp)

    print(f"[done] {run_cfg.slug} seed{seed}: "
          f"macroAcc {test_metrics['macro_accuracy']:.4f} | "
          f"HD {test_metrics['mean_hd']:.4f} | top1 {test_metrics['top1_accuracy']:.4f}  "
          f"-> {writer.metrics_path}")


def load_campaign(path):
    """Read a campaign YAML -> (run_ids, campaign_seeds_or_None)."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    run_ids = [str(r) for r in (data.get("runs") or [])]
    if not run_ids:
        raise SystemExit(f"campaign {path} lists no runs (under 'runs:')")
    return run_ids, data.get("seeds")


def seeds_for(run_cfg, cli_seed, campaign_seeds=None):
    """Precedence: --seed (single) > campaign seeds > the run config's own seeds."""
    if cli_seed is not None:
        return [cli_seed]
    if campaign_seeds is not None:
        return list(campaign_seeds)
    return run_cfg.seeds


def run_batch(run_ids, campaign_seeds, bench, args, device):
    """Execute a list of run ids across their seeds, resumable (skips done)."""
    print(f"[campaign] {len(run_ids)} runs: {', '.join(run_ids)}")
    for i, rid in enumerate(run_ids, 1):
        run_cfg = RunConfig.find(rid)
        seeds = seeds_for(run_cfg, args.seed, campaign_seeds)
        print(f"\n[campaign {i}/{len(run_ids)}] {rid} seeds={seeds}")
        for seed in seeds:
            execute_run(run_cfg, bench, seed, args, device)


def main():
    args = parse_args()
    bench = resolve_benchmark(args)
    device = resolve_device(args.gpu, bench.device)   # --gpu > config 'device' > auto
    print(f"[device] {device}")

    if args.campaign:
        run_ids, campaign_seeds = load_campaign(args.campaign)
        run_batch(run_ids, campaign_seeds, bench, args, device)
    elif args.run_id:
        run_cfg = RunConfig.find(args.run_id)
        for seed in seeds_for(run_cfg, args.seed):
            execute_run(run_cfg, bench, seed, args, device)
    else:
        raise SystemExit("give a run id (e.g. C09) or --campaign")


if __name__ == "__main__":
    main()
