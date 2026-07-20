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

    def already_done(self) -> bool:
        return os.path.exists(self.metrics_path)

    def _yaml(self, obj_dict, name):
        import yaml
        with open(os.path.join(self.dir, name), "w", encoding="utf-8") as f:
            yaml.safe_dump(obj_dict, f, sort_keys=False)

    def save(self, result, run_cfg, bench, model=None, idx_to_sp=None):
        os.makedirs(self.dir, exist_ok=True)
        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        self._yaml(run_cfg.to_serializable_dict(), "run_config.yaml")
        self._yaml(bench.__dict__, "benchmark_config.yaml")
        if model is not None:
            torch.save({"model": model.state_dict(), "idx_to_sp": idx_to_sp,
                        "run_config": run_cfg.__dict__, "benchmark_config": bench.__dict__},
                       os.path.join(self.dir, "checkpoint.pt"))


def execute_run(run_cfg, bench, seed, args, device):
    writer = ResultWriter(args.results_dir, run_cfg.slug, seed)
    if writer.already_done() and not args.overwrite:
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
