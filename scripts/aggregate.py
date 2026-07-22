"""
Aggregate all results/<slug>/seed<N>/metrics.json into the benchmark table.

    python scripts/aggregate.py

Three phases, kept separate (single responsibility each):
    load_results()      read per-run JSON   -> {slug: RunGroup}
    RunGroup.summary()  compute mean +/- std (pure)
    render_*()          format Markdown / JSON

Outputs:
    RESULTS.md            <- the paper's main benchmark table (mean +/- std)
    results/summary.json  <- machine-readable aggregate

A reviewer runs this to reproduce Table 6.1 from the per-run JSON files.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Metrics shown in the main table (json key -> column header), in report order.
TABLE_METRICS = [
    ("macro_accuracy", "MacroAcc"),
    ("top1_accuracy", "Top-1"),
    ("top5_accuracy", "Top-5"),
    ("mean_hd", "HD"),
    ("mistake_severity", "MistSev"),
    ("genus_accuracy", "GenusAcc"),
    ("family_accuracy", "FamilyAcc"),
]
GROUPS = ("head", "medium", "tail")


# --- statistics (pure) -------------------------------------------------------

def mean_std(xs: List[float]) -> Tuple[float, float]:
    """Sample mean and std. (nan, nan) for empty; std 0 for a single value."""
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    mu = sum(xs) / n
    if n == 1:
        # std is UNDEFINED for one sample, not zero. Returning 0.0 renders as
        # "mean ± 0.000", which reads as a stable estimate. nan -> fmt shows the
        # bare mean instead, honestly signalling a single seed.
        return mu, float("nan")
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return mu, math.sqrt(var)


def fmt(stat: Tuple[float, float]) -> str:
    mu, sd = stat
    if math.isnan(mu):
        return "--"
    if math.isnan(sd):          # single seed: mean only, no fake ±0.000
        return f"{mu:.3f}"
    return f"{mu:.3f}±{sd:.3f}"


# --- data model --------------------------------------------------------------

@dataclass
class RunGroup:
    """All seeds of one config, and the per-metric samples gathered across them."""
    run_id: str
    model_family: str
    seeds: List[int] = field(default_factory=list)
    metric_samples: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    group_samples: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))

    def add(self, record: dict) -> None:
        """Fold one seed's metrics.json record into this group."""
        self.seeds.append(record.get("seed"))
        test = record.get("test", {})
        for key, _ in TABLE_METRICS:
            if key in test:
                self.metric_samples[key].append(test[key])
        for grp, val in (test.get("group_accuracy") or {}).items():
            self.group_samples[grp].append(val)

    def metric(self, key: str) -> Tuple[float, float]:
        return mean_std(self.metric_samples.get(key, []))

    def group(self, grp: str) -> Tuple[float, float]:
        return mean_std(self.group_samples.get(grp, []))

    def summary(self) -> dict:
        """Machine-readable mean/std for summary.json."""
        as_stat = lambda s: dict(zip(("mean", "std"), s))
        return {
            "run_id": self.run_id,
            "model_family": self.model_family,
            "n_seeds": len(self.seeds),
            "seeds": sorted(s for s in self.seeds if s is not None),
            "metrics": {k: as_stat(self.metric(k)) for k in self.metric_samples},
            "group_accuracy": {g: as_stat(self.group(g)) for g in self.group_samples},
        }


# --- load (I/O only) ---------------------------------------------------------

def load_results(results_dir: str) -> Dict[str, RunGroup]:
    """Read every results/<slug>/seed<N>/metrics.json into {slug: RunGroup}."""
    groups: Dict[str, RunGroup] = {}
    if not os.path.isdir(results_dir):
        return groups
    for slug in sorted(os.listdir(results_dir)):
        slug_dir = os.path.join(results_dir, slug)
        if not os.path.isdir(slug_dir):
            continue
        for seed_dir in sorted(os.listdir(slug_dir)):
            mpath = os.path.join(slug_dir, seed_dir, "metrics.json")
            if not os.path.exists(mpath):
                continue
            with open(mpath) as f:
                record = json.load(f)
            group = groups.get(slug)
            if group is None:
                group = groups[slug] = RunGroup(
                    run_id=record.get("run_id", slug),
                    model_family=record.get("model_family", "?"),
                )
            group.add(record)
    return groups


def _run_key(run_id: str) -> Tuple:
    """Natural sort key: split 'A10' -> ('A', 10) so A2 precedes A10 (string
    sort gives A1, A10, A11, A2...). Panel (C..) sorts before ablations (A..)."""
    import re
    m = re.fullmatch(r"([A-Za-z]+)(\d+)", run_id)
    prefix, num = (m.group(1), int(m.group(2))) if m else (run_id, -1)
    return (0 if prefix.startswith("C") else 1, prefix, num)


def report_order(groups: Dict[str, RunGroup]) -> List[str]:
    """Panel configs (C..) first, then ablations (A..), each in numeric order."""
    return sorted(groups, key=lambda s: _run_key(groups[s].run_id))


# --- render (formatting only) ------------------------------------------------

def _table(headers: List[str], rows: List[List[str]]) -> List[str]:
    return (["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
            + ["| " + " | ".join(r) + " |" for r in rows])


def render_markdown(groups: Dict[str, RunGroup]) -> str:
    order = report_order(groups)
    lines = [
        "# Benchmark Results", "",
        "Mean ± std over seeds. Generated by `scripts/aggregate.py` from the "
        "per-run `results/<slug>/seed<N>/metrics.json` files.", "",
    ]

    main_rows = [
        [g.run_id, g.model_family, str(len(g.seeds))]
        + [fmt(g.metric(k)) for k, _ in TABLE_METRICS]
        for g in (groups[s] for s in order)
    ]
    lines += _table(["Run", "Family", "Seeds"] + [h for _, h in TABLE_METRICS], main_rows)

    lines += ["", "## Head / Medium / Tail accuracy (macro, by train frequency)", ""]
    ht_rows = [[g.run_id] + [fmt(g.group(grp)) for grp in GROUPS]
               for g in (groups[s] for s in order)]
    lines += _table(["Run", "Head", "Medium", "Tail"], ht_rows)
    return "\n".join(lines) + "\n"


def render_summary(groups: Dict[str, RunGroup]) -> str:
    return json.dumps({slug: g.summary() for slug, g in groups.items()}, indent=2)


# --- orchestration -----------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--out_md", default="RESULTS.md")
    args = p.parse_args()

    groups = load_results(args.results_dir)
    if not groups:
        print(f"no results under {args.results_dir}/ — run a config first "
              "(python scripts/run.py C09 --seed 0)")
        return

    with open(args.out_md, "w", encoding="utf-8", newline="\n") as f:
        f.write(render_markdown(groups))
    with open(os.path.join(args.results_dir, "summary.json"), "w", encoding="utf-8") as f:
        f.write(render_summary(groups))

    n_runs = sum(len(g.seeds) for g in groups.values())
    print(f"aggregated {len(groups)} config(s) ({n_runs} runs) -> "
          f"{args.out_md} + {args.results_dir}/summary.json")


if __name__ == "__main__":
    main()
