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
from typing import Dict, List, Optional, Tuple

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

def mean_std(xs: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Sample mean and std. (None, None) for empty; (mu, None) for one value
    (std undefined); (mu, sample-std) for n>=2. None (not NaN) so summary.json
    stays valid JSON."""
    n = len(xs)
    if n == 0:
        return None, None
    mu = sum(xs) / n
    if n == 1:
        # std is UNDEFINED for one sample, not zero. Returning 0.0 renders as
        # "mean ± 0.000", which reads as a stable estimate. None -> fmt shows the
        # bare mean, and summary.json gets `null` (valid JSON) rather than NaN
        # (which Python emits but strict parsers reject).
        return mu, None
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return mu, math.sqrt(var)


def fmt(stat: Tuple[Optional[float], Optional[float]]) -> str:
    mu, sd = stat
    if mu is None:
        return "--"
    if sd is None:              # single seed: mean only, no fake ±0.000
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

    # provenance fingerprints seen across this group's seeds (audit #10). Two seeds
    # are only comparable if their code_revision + dataset hash + benchmark def match.
    provenances: Dict[int, dict] = field(default_factory=dict)
    code_revisions: Dict[int, str] = field(default_factory=dict)

    def add(self, record: dict) -> None:
        """Fold one seed's metrics.json record into this group."""
        seed = record.get("seed")
        self.seeds.append(seed)
        # Track provenance so mismatched seeds can be caught before averaging.
        prov = record.get("provenance")
        # compare the code/data/benchmark part, not the resolved run_config (seed
        # legitimately differs there) — pull the comparable subset.
        if prov is not None:
            self.provenances[seed] = {
                "code_revision": prov.get("code_revision"),
                "dataset_sha256": prov.get("dataset_sha256"),
                "benchmark": prov.get("benchmark"),
            }
        self.code_revisions[seed] = record.get("code_revision", "unknown")
        test = record.get("test", {})
        for key, _ in TABLE_METRICS:
            if key in test:
                self.metric_samples[key].append(test[key])
        for grp, val in (test.get("group_accuracy") or {}).items():
            self.group_samples[grp].append(val)

    def provenance_issue(self) -> Optional[Tuple[str, str]]:
        """Return (severity, description) if this group's seeds are not perfectly
        comparable, else None. Severity is 'fatal' or 'warn'.

        The distinction matters for --strict-provenance (see main()):
          - 'fatal' — the DATASET or BENCHMARK DEFINITION differs across seeds.
            This genuinely invalidates the average (different data / different
            protocol), so strict mode aborts.
          - 'warn'  — only the CODE REVISION is missing or differs across seeds.
            The panel was trained over weeks of active development; the git rev
            changed between sessions mostly for docs/config/test commits that do
            not touch the training or eval path (audit-verified — see the
            reproducibility note in README/RESULTS). The numbers are valid; the
            fingerprints just are not bit-identical. Strict mode warns, not abort.

        Fingerprint comparison uses dataset_sha256 + benchmark only, so a code-rev
        difference alone is never fatal.
        """
        provs = self.provenances
        revs = sorted(set(self.code_revisions.values()))

        # Fatal: the data/protocol themselves differ across the fingerprinted seeds.
        data_bench = {
            json.dumps({"dataset_sha256": p.get("dataset_sha256"),
                        "benchmark": p.get("benchmark")}, sort_keys=True)
            for p in provs.values()
        }
        if len(data_bench) > 1:
            return ("fatal",
                    f"seeds have DIFFERENT dataset/benchmark fingerprints — "
                    f"averaging them mixes non-comparable runs. "
                    f"code_revisions: {revs}")

        # Warn: provenance missing on some seeds (pre-provenance legacy results).
        if len(provs) < len(self.seeds):
            missing = len(self.seeds) - len(provs)
            return ("warn",
                    f"{missing}/{len(self.seeds)} seed(s) have no provenance "
                    f"fingerprint (pre-provenance results); code_revisions seen: "
                    f"{revs} — dataset/benchmark unverifiable but code path is "
                    f"stable across these revs (see README reproducibility note)")

        # Warn: data/benchmark match, only the code revision differs.
        if len(revs) > 1:
            return ("warn",
                    f"seeds share dataset/benchmark but differ in code_revision "
                    f"{revs} — dev-history drift, training/eval path unchanged "
                    f"(see README reproducibility note)")
        return None

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
        "> **Reproducibility note.** The panel was trained over an extended "
        "development period, so a run's three seeds may carry different — or, for "
        "the earliest runs, absent — `code_revision` fingerprints. The differing "
        "commits are documentation, config, and test changes that do not touch the "
        "training or evaluation path; the **dataset and benchmark definition are "
        "identical across all seeds**. `aggregate.py --strict-provenance` therefore "
        "aborts only on a dataset/benchmark mismatch and treats code-revision drift "
        "as a warning. Newly added runs stamp a full fingerprint; to reproduce a "
        "single number from scratch under one revision, re-run that config with "
        "`scripts/run.py`.", "",
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
    # allow_nan=False: reject any stray NaN/Infinity instead of emitting the
    # non-standard tokens Python's encoder would otherwise write, keeping
    # summary.json parseable by strict readers.
    return json.dumps({slug: g.summary() for slug, g in groups.items()},
                      indent=2, allow_nan=False)


# --- orchestration -----------------------------------------------------------

def _campaign_slugs(campaign_path: str) -> Optional[set]:
    """Read the declared run ids from a campaign YAML and map to result slugs.
    Returns a set of slugs to KEEP, or None if the file can't be read. This is how
    the paper table excludes deployment/non-declared runs (D1/D2) and any post-hoc
    config not in the campaign (audit #14)."""
    import yaml
    from bioreef.run_config import RunConfig
    with open(campaign_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    ids = data.get("runs") or data.get("run_ids") or []
    slugs = set()
    for rid in ids:
        try:
            slugs.add(RunConfig.find(rid).slug)
        except Exception:
            slugs.add(rid)   # fall back to the raw id if the config can't resolve
    return slugs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--out_md", default="RESULTS.md")
    p.add_argument("--campaign", default=None,
                   help="campaign YAML: include ONLY its declared runs in the table "
                        "(excludes D1/D2 + any config not in the campaign). Omit to "
                        "aggregate every result found.")
    p.add_argument("--strict-provenance", action="store_true",
                   help="hard-fail (instead of warn) if any config's seeds have "
                        "mismatched or missing provenance fingerprints")
    args = p.parse_args()

    groups = load_results(args.results_dir)
    if not groups:
        print(f"no results under {args.results_dir}/ — run a config first "
              "(python scripts/run.py C09 --seed 0)")
        return

    # --- campaign filter (audit #14) ---------------------------------------
    if args.campaign:
        keep = _campaign_slugs(args.campaign)
        dropped = [s for s in groups if s not in keep]
        groups = {s: g for s, g in groups.items() if s in keep}
        if dropped:
            print(f"[campaign] excluded {len(dropped)} non-declared config(s): "
                  f"{sorted(dropped)}")

    # --- provenance check (audit #10) --------------------------------------
    # Two severities (see ResultGroup.provenance_issue):
    #   fatal — dataset/benchmark differ across seeds (invalidates the average).
    #   warn  — only the code revision is missing/mixed (dev-history drift; the
    #           training/eval path is unchanged across those revs, so numbers hold).
    # --strict-provenance aborts on FATAL only; code-rev drift is always a warning.
    issues = {slug: res for slug, res in
              ((s, g.provenance_issue()) for s, g in groups.items()) if res}
    fatal = {s: msg for s, (sev, msg) in issues.items() if sev == "fatal"}
    warns = {s: msg for s, (sev, msg) in issues.items() if sev == "warn"}
    if fatal:
        print("\n[provenance] FATAL — dataset/benchmark differs across seeds:")
        for slug, msg in sorted(fatal.items()):
            print(f"  - {slug}: {msg}")
    if warns:
        print("\n[provenance] warning — code-revision drift only (path unchanged):")
        for slug, msg in sorted(warns.items()):
            print(f"  - {slug}: {msg}")
    if fatal and args.strict_provenance:
        raise SystemExit(
            "\n[abort] --strict-provenance set and dataset/benchmark provenance "
            "differs across seeds. Re-run the affected configs on one dataset/"
            "benchmark definition before building the paper table.")
    if warns and args.strict_provenance:
        print("  (code-rev drift is a warning even under --strict-provenance; "
              "only dataset/benchmark mismatch is fatal)\n")
    elif issues:
        print("  (warning only; pass --strict-provenance to make dataset/"
              "benchmark mismatch fatal)\n")

    with open(args.out_md, "w", encoding="utf-8", newline="\n") as f:
        f.write(render_markdown(groups))
    with open(os.path.join(args.results_dir, "summary.json"), "w", encoding="utf-8") as f:
        f.write(render_summary(groups))

    n_runs = sum(len(g.seeds) for g in groups.values())
    print(f"aggregated {len(groups)} config(s) ({n_runs} runs) -> "
          f"{args.out_md} + {args.results_dir}/summary.json")


if __name__ == "__main__":
    main()
