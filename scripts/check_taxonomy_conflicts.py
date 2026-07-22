"""List every species whose metadata gives conflicting genus/family.

    python scripts/check_taxonomy_conflicts.py

Reads the CSV named in configs/benchmark.yaml (data.csv_path). Prints each
binomial that appears with more than one (genus, family) so they can all be
corrected in one pass, instead of discovering them one failed export at a time.
Exit code 1 if any conflict is found, 0 if the taxonomy is clean.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from bioreef.config import BenchmarkConfig, DEFAULT_CONFIG_PATH
from bioreef.data.split import binomial, canonical_genus


def main():
    cfg = BenchmarkConfig.from_yaml(DEFAULT_CONFIG_PATH)
    if not cfg.csv_path:
        raise SystemExit("no data.csv_path in configs/benchmark.yaml")

    df = pd.read_csv(cfg.csv_path).dropna(subset=["species", "genus", "family"])
    seen, conflicts = {}, {}
    for _, r in df.iterrows():
        name = binomial(r["genus"], r["species"])
        tax = (canonical_genus(r["genus"]), str(r["family"]).strip())
        if name in seen and seen[name] != tax:
            conflicts.setdefault(name, {seen[name]}).add(tax)
        seen[name] = tax

    if not conflicts:
        print("no taxonomy conflicts — all species have a single (genus, family).")
        return 0

    print(f"{len(conflicts)} conflicting species:")
    for name, variants in sorted(conflicts.items()):
        fams = sorted(f for _, f in variants)
        print(f"  {name:32s} families: {fams}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
