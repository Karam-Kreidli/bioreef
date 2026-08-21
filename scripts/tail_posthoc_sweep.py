"""
Offline TAIL post-hoc sweep — compare the trained model against two CE-preserving
tail fixes on the EXACT same seed(s), using dumps from `dump_posthoc.py`.
No retraining, no backbone, no GPU — pure numpy over saved features/logits.

Two methods, both keep the plain-CE species head; they only adjust the decision:
  * tau-norm      : rescale each classifier weight row by ||w_c||^tau (Kang 2020).
                    Head-class weight norms grow with frequency; shrinking them
                    lifts the tail. Needs FEATURES + head weights (in the dump).
  * logit-adjust  : subtract tau * log(prior_c) from each logit (Menon 2020 /
                    Balanced Softmax). Bayes-consistent prior correction; this
                    IS cross-entropy with a per-class margin. Needs logits + counts.

tau = 0 reproduces the baseline exactly (a built-in check that both paths and
the metric suite agree with metrics.json).

    # one seed:
    python scripts/tail_posthoc_sweep.py --seed_dir results/A15_hslm_ce_unfrozen/seed0
    # all seeds of a run, averaged (matches the 3-seed campaign protocol):
    python scripts/tail_posthoc_sweep.py --slug A15_hslm_ce_unfrozen
    # custom grid:
    python scripts/tail_posthoc_sweep.py --slug A15_hslm_ce_unfrozen --taus 0 .25 .5 .75 1 1.25 1.5
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bioreef.eval import evaluate_classification


# --- the two CE-preserving tail adjustments ---------------------------------

def tau_norm_logits(features, W, b, tau, use_bias=True):
    """Recompute logits with tau-normalized classifier weights."""
    norms = np.linalg.norm(W, axis=1, keepdims=True)          # (C, 1)
    Wn = W / np.power(np.maximum(norms, 1e-12), tau)
    lg = features @ Wn.T
    return lg + b if use_bias else lg


def logit_adjust(logits, sp_counts, tau):
    """Post-hoc logit adjustment: subtract tau * log(prior) per class."""
    prior = np.asarray(sp_counts, dtype=np.float64)
    prior = prior / max(prior.sum(), 1e-12)
    return logits - tau * np.log(prior + 1e-12)


# --- metrics ----------------------------------------------------------------

def panel(logits, targets, num_classes, idx_to_sp, tree, sp_counts):
    preds = logits.argmax(1)
    m = evaluate_classification(preds, targets, logits, num_classes,
                                idx_to_sp, tree, sp_counts)
    ga = m["group_accuracy"]
    return {"top1": m["top1_accuracy"], "macro": m["macro_accuracy"],
            "head": ga["head"], "med": ga["medium"], "tail": ga["tail"],
            "hd": m["mean_hd"], "mist": m["mistake_severity"]}


def load_dump(seed_dir):
    d = np.load(os.path.join(seed_dir, "posthoc_dump.npz"), allow_pickle=True)
    with open(os.path.join(seed_dir, "posthoc_meta.pkl"), "rb") as fh:
        meta = pickle.load(fh)
    return d, meta


def sweep_seed(seed_dir, taus):
    d, meta = load_dump(seed_dir)
    feats, logits, targets = d["features"], d["logits"], d["targets"]
    W, b, sp = d["head_W"], d["head_b"], d["sp_counts"]
    nc, idx, tree = int(d["num_classes"]), meta["idx_to_sp"], meta["taxonomy_tree"]

    def M(lg):
        return panel(lg, targets, nc, idx, tree, sp)

    out = {"baseline": {0.0: M(logits)}, "tau_norm": {}, "logit_adj": {}}
    for t in taus:
        out["tau_norm"][t] = M(tau_norm_logits(feats, W, b, t))
        out["logit_adj"][t] = M(logit_adjust(logits, sp, t))
    return out


# --- seed aggregation + printing --------------------------------------------

KEYS = ["top1", "macro", "head", "med", "tail", "hd", "mist"]


def mean_over_seeds(per_seed):
    """per_seed: list of {method: {tau: {metric: val}}} -> mean & std."""
    agg = {}
    methods = per_seed[0].keys()
    for method in methods:
        agg[method] = {}
        for tau in per_seed[0][method]:
            stack = {k: np.array([s[method][tau][k] for s in per_seed]) for k in KEYS}
            agg[method][tau] = {k: (stack[k].mean(), stack[k].std()) for k in KEYS}
    return agg


def fmt(mu_sd, n):
    mu, sd = mu_sd
    return f"{mu:.3f}" + (f"±{sd:.3f}" if n > 1 else "")


def print_table(agg, n_seeds):
    base = agg["baseline"][0.0]
    bt = base["tail"][0]
    print("\n" + "=" * 92)
    print(f"TAIL POST-HOC SWEEP  ({n_seeds} seed{'s' if n_seeds > 1 else ''}, "
          f"same trained weights — no retraining)")
    print("=" * 92)
    hdr = f"{'method':11} {'tau':>5} | {'Top-1':>12} {'Macro':>12} " \
          f"{'Head':>7} {'Med':>7} {'Tail':>12} {'HD':>7} {'MistSev':>8} | dTail"
    print(hdr); print("-" * len(hdr))

    def row(method, tau, r):
        dtail = r["tail"][0] - bt
        star = "  *" if (r["tail"][0] > bt and r["top1"][0] >= base["top1"][0] - 1e-6) else ""
        print(f"{method:11} {tau:>5} | {fmt(r['top1'],n_seeds):>12} "
              f"{fmt(r['macro'],n_seeds):>12} {r['head'][0]:>7.3f} {r['med'][0]:>7.3f} "
              f"{fmt(r['tail'],n_seeds):>12} {r['hd'][0]:>7.3f} {r['mist'][0]:>8.3f} | "
              f"{dtail:+.3f}{star}")

    row("baseline", "—", base)
    print("-" * len(hdr))
    for method in ("tau_norm", "logit_adj"):
        for tau in sorted(agg[method]):
            row(method, f"{tau:g}", agg[method][tau])
        print("-" * len(hdr))
    print("* = tail improved without losing Top-1 vs baseline.  "
          "Judge on Tail mean AND spread; the A15 tail is high-variance.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seed_dir", help="a single results/<slug>/seed<N>/ dir")
    g.add_argument("--slug", help="average over results/<slug>/seed*/")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    args = ap.parse_args()

    if args.seed_dir:
        seed_dirs = [args.seed_dir]
    else:
        seed_dirs = sorted(glob.glob(os.path.join(args.results_dir, args.slug, "seed*")))
        seed_dirs = [d for d in seed_dirs
                     if os.path.exists(os.path.join(d, "posthoc_dump.npz"))]
    if not seed_dirs:
        raise SystemExit("no seed dirs with posthoc_dump.npz found — run "
                         "scripts/dump_posthoc.py on the checkpoint(s) first.")

    per_seed = [sweep_seed(d, args.taus) for d in seed_dirs]
    print(f"[seeds] {len(seed_dirs)}: {', '.join(seed_dirs)}")
    agg = mean_over_seeds(per_seed)
    print_table(agg, len(seed_dirs))


if __name__ == "__main__":
    main()
