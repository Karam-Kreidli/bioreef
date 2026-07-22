"""Unit tests for the aggregate script's pure pieces (stats + RunGroup folding).

Importable because aggregate.py keeps statistics/data-model separate from I/O
and formatting (single responsibility) — so we test them without touching disk.
"""

import importlib.util
import math
import os

# Load scripts/aggregate.py as a module.
_agg_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "aggregate.py")
_spec = importlib.util.spec_from_file_location("aggregate", _agg_path)
agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg)


def approx(a, b, tol=1e-9):
    return abs(a - b) < tol


def test_mean_std():
    assert agg.mean_std([]) == (None, None)                  # empty -> None (valid JSON)
    assert agg.mean_std([0.5]) == (0.5, None)                # single -> std UNDEFINED, not 0
    mu, sd = agg.mean_std([0.40, 0.42, 0.44])
    assert approx(mu, 0.42) and approx(sd, 0.02)             # sample std
    print("test_mean_std OK")


def test_fmt():
    assert agg.fmt((None, None)) == "--"
    assert agg.fmt((0.5, None)) == "0.500"                   # single seed: mean only
    assert agg.fmt((0.42, 0.02)) == "0.420±0.020"
    print("test_fmt OK")


def test_rungroup_folds_seeds():
    g = agg.RunGroup(run_id="C09", model_family="dino")
    for seed, macro in zip([0, 1, 2], [0.40, 0.42, 0.44]):
        g.add({"seed": seed, "test": {
            "macro_accuracy": macro, "mean_hd": 1.5,
            "group_accuracy": {"head": macro + 0.2, "medium": macro, "tail": macro - 0.2},
        }})
    assert g.seeds == [0, 1, 2]
    assert approx(g.metric("macro_accuracy")[0], 0.42)
    assert approx(g.metric("macro_accuracy")[1], 0.02)
    assert approx(g.group("tail")[0], 0.22)
    s = g.summary()
    assert s["n_seeds"] == 3 and s["run_id"] == "C09"
    assert approx(s["metrics"]["macro_accuracy"]["mean"], 0.42)
    print("test_rungroup_folds_seeds OK")


def test_report_order_panels_before_ablations():
    groups = {
        "A1_x": agg.RunGroup(run_id="A1", model_family="dino"),
        "C09_x": agg.RunGroup(run_id="C09", model_family="dino"),
        "C01_x": agg.RunGroup(run_id="C01", model_family="dino"),
    }
    order = agg.report_order(groups)
    ids = [groups[s].run_id for s in order]
    assert ids == ["C01", "C09", "A1"], ids   # C.. before A..
    print("test_report_order_panels_before_ablations OK")


if __name__ == "__main__":
    test_mean_std()
    test_fmt()
    test_rungroup_folds_seeds()
    test_report_order_panels_before_ablations()
    print("\nALL AGGREGATE TESTS PASSED")
