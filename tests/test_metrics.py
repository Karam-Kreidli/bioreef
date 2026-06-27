"""Unit tests for the metric suite — hand-computed expected values.

Run: python -m pytest tests/test_metrics.py -v   (or: python tests/test_metrics.py)

A metric bug found after the multi-seed runs is the expensive mistake, so these
pin every number by hand on a tiny toy taxonomy.
"""

import numpy as np

from bioreef.eval import (
    hierarchical_distance,
    macro_accuracy,
    micro_accuracy,
    per_class_accuracy,
    group_accuracy,
    topk_accuracy,
    hierarchical_metrics,
    freq_groups,
    evaluate_classification,
)

# Toy taxonomy: 4 species across 2 genera, 2 families.
#   FamA: GenA{sp0, sp1}          FamB: GenB{sp2}, GenC{sp3}
IDX_TO_SP = {0: "sp0", 1: "sp1", 2: "sp2", 3: "sp3"}
TREE = {
    "sp0": {"species": "sp0", "genus": "GenA", "family": "FamA"},
    "sp1": {"species": "sp1", "genus": "GenA", "family": "FamA"},
    "sp2": {"species": "sp2", "genus": "GenB", "family": "FamB"},
    "sp3": {"species": "sp3", "genus": "GenC", "family": "FamB"},
}


def approx(a, b, tol=1e-9):
    return abs(a - b) < tol


def test_hd_distance_rule():
    assert hierarchical_distance("sp0", "sp0", TREE) == (0.0, "species")
    assert hierarchical_distance("sp0", "sp1", TREE) == (1.0, "genus")    # same genus
    assert hierarchical_distance("sp2", "sp3", TREE) == (2.0, "family")   # same family, diff genus
    assert hierarchical_distance("sp0", "sp2", TREE) == (3.0, "root")     # diff family
    assert hierarchical_distance("sp0", "ZZZ", TREE) == (3.0, "unknown")  # missing taxon
    print("test_hd_distance_rule OK")


def test_micro_and_macro_accuracy():
    # targets:  [0,0,1,2,2,2]  preds: [0,1,1,2,2,0]
    # per-class: cls0: 1/2=0.5, cls1: 1/1=1.0, cls2: 2/3=0.667, cls3: None
    targets = np.array([0, 0, 1, 2, 2, 2])
    preds = np.array([0, 1, 1, 2, 2, 0])
    pc = per_class_accuracy(preds, targets, 4)
    assert approx(pc[0], 0.5) and approx(pc[1], 1.0) and approx(pc[2], 2 / 3)
    assert pc[3] is None
    # micro = 4 correct / 6
    assert approx(micro_accuracy(preds, targets), 4 / 6)
    # macro = mean of present classes (3) = (0.5 + 1.0 + 0.667)/3
    assert approx(macro_accuracy(preds, targets, 4), (0.5 + 1.0 + 2 / 3) / 3)
    print("test_micro_and_macro_accuracy OK")


def test_freq_groups_and_group_accuracy():
    # counts: cls0=150(head) cls1=50(med) cls2=10(tail) cls3=200(head)
    spc = [150, 50, 10, 200]
    g = freq_groups(spc, head_thresh=100, tail_thresh=20)
    assert g["head"] == [0, 3] and g["medium"] == [1] and g["tail"] == [2]
    # accuracy within groups
    targets = np.array([0, 0, 1, 2, 2, 2, 3])
    preds = np.array([0, 1, 1, 2, 2, 0, 3])
    ga = group_accuracy(preds, targets, 4, g)
    # head = mean(cls0=0.5, cls3=1.0)=0.75 ; medium=cls1=1.0 ; tail=cls2=2/3
    assert approx(ga["head"], 0.75) and approx(ga["medium"], 1.0) and approx(ga["tail"], 2 / 3)
    print("test_freq_groups_and_group_accuracy OK")


def test_topk():
    # 2 samples, 4 classes. scores favor these argsorts:
    scores = np.array([
        [0.1, 0.2, 0.6, 0.1],   # ranked: sp2 > sp1 > {sp0,sp3}
        [0.7, 0.1, 0.1, 0.1],   # ranked: sp0 top
    ])
    targets = np.array([1, 0])
    assert approx(topk_accuracy(scores, targets, k=1), 0.5)   # only sample2 top-1 correct
    assert approx(topk_accuracy(scores, targets, k=2), 1.0)   # sample1 sp1 in top2
    print("test_topk OK")


def test_hierarchical_metrics():
    # targets [0,1,2,3], preds [0,0,3,3]
    #   0->0 species (hd0) | 1->0 genus (hd1) | 2->3 family (hd2) | 3->3 species (hd0)
    targets = np.array([0, 1, 2, 3])
    preds = np.array([0, 0, 3, 3])
    m = hierarchical_metrics(preds, targets, IDX_TO_SP, TREE)
    assert approx(m["mean_hd"], (0 + 1 + 2 + 0) / 4)            # 0.75
    # errors are the 2 wrong (hd1, hd2) -> severity = 1.5
    assert approx(m["mistake_severity"], (1 + 2) / 2)
    assert approx(m["species_accuracy"], 2 / 4)                 # 2 exact
    assert approx(m["genus_accuracy"], (2 + 1) / 4)            # +1 genus-correct
    assert approx(m["family_accuracy"], (2 + 1 + 1) / 4)       # +1 family-correct
    assert approx(m["cross_family_error_rate"], 0.0)
    print("test_hierarchical_metrics OK")


def test_evaluate_classification_panel():
    targets = np.array([0, 1, 2, 3])
    preds = np.array([0, 0, 3, 3])
    scores = np.eye(4)[preds]  # one-hot at the prediction
    spc = [150, 50, 10, 200]
    r = evaluate_classification(preds, targets, scores, 4, IDX_TO_SP, TREE, spc)
    assert "macro_accuracy" in r and "mean_hd" in r and "group_accuracy" in r
    assert approx(r["mean_hd"], 0.75)
    assert approx(r["top1_accuracy"], 0.5)
    assert r["group_sizes"] == {"head": 2, "medium": 1, "tail": 1}
    print("test_evaluate_classification_panel OK")


if __name__ == "__main__":
    test_hd_distance_rule()
    test_micro_and_macro_accuracy()
    test_freq_groups_and_group_accuracy()
    test_topk()
    test_hierarchical_metrics()
    test_evaluate_classification_panel()
    print("\nALL METRIC TESTS PASSED")
