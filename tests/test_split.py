"""Validate the leakage-safe deployment-grouped split on the REAL OzFish CSV.

Images are not on this machine (they live on the training VM), so we monkeypatch
os.path.exists to accept every row — this exercises the grouping/stratification
logic on the true metadata distribution without needing the frames.

Checks:
  1. No deployment appears in more than one fold (the leakage guarantee).
  2. Stereo pairs (A000001_L / A000001_R) land in the same fold.
  3. Ratios are close to the 70/15/15 target.
  4. The split is reproducible (same seed -> identical assignment).
  5. Different seeds give different assignments.
"""

import os
import sys
from collections import defaultdict

import bioreef.data.split as split_mod
from bioreef.data.split import split_dataset, deployment_id

# Real metadata from the pipeline repo (sibling dir).
CSV = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "Junior-Project", "frame_metadata.csv"
))


def _run_split(seed=0, monkeypatch_exists=True):
    real_exists = os.path.exists
    if monkeypatch_exists:
        # Accept the CSV path itself; pretend every image file exists.
        split_mod.os.path.exists = lambda p: True if p.endswith(".png") else real_exists(p)
    try:
        return split_dataset(CSV, img_dir="", min_samples=20, seed=seed)
    finally:
        split_mod.os.path.exists = real_exists


def _fold_of_deployment(samples_by_fold):
    dep_fold = {}
    for fold, samples in samples_by_fold.items():
        for s in samples:
            dep_fold[s["deployment"]] = fold
    return dep_fold


def test_real_split_is_leakage_safe():
    if not os.path.exists(CSV):
        print(f"SKIP: CSV not found at {CSV}")
        return

    train, val, test, num_classes, c2s, spc = _run_split(seed=0)
    folds = {"train": train, "val": val, "test": test}
    total = len(train) + len(val) + len(test)
    print(f"\nclasses={num_classes}  crops={total}")
    for f in folds:
        print(f"  {f:5s}: {len(folds[f]):6d} ({100*len(folds[f])/total:.1f}%)")

    # 1. Deployment disjointness.
    deps = {f: set(s["deployment"] for s in folds[f]) for f in folds}
    assert not (deps["train"] & deps["val"]), "train/val deployment leak!"
    assert not (deps["train"] & deps["test"]), "train/test deployment leak!"
    assert not (deps["val"] & deps["test"]), "val/test deployment leak!"
    print("  [1] deployment disjointness: OK")

    # 2. Stereo pairs share a fold (both cams have the same deployment id, so
    #    grouping by deployment guarantees it — assert the invariant holds).
    dep_fold = _fold_of_deployment(folds)
    # every sample's deployment maps to exactly the fold it's in
    for f in folds:
        for s in folds[f]:
            assert dep_fold[s["deployment"]] == f
    print("  [2] stereo L/R kept together (deployment grouping): OK")

    # 3. Ratios within tolerance.
    assert 0.62 <= len(train) / total <= 0.78, "train ratio off"
    assert 0.08 <= len(val) / total <= 0.22, "val ratio off"
    assert 0.08 <= len(test) / total <= 0.22, "test ratio off"
    print("  [3] ratios near 70/15/15: OK")

    # 4. Species stratification — with the >=3-deployment inclusion rule every
    #    benchmark species CAN be split, so test coverage should be ~complete.
    test_species = set(s["species"] for s in test)
    coverage = len(test_species) / num_classes
    print(f"  [4] benchmark species: {num_classes} | test covers "
          f"{len(test_species)} ({coverage:.1%})")
    # >=3 deployments guarantees a species CAN be split, not that it always
    # reaches all three folds (a 3-deployment species may land 2 train + 1 val).
    assert num_classes == 321, f"expected 321 benchmark species, got {num_classes}"
    assert coverage >= 0.95, "test species coverage too low for a >=3-dep benchmark"
    print("test_real_split_is_leakage_safe OK")


def test_split_reproducible():
    if not os.path.exists(CSV):
        print("SKIP: CSV not found")
        return
    a = _run_split(seed=0)[0]
    b = _run_split(seed=0)[0]
    assert [s["img_path"] for s in a] == [s["img_path"] for s in b]
    print("test_split_reproducible OK (same seed -> identical train set)")

    c = _run_split(seed=1)[0]
    # different seed should change at least some deployment assignments
    deps_a = set(s["deployment"] for s in a)
    deps_c = set(s["deployment"] for s in c)
    assert deps_a != deps_c, "seed had no effect on the split"
    print("test_split seed-sensitivity OK (seed 0 != seed 1)")


def test_split_hashseed_invariant():
    """The split must be identical across processes with different PYTHONHASHSEED.
    Regression for the bug where sets of deployment strings were iterated directly
    (their order is hash-seed dependent), so every run produced a slightly
    different split. Runs the split in fresh subprocesses under distinct seeds and
    asserts the fold assignment is byte-identical."""
    import subprocess
    if not os.path.exists(CSV):
        print("SKIP: CSV not found")
        return

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    prog = (
        "import bioreef.data.split as m, os, hashlib\n"
        "r=os.path.exists\n"
        "m.os.path.exists=lambda p: True if str(p).endswith('.png') else r(p)\n"
        "tr,va,te,nc,c2s,spc=m.split_dataset(%r, img_dir='', min_samples=20, min_deployments=3, seed=0)\n"
        "sig=hashlib.md5(str(sorted(s['img_path'] for s in tr)).encode()).hexdigest()\n"
        "print(f'{nc}|{sig}')\n" % CSV
    )
    sigs = []
    for hs in ("0", "1", "2", "7"):
        env = dict(os.environ, PYTHONHASHSEED=hs, PYTHONPATH=repo)
        out = subprocess.check_output([sys.executable, "-c", prog], env=env, text=True)
        sigs.append(out.strip().splitlines()[-1])
    assert len(set(sigs)) == 1, f"split varies with PYTHONHASHSEED: {set(sigs)}"
    print(f"test_split_hashseed_invariant OK (identical across 4 hash seeds: {sigs[0][:20]}...)")


if __name__ == "__main__":
    test_real_split_is_leakage_safe()
    test_split_reproducible()
    test_split_hashseed_invariant()
    print("\nALL SPLIT TESTS PASSED")
