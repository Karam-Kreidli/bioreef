"""
Patch a cloned MATANet repo so it reads OUR OzFish data instead of the
hardcoded FathomNet paths. Idempotent: safe to re-run.

    python matanet/patch_matanet.py --repo /path/to/fathomnet-cvpr2025-ssl

Edits (surgical, path-only — the model/architecture is untouched):
  1. src/datautils.py  : load each image from anno['image_path'] (our export
                         stores the resolved absolute path) instead of the
                         hardcoded train_data/test_data dirs -> handles our
                         filenames and multi-folder datasets.
  2. B1.BuildModel.py  : train_anno_path <- config.train_anno_path
  3. C1.TestModel.py   : test_anno_path <- config.test_anno_path,
                         model_path     <- config.trained_ckpt_path,
                         submission out <- config.submission_path,
                         drop the trailing debug lines that read a nonexistent
                         file (they crash the run).

Pinned to MATANet commit 922c2176893ef1d03de8b8701cd882b5764f9ae9 (MIT license).
"""

import argparse
import os

# (file, old, new) — each applied only if `old` is still present.
EDITS = {
    "src/datautils.py": [
        (
            "        if self.phase == 'train' or self.phase == 'valid':\n"
            "            image_path = os.path.join('./dataset/fathomnet-2025/train_data/images',str(img_id)+'.png')\n"
            "        else:\n"
            "            image_path = os.path.join('./dataset/fathomnet-2025/test_data/images',str(img_id)+'.png')",
            "        # PATCHED (OzFish): our export stores the resolved absolute path.\n"
            "        image_path = anno.get('image_path', str(img_id) + '.png')",
        ),
    ],
    "B1.BuildModel.py": [
        (
            "train_anno_path = './dataset/fathomnet-2025/dataset_train.json'",
            "train_anno_path = config.train_anno_path  # PATCHED (OzFish)",
        ),
    ],
    "C1.TestModel.py": [
        (
            "test_anno_path = './dataset/fathomnet-2025/dataset_test.json'",
            "test_anno_path = config.test_anno_path  # PATCHED (OzFish)",
        ),
        (
            "    model_path = f'~/Project/cvprcom/logs/{config.project_name}/Fold-{current_fold}/last.ckpt'",
            "    model_path = config.trained_ckpt_path  # PATCHED (OzFish)",
        ),
        (
            'voted_submission.to_csv(f"./results/submission_{config.project_name}_0526_final.csv", index=False)\n'
            'ddd = pd.read_csv(f"./results/submission_experiment51_0522_01.csv")\n'
            "sum((voted_submission['concept_name'] == ddd['concept_name']).values)",
            "voted_submission.to_csv(config.submission_path, index=False)  # PATCHED (OzFish)\n"
            "print(f'wrote predictions -> {config.submission_path}')",
        ),
    ],
}


def apply(path, edits):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    changed = 0
    for old, new in edits:
        if new.split("\n")[0] in text and old not in text:
            continue  # already patched
        if old in text:
            text = text.replace(old, new, 1)
            changed += 1
        else:
            raise SystemExit(
                f"could not find expected block in {path} — the MATANet repo may "
                f"be a different commit than the pinned 922c217. Block:\n{old[:80]}..."
            )
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return changed


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True, help="path to the cloned MATANet repo")
    args = p.parse_args()

    total = 0
    for rel, edits in EDITS.items():
        fpath = os.path.join(args.repo, rel)
        if not os.path.exists(fpath):
            raise SystemExit(f"not found: {fpath} (is --repo the MATANet root?)")
        n = apply(fpath, edits)
        total += n
        print(f"  {rel}: {'patched' if n else 'already patched'}")
    print(f"[patch] done ({total} edit(s) applied). MATANet now reads OzFish paths "
          "from its config.")


if __name__ == "__main__":
    main()
