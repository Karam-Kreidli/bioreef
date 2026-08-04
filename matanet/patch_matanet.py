"""
Patch a cloned MATANet repo so it reads OUR OzFish data instead of the
hardcoded FathomNet paths. Idempotent: safe to re-run.

    python matanet/patch_matanet.py --repo /path/to/fathomnet-cvpr2025-ssl

Edits (surgical — paths, plus one OPT-IN memory edit; architecture untouched):
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
  4. src/model.py      : OPT-IN gradient checkpointing on the ~4 fine-tuned
                         DINOv2-large encoders, gated on config.gradient_checkpointing
                         (default absent -> off, so the published path is
                         unchanged). Recompute, not approximation: numerically
                         identical, ~30% slower, ~40% less activation memory.
                         Only needed to fit the 24GB A10G (MATANet's config OOMs
                         it even at fp16); a no-op when the flag is absent/false.

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
    # OPT-IN gradient checkpointing, inserted right after the object encoder is
    # built. Gated on config.gradient_checkpointing (getattr default False), so
    # when the flag is absent the encoders behave exactly as published. Anchored
    # to the object-encoder construction line, which is stable across the pinned
    # commit.
    "src/model.py": [
        (
            "        self.obj_vit_region_encoder  = AutoModel.from_pretrained(self.hparams.obj_vit_encoder_path)\n",
            "        self.obj_vit_region_encoder  = AutoModel.from_pretrained(self.hparams.obj_vit_encoder_path)\n"
            "\n"
            "        # PATCHED (OzFish): opt-in gradient checkpointing on the ~4 fine-tuned\n"
            "        # DINOv2-large encoders (one per context scale + the object encoder).\n"
            "        # Recompute activations in the backward pass instead of storing them:\n"
            "        # numerically identical to the published run, ~30% slower, ~40% less\n"
            "        # activation memory. Off unless config.gradient_checkpointing is true,\n"
            "        # so the default path is byte-for-byte the published one. Needed only\n"
            "        # to fit the 24GB A10G (MATANet's config OOMs it even at fp16).\n"
            "        if getattr(self.hparams, 'gradient_checkpointing', False):\n"
            "            for _enc in list(self.img_vit_region_encoders.values()) + [self.obj_vit_region_encoder]:\n"
            "                if hasattr(_enc, 'gradient_checkpointing_enable'):\n"
            "                    try:\n"
            "                        _enc.gradient_checkpointing_enable(\n"
            "                            gradient_checkpointing_kwargs={'use_reentrant': False})\n"
            "                    except TypeError:\n"
            "                        _enc.gradient_checkpointing_enable()\n",
        ),
    ],
}


def apply(path, edits):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    changed = 0
    for old, new in edits:
        # Idempotency by the FULL replacement block, not its first line. An
        # insert-after edit (new starts with old, then appends) leaves `old`
        # present AND shares its first line with the original, so the old
        # "first line in text and old not in text" test double-applied it
        # (#model.py checkpointing edit). If the whole `new` is already there,
        # this edit is done.
        if new in text:
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
