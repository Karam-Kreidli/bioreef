"""Regression tests for the augmentation-order bug (the one that trained on
background): augmentation must be applied AFTER cropping, must keep the fish in
the ROI, must transform all context streams identically, and must be a no-op for
val/test. These guard a class of bug that runs without error but is silently
wrong, so it cannot be caught by a smoke run alone.
"""

import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bioreef.data.context import ContextHarvester
from bioreef.data.augmentation import MarineAugmentor


def _frame_with_fish(bbox, fill=0.42):
    """A frame with a solid ellipse 'fish' filling ~84% of the bbox."""
    frame = np.zeros((1080, 1920, 3), np.uint8)
    x, y, w, h = bbox
    cv2.ellipse(frame, (x + w // 2, y + h // 2),
                (int(w * fill), int(h * fill)), 0, 0, 360, (0, 0, 255), -1)
    return frame


def _red(crop):
    return int(((crop[:, :, 2] > 150) & (crop[:, :, 1] < 80) & (crop[:, :, 0] < 80)).sum())


def test_fish_retained_through_rotation():
    """Full 0-360 rotation must NOT clip the fish out of the ROI (letterboxing
    centres it with margin). Regression for augment-before-crop, which dropped the
    fish entirely on flips/rotations."""
    bbox = (800, 400, 200, 140)
    harv = ContextHarvester(target_resolution=224, small_object_threshold=0.0)
    frame = _frame_with_fish(bbox)
    base = _red(harv.harvest_uint8(frame, bbox)["roi"])
    assert base > 0

    aug = MarineAugmentor(horizontal_flip_prob=0, vertical_flip_prob=0,
                          rotation_limit=360, noise_var_limit=(0, 0),
                          marine_snow_prob=0, motion_blur_prob=0, brightness_limit=0,
                          contrast_limit=0, saturation_limit=0, enabled=True)
    np.random.seed(0)
    retentions = []
    for _ in range(100):
        roi = aug.transform_streams(harv.harvest_uint8(frame, bbox))["roi"]
        retentions.append(_red(roi) / base)
    r = np.array(retentions)
    assert r.mean() > 0.9, f"fish lost on rotation: mean retention {r.mean():.2%}"
    assert (r < 0.7).mean() == 0.0, "some rotations drop >30% of the fish"
    print(f"test_fish_retained_through_rotation OK (mean {r.mean():.1%}, min {r.min():.1%})")


def test_geometric_transform_shared_across_streams():
    """The geometric transform (flip) must be identical across all context
    streams so they stay spatially coherent for MCEAM."""
    frame = np.zeros((1080, 1920, 3), np.uint8)
    bx, by, bw, bh = 900, 500, 200, 120
    frame[by:by + bh, bx:bx + bw // 2] = (0, 0, 255)   # red LEFT half of the bbox
    harv = ContextHarvester(target_resolution=224, small_object_threshold=0.0)
    aug = MarineAugmentor(horizontal_flip_prob=1.0, vertical_flip_prob=0.0,
                          rotation_limit=0, noise_var_limit=(0, 0), marine_snow_prob=0,
                          motion_blur_prob=0, brightness_limit=0, contrast_limit=0,
                          saturation_limit=0, enabled=True)
    crops = aug.transform_streams(harv.harvest_uint8(frame, (bx, by, bw, bh)))

    def red_side(c):
        W = c.shape[1]
        left = (c[:, :W // 2, 2] > 150).sum()
        right = (c[:, W // 2:, 2] > 150).sum()
        return "LEFT" if left > right else "RIGHT"

    sides = {name: red_side(c) for name, c in crops.items()}
    assert len(set(sides.values())) == 1, f"streams flipped inconsistently: {sides}"
    print(f"test_geometric_transform_shared_across_streams OK ({sides})")


def test_val_test_no_augmentation():
    """is_train=False must apply NO augmentation (deterministic); is_train=True
    must actually augment (varies run to run)."""
    import bioreef.data.dataset as d
    orig = d.safe_imread
    d.safe_imread = lambda p: (np.random.RandomState(1).rand(300, 300, 3) * 255).astype(np.uint8)
    try:
        from bioreef.data import FishCropDataset
        s = [{"img_path": "x.png", "bbox": [50, 50, 120, 90], "class_idx": 0, "species": "sp"}]
        val = FishCropDataset(s, is_train=False)
        assert torch.equal(val[0]["streams"]["roi"], val[0]["streams"]["roi"]), \
            "val/test must be deterministic (no augmentation)"
        train = FishCropDataset(s, is_train=True)
        assert not torch.equal(train[0]["streams"]["roi"], train[0]["streams"]["roi"]), \
            "train must be augmented (should vary)"
        print("test_val_test_no_augmentation OK")
    finally:
        d.safe_imread = orig


if __name__ == "__main__":
    test_fish_retained_through_rotation()
    test_geometric_transform_shared_across_streams()
    test_val_test_no_augmentation()
    print("\nALL AUGMENTATION TESTS PASSED")
