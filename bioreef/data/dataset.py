"""Classification dataset: per-sample read -> (optional restore) -> augment ->
4-stream context crops.

is_train controls augmentation: True for the train split, False for val/test
(MarineAugmentor(enabled=False) returns the raw frame). Restoration is off by
default (paper uses raw crops, Section 5.1)."""

import numpy as np
from torch.utils.data import Dataset

from bioreef.data.context import ContextHarvester
from bioreef.data.restoration import WaterNetRestorer
from bioreef.data.augmentation import MarineAugmentor
from bioreef.training.ddp import safe_imread


class FishCropDataset(Dataset):
    def __init__(self, samples, is_train=True, use_waternet=False):
        self.samples = samples
        self.harvester = ContextHarvester(target_resolution=224, small_object_threshold=0.05)
        self.restorer = WaterNetRestorer() if use_waternet else None
        self.augmentor = MarineAugmentor(enabled=is_train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        frame = safe_imread(s["img_path"])
        if frame is None:
            frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 128
        if self.restorer is not None:
            frame = self.restorer(frame)
        augmented = self.augmentor(frame)
        streams = self.harvester.harvest(augmented, s["bbox"])
        return {"streams": streams, "label": s["class_idx"], "species": s["species"]}
