"""Classification dataset: per-sample read -> augment -> 4-stream context crops.

is_train controls augmentation: True for the train split, False for val/test
(MarineAugmentor(enabled=False) returns raw crops). The paper uses raw crops with
no enhancement/restoration step (Section 5.1)."""

from torch.utils.data import Dataset

from bioreef.data.context import ContextHarvester
from bioreef.data.augmentation import MarineAugmentor
from bioreef.training.ddp import safe_imread


class FishCropDataset(Dataset):
    def __init__(self, samples, is_train=True):
        self.samples = samples
        self.harvester = ContextHarvester(target_resolution=224, small_object_threshold=0.05)
        self.augmentor = MarineAugmentor(enabled=is_train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        frame = safe_imread(s["img_path"])
        if frame is None:
            # A decode failure used to become a uniform gray 1080x1920 frame. That
            # silently injects a meaningless example — harmless-looking in training,
            # but in val/test the model is SCORED on a blank image. For a benchmark
            # a corrupt file must be loud, not quietly synthesized.
            raise RuntimeError(
                f"failed to decode image: {s['img_path']} (species={s['species']}). "
                "The file exists but OpenCV could not read it — it is likely "
                "truncated or corrupt. Repair or remove the file (and re-export the "
                "split) rather than training on a synthetic blank frame."
            )
        # Correct order: crop the CLEAN frame with the bbox (fish centred), THEN
        # augment the crops. Augmenting the frame first would move the fish out of
        # its (unchanged) bbox on flips/rotations and crop the wrong region.
        crops = self.harvester.harvest_uint8(frame, s["bbox"])
        crops = self.augmentor.transform_streams(crops)   # no-op when is_train=False
        streams = self.harvester.normalize_streams(crops)
        return {"streams": streams, "label": s["class_idx"], "species": s["species"]}
