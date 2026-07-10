"""
Frozen-backbone feature cache for the linear-probe family (C01/C02/A2/A9).

For a frozen backbone the ROI [CLS] embedding of a crop never changes across
epochs or seeds, yet the default loop re-decodes the full 1080p frame, re-harvests
4 context streams, and re-runs DINOv3 every epoch. That per-crop cost is the
data-loading wall (measured: the run is CPU/decode bound, not GPU bound).

This computes the frozen backbone's ROI [CLS] features ONCE (un-augmented — the
standard linear-probe protocol), caches them to disk keyed by the exact backbone
+ split definition, and lets the trainer fit only the tiny trainable head (for
C01: roi_only_proj + head) on cached 768-d vectors. Turns a ~hours probe into
minutes, and the cache is reused across all seeds.

Only valid when the backbone is frozen AND context is off (no MCEAM) — the cached
[CLS] is exactly the model's frozen input to the trainable modules there.
"""

import hashlib
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bioreef.data import FishCropDataset

DEFAULT_CACHE_DIR = os.path.join("results", "feature_cache")


def _cache_key(bench, run_cfg, split_name, n_samples):
    """A cache is valid only for one (backbone, split definition, split, size).
    Any change to the inclusion rules / seed / backbone yields a new key, so a
    stale cache can never silently serve wrong features."""
    parts = [
        run_cfg.backbone,
        f"ms{bench.min_samples}", f"md{bench.min_deployments}",
        f"ph{int(bench.filter_placeholders)}", f"seed{bench.split_seed}",
        f"ratios{'-'.join(str(r) for r in bench.ratios)}",
        split_name, f"n{n_samples}",
    ]
    return hashlib.md5("_".join(parts).encode()).hexdigest()[:16]


@torch.no_grad()
def compute_or_load(backbone, samples, bench, run_cfg, split_name, device,
                    batch_size=64, num_workers=8, cache_dir=DEFAULT_CACHE_DIR,
                    log=print):
    """Return (features (N, D) float32 tensor, labels (N,) long tensor) for the
    frozen backbone's ROI [CLS] over `samples`, from disk cache if present.

    Features are un-augmented (is_train=False). `backbone` is the frozen
    ViTBackbone (already on `device`, in eval)."""
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(bench, run_cfg, split_name, len(samples))
    path = os.path.join(cache_dir, f"{key}.pt")

    if os.path.exists(path):
        blob = torch.load(path, map_location="cpu")
        log(f"[cache] {split_name}: loaded {tuple(blob['features'].shape)} from {path}")
        return blob["features"], blob["labels"]

    log(f"[cache] {split_name}: computing frozen features (once) -> {path}")
    ds = FishCropDataset(samples, is_train=False)          # NO augmentation
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)

    feats, labels = [], []
    backbone.eval()
    for batch in tqdm(dl, desc=f"cache {split_name}"):
        roi = batch["streams"]["roi"].to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            cls_token, _ = backbone._extract_features(roi)  # (B, D)
        feats.append(cls_token.float().cpu())
        labels.extend(batch["label"].tolist())

    features = torch.cat(feats, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    torch.save({"features": features, "labels": labels}, path)
    log(f"[cache] {split_name}: cached {tuple(features.shape)} "
        f"({features.numel() * 4 / 1e6:.0f} MB) -> {path}")
    return features, labels


class TensorDataset(torch.utils.data.Dataset):
    """Minimal (feature, label) dataset over cached tensors — no image I/O."""

    def __init__(self, features, labels):
        self.features, self.labels = features, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.features[i], self.labels[i]
