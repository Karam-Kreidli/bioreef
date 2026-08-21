"""ContextHarvester — multi-scale concentric cropping for MCEAM."""

import logging
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger("bioreef.data.context")


class ContextHarvester:
    """
    4-stream concentric crops for MCEAM (all letterboxed + ImageNet-normalized):
        roi (1x) morphology . social (3x) neighbours . habitat (5x) substrate .
        full_frame macro-environment.
    Size-adaptive: a fish below small_object_threshold gets an extra
    letterbox-to-highres_initial step before the final resize to target_res.
    NOTE: since the source crop already exists at its native resolution, this
    intermediate up-then-down resize does NOT recover lost detail — it only changes
    the interpolation path (and adds a mild blur). It is retained for exact
    reproducibility of the reported runs; whether it helps is an open ablation, so
    do not describe it as "preserving texture" in the paper without that evidence.
    """

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        crop_scales: List[int] = (1, 3, 5),
        target_resolution: int = 224,
        small_object_threshold: float = 0.05,
        highres_initial: int = 512,
        include_full_frame: bool = True,
    ):
        self.crop_scales = crop_scales
        self.target_res = target_resolution
        self.small_thresh = small_object_threshold
        self.highres_initial = highres_initial
        self.include_full_frame = include_full_frame

    def _extract_crop(self, frame, cx, cy, crop_w, crop_h):
        """Crop centered at (cx, cy), zero-padded at frame boundaries."""
        h, w = frame.shape[:2]
        x1 = cx - crop_w // 2
        y1 = cy - crop_h // 2
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        # Clamp BOTH ends into [0, w] / [0, h]. Clamping only one end (max(0,x1),
        # min(w,x2)) breaks for a box entirely off one side: e.g. x1=x2=-50 gives
        # src_x1=0, src_x2=-50 -> a reversed/negative-width slice. Clamping both
        # ends yields an empty (not reversed) intersection, which copies nothing.
        src_x1, src_x2 = min(max(x1, 0), w), min(max(x2, 0), w)
        src_y1, src_y2 = min(max(y1, 0), h), min(max(y2, 0), h)

        crop = np.zeros((crop_h, crop_w, 3), dtype=frame.dtype)
        # No overlap with the frame -> return the zero-padded canvas as-is.
        if src_x2 <= src_x1 or src_y2 <= src_y1:
            return crop
        dst_x1 = src_x1 - x1
        dst_y1 = src_y1 - y1
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        crop[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]
        return crop

    def _letterbox_resize(self, image, target):
        """Aspect-preserving resize (zero-pad then bicubic to target square).
        Naive square resize would distort elongated species (e.g. barracuda)."""
        h, w = image.shape[:2]
        scale = target / max(h, w)
        # max(1, ...): an extreme aspect ratio can truncate the short side to 0
        # (e.g. a 3x2000 crop scaled down), and cv2.resize raises on a zero dim.
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        canvas = np.zeros((target, target, 3), dtype=image.dtype)
        pad_y = (target - new_h) // 2
        pad_x = (target - new_w) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return canvas

    def _normalize(self, image):
        """BGR uint8 crop -> RGB float tensor + ImageNet Z-score.

        The BGR->RGB conversion is REQUIRED, not cosmetic: cv2.imread returns BGR,
        but DINO/timm ImageNet-pretrained weights were trained on RGB, and
        IMAGENET_MEAN/STD are per-channel constants in RGB order. Normalizing BGR
        with RGB statistics feeds every pretrained backbone channel-swapped input
        (it also mismatches the R and B mean/std). Comparisons stay internally
        consistent either way, but absolute accuracy suffers — so convert here,
        the single gateway from uint8 crops to model tensors."""
        img = image[:, :, ::-1]                       # BGR -> RGB
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img = (img - self.IMAGENET_MEAN) / self.IMAGENET_STD
        return torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)

    def harvest_uint8(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Dict[str, np.ndarray]:
        """4-stream harvest -> dict of letterboxed uint8 BGR crops (res,res,3),
        BEFORE normalization. Cropping uses the bbox on the CLEAN frame, so the
        fish is correctly centred; augmentation is applied to these crops
        afterwards (never to the frame before cropping — that would move the fish
        out of the bbox on flips/rotations)."""
        x, y, w, h = bbox
        cx, cy = x + w // 2, y + h // 2
        frame_area = frame.shape[0] * frame.shape[1]
        fish_area = w * h

        crops = {}
        for scale in self.crop_scales:
            crop_w, crop_h = int(w * scale), int(h * scale)
            raw_crop = self._extract_crop(frame, cx, cy, crop_w, crop_h)

            # Size-adaptive ROI: high-res initial crop for small objects.
            if scale == 1 and (fish_area / frame_area) < self.small_thresh:
                raw_crop = self._letterbox_resize(raw_crop, self.highres_initial)

            resized = self._letterbox_resize(raw_crop, self.target_res)
            scale_name = {1: "roi", 3: "social", 5: "habitat"}.get(scale, f"context_{scale}x")
            crops[scale_name] = resized

        if self.include_full_frame:
            crops["full_frame"] = self._letterbox_resize(frame, self.target_res)

        return crops

    def normalize_streams(self, crops: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """uint8 BGR crops -> normalized (3,res,res) tensors (ImageNet Z-score)."""
        return {name: self._normalize(img) for name, img in crops.items()}

    def harvest(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Dict[str, torch.Tensor]:
        """Crop + normalize with NO augmentation (val/test path, and feature
        caching). For training use harvest_uint8 -> augment -> normalize_streams."""
        return self.normalize_streams(self.harvest_uint8(frame, bbox))
