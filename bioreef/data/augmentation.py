"""MarineAugmentor — underwater-domain training augmentation.

Train-only by construction: __call__ is a no-op when enabled=False, and the
dataset passes is_train through to enabled (so val/test get raw crops).
Augmentation randomness draws from NumPy's global RNG — seed it via
bioreef.training.seed.set_seed for reproducibility.
"""

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger("bioreef.data.augmentation")


class MarineAugmentor:
    """
    Underwater-domain augmentation: geometric (flips/full rotation), turbidity
    noise, marine snow, motion blur, and photometric jitter.
    """

    def __init__(
        self,
        horizontal_flip_prob: float = 0.5,
        vertical_flip_prob: float = 0.3,
        rotation_limit: int = 360,
        noise_var_limit: Tuple[float, float] = (10.0, 50.0),
        marine_snow_prob: float = 0.3,
        marine_snow_density: float = 0.005,
        marine_snow_opacity: float = 0.4,
        motion_blur_prob: float = 0.2,
        motion_blur_limit: int = 7,
        brightness_limit: float = 0.1,
        contrast_limit: float = 0.1,
        saturation_limit: float = 0.1,
        enabled: bool = True,
    ):
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.rotation_limit = rotation_limit
        self.noise_var_limit = noise_var_limit
        self.marine_snow_prob = marine_snow_prob
        self.marine_snow_density = marine_snow_density
        self.marine_snow_opacity = marine_snow_opacity
        self.motion_blur_prob = motion_blur_prob
        self.motion_blur_limit = motion_blur_limit
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.saturation_limit = saturation_limit
        self.enabled = enabled

    def _apply_geometric(self, image):
        if np.random.random() < self.horizontal_flip_prob:
            image = np.fliplr(image).copy()
        if np.random.random() < self.vertical_flip_prob:
            image = np.flipud(image).copy()
        if self.rotation_limit > 0:
            angle = np.random.uniform(0, self.rotation_limit)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return image

    def _apply_turbidity_noise(self, image):
        var = np.random.uniform(*self.noise_var_limit)
        gaussian = np.random.normal(0, var ** 0.5, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + gaussian
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def _apply_marine_snow(self, image):
        if np.random.random() > self.marine_snow_prob:
            return image
        h, w = image.shape[:2]
        num_particles = int(h * w * self.marine_snow_density)
        overlay = image.copy().astype(np.float32)
        for _ in range(num_particles):
            px, py = np.random.randint(0, w), np.random.randint(0, h)
            radius = np.random.randint(1, 4)
            cv2.circle(overlay, (px, py), radius, (255, 255, 255), -1)
        blended = cv2.addWeighted(
            image.astype(np.float32), 1.0 - self.marine_snow_opacity,
            overlay, self.marine_snow_opacity, 0,
        )
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _apply_motion_blur(self, image):
        if np.random.random() > self.motion_blur_prob:
            return image
        ksize = int(np.random.choice(range(3, self.motion_blur_limit + 1, 2)))
        angle = np.random.uniform(0, 360)
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        kernel[ksize // 2, :] = 1.0 / ksize
        M = cv2.getRotationMatrix2D((ksize // 2, ksize // 2), angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
        kernel = kernel / kernel.sum()
        return cv2.filter2D(image, -1, kernel)

    def _apply_photometric_jitter(self, image):
        beta = np.random.uniform(-self.brightness_limit, self.brightness_limit)
        alpha = 1.0 + np.random.uniform(-self.contrast_limit, self.contrast_limit)
        result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta * 255)
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat_factor = 1.0 + np.random.uniform(-self.saturation_limit, self.saturation_limit)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _sample_geometric(self):
        """Draw ONE geometric transform (flip flags + rotation angle) to share
        across all context streams so they stay spatially coherent for MCEAM."""
        return {
            "hflip": np.random.random() < self.horizontal_flip_prob,
            "vflip": np.random.random() < self.vertical_flip_prob,
            "angle": (np.random.uniform(0, self.rotation_limit)
                      if self.rotation_limit > 0 else 0.0),
        }

    def _apply_geometric_params(self, image, p):
        """Apply a pre-sampled geometric transform (same params for every stream)."""
        if p["hflip"]:
            image = np.fliplr(image).copy()
        if p["vflip"]:
            image = np.flipud(image).copy()
        if p["angle"]:
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), p["angle"], 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return image

    def _apply_photometric_and_noise(self, image):
        """The fish-position-preserving ops (safe to apply per-stream)."""
        image = self._apply_turbidity_noise(image)
        image = self._apply_marine_snow(image)
        image = self._apply_motion_blur(image)
        image = self._apply_photometric_jitter(image)
        return image

    def transform_streams(self, crops: dict) -> dict:
        """Augment ALREADY-CROPPED context streams (the correct order — the fish
        is centred in each crop, so flips/rotations keep it in frame).

        The geometric transform (flip/rotate) is sampled ONCE and shared across
        all streams so they remain spatially aligned for MCEAM cross-attention;
        photometric + noise are drawn per stream. No-op when disabled."""
        if not self.enabled:
            return crops
        geo = self._sample_geometric()
        out = {}
        for name, img in crops.items():
            img = self._apply_geometric_params(img, geo)
            img = self._apply_photometric_and_noise(img)
            out[name] = img
        return out

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Legacy single-image path (val/test no-op). NOTE: for training, use
        transform_streams on the CROPPED streams — applying geometric aug to the
        whole frame before cropping moves the fish out of its bbox."""
        if not self.enabled:
            return image
        image = self._apply_geometric(image)
        return self._apply_photometric_and_noise(image)
