"""Fixed protocol hyperparameters — the panel-wide constants that are NOT part of
any per-run config but still fully determine a result. Centralised here (instead of
scattered as magic numbers) so they can be (a) referenced from one place and (b)
recorded in each run's provenance, so the saved result completely describes the
experiment (audit #31/#32).

Changing any value here changes every run's behaviour, so a value change should be
deliberate and is captured by the provenance fingerprint (a run recorded under the
old constants will show a differing fingerprint from one under the new ones).
"""

# Optimiser / EMA (bioreef/training/loop.py).
WEIGHT_DECAY = 0.01              # AdamW weight decay (all runs)
EMA_DECAY = 0.999               # exponential moving average of trainable params

# Long-tail loss (bioreef/training/losses.py, model/hslm_loss.py).
CBFOCAL_BETA = 0.9999           # class-balanced re-weighting beta
FOCAL_GAMMA = 2.0               # focal-loss focusing parameter

# Context harvesting / crop preprocessing (bioreef/data/context.py, dataset.py).
TARGET_RESOLUTION = 224         # square crop side fed to the backbone
SMALL_OBJECT_THRESHOLD = 0.05   # fish below this frame-area fraction get the
                                # high-res intermediate crop
HIGHRES_INITIAL = 512           # intermediate crop size for small objects
CONTEXT_SCALES = (3, 5)         # social (3x) / habitat (5x) context crop scales
                                # (+ full_frame); mirrors MCEAM context streams

# Augmentation (bioreef/data/augmentation.py) — the light marine augmentor's
# probabilities/limits. Kept here for the record; the augmentor's own defaults are
# the source of truth for behaviour.
AUGMENTATION = {
    "horizontal_flip_prob": 0.5,
    "vertical_flip_prob": 0.0,
    "rotation_limit_deg": 30,
    "noise_prob": 1.0,
    "noise_var_limit": (5.0, 15.0),
    "marine_snow_prob": 0.1,
    "motion_blur_prob": 0.1,
    "brightness_limit": 0.1,
    "contrast_limit": 0.1,
    "saturation_limit": 0.1,
}


def as_dict() -> dict:
    """All fixed protocol constants as a JSON-serialisable dict, for provenance."""
    return {
        "weight_decay": WEIGHT_DECAY,
        "ema_decay": EMA_DECAY,
        "cbfocal_beta": CBFOCAL_BETA,
        "focal_gamma": FOCAL_GAMMA,
        "target_resolution": TARGET_RESOLUTION,
        "small_object_threshold": SMALL_OBJECT_THRESHOLD,
        "highres_initial": HIGHRES_INITIAL,
        "context_scales": list(CONTEXT_SCALES),
        "augmentation": AUGMENTATION,
    }
