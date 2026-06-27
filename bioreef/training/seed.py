"""Unified RNG seeding for reproducible runs (paper item C).

Call set_seed(args.seed) once at the top of main() in every entry point. This
covers Python's random, NumPy's global RNG (augmentation draws from it), and
PyTorch CPU+CUDA init (MCEAM + head weight init). The balanced sampler takes the
same seed explicitly. Without this, the >=3-seed mean +/- std does not capture
initialization or augmentation variance.
"""

import logging
import os
import random

logger = logging.getLogger("bioreef.training.seed")


def set_seed(seed: int, deterministic: bool = False):
    """Seed all RNGs. deterministic=True forces cudnn determinism (slower, but
    bit-reproducible); leave False for the speed/variance trade-off the paper
    uses (variance is captured by running multiple seeds)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    import numpy as np
    np.random.seed(seed)

    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    logger.info("Seed set to %d (deterministic=%s).", seed, deterministic)
