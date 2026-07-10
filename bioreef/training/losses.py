"""Training losses.

CBFocalLoss is the flat-head species loss (ablations A7/A8 swap it for plain CE);
the hierarchical HSLMLoss (bioreef.model) generalizes it with genus/family terms.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CBFocalLoss(nn.Module):
    """Cui et al. (2019) — effective-number class weighting + focal modulation."""

    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0, device="cuda"):
        super().__init__()
        samples_per_class = np.array(samples_per_class, dtype=np.float64)
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / effective_num
        weights = weights / np.sum(weights) * len(samples_per_class)
        self.register_buffer(
            "weights", torch.tensor(weights, dtype=torch.float32, device=device)
        )
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Focal modulation must use the UNWEIGHTED prob of the true class: pt is
        # softmax(inputs)[target], so (1-pt)^gamma reflects true confidence. The
        # class-balanced weight is then applied as a scale (Cui et al. 2019).
        # Computing pt from weighted CE would fold the weight into pt and distort
        # the focal factor for exactly the rare classes it targets.
        ce = F.cross_entropy(inputs, targets, reduction="none")   # unweighted
        pt = torch.exp(-ce)                                        # softmax prob of true class
        w = self.weights[targets]                                 # per-sample class weight
        focal_loss = w * (1 - pt) ** self.gamma * ce
        return focal_loss.mean()
