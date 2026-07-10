"""
HSLM Loss — Hierarchical Separation-Induced Learning (marginalization variant).

The species head emits S-way logits. HSLM keeps that head and adds two auxiliary
objectives by *marginalizing* the species distribution up the taxonomy tree:

    p_genus[b, g]  = sum_{s : genus(s)=g}  p_species[b, s]
    p_family[b, f] = sum_{s : family(s)=f} p_species[b, s]

    L = w_species . L_species  +  w_genus . L_genus  +  w_family . L_family

  - L_species : CB-Focal loss (Cui et al. 2019) — long-tail class balancing
                where the imbalance lives (species level). HSLM with
                w_genus=w_family=0 is a strict CB-Focal equivalent.
  - L_genus / L_family : NLL on the marginalized distributions. A within-genus
                confusion barely moves L_genus; a cross-family mistake drives
                L_family hard — so coarse errors are penalised more.

Weights default family=3 / genus=2 / species=1: trades a little raw species
Top-1 for better Hierarchical Distance (the primary classifier metric). Exposed
as constructor args so they are tunable / ablatable (paper 4.3, ablation A5).

NOTE: this differs from MATANet's parallel per-level classifier heads — it is a
marginalization-based mechanism, not MATANet's HSLM.

Reference: Cui et al. (2019), "Class-Balanced Loss Based on Effective Number of
Samples."
"""

import logging
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("bioreef.model.hslm")


class HSLMLoss(nn.Module):
    """Hierarchical loss over species / genus / family (see module docstring)."""

    def __init__(
        self,
        samples_per_class: Sequence[int],
        species_to_genus: Sequence[int],
        species_to_family: Sequence[int],
        num_genera: int,
        num_families: int,
        family_weight: float = 3.0,
        genus_weight: float = 2.0,
        species_weight: float = 1.0,
        beta: float = 0.9999,
        gamma: float = 2.0,
        eps: float = 1e-8,
        device: str = "cuda",
    ):
        super().__init__()

        # CB-Focal class-balanced weights for the species term.
        spc = np.asarray(samples_per_class, dtype=np.float64)
        effective_num = 1.0 - np.power(beta, spc)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
        weights = weights / np.sum(weights) * len(spc)
        self.register_buffer(
            "cb_weights", torch.tensor(weights, dtype=torch.float32, device=device)
        )

        # Taxonomy index maps (species idx -> genus / family idx).
        s2g = torch.as_tensor(species_to_genus, dtype=torch.long, device=device)
        s2f = torch.as_tensor(species_to_family, dtype=torch.long, device=device)
        assert s2g.numel() == len(spc) == s2f.numel(), (
            "species_to_genus / species_to_family must have one entry per "
            f"species ({len(spc)}); got {s2g.numel()} and {s2f.numel()}"
        )
        self.register_buffer("species_to_genus", s2g)
        self.register_buffer("species_to_family", s2f)

        self.num_genera = int(num_genera)
        self.num_families = int(num_families)
        self.gamma = gamma
        self.eps = eps
        self.w_species = species_weight
        self.w_genus = genus_weight
        self.w_family = family_weight

        # Per-component values from the most recent forward() — for logging.
        self.last_components: Dict[str, float] = {}

        logger.info(
            "HSLMLoss: %d species -> %d genera -> %d families | "
            "weights species=%.1f genus=%.1f family=%.1f",
            len(spc), self.num_genera, self.num_families,
            species_weight, genus_weight, family_weight,
        )

    def _marginalize(self, p_species, mapping, num_groups):
        """Sum species probs (B,S) into parent groups via `mapping` (S,) ->
        (B, num_groups)."""
        B = p_species.shape[0]
        p_group = p_species.new_zeros(B, num_groups)
        idx = mapping.unsqueeze(0).expand(B, -1)
        p_group.scatter_add_(1, idx, p_species)
        return p_group

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits (B,S) + targets (B,) -> scalar total loss. Per-level
        components stashed in self.last_components for logging."""
        # Species term: CB-Focal. Focal factor uses the UNWEIGHTED true-class prob
        # (pt = softmax(logits)[target]); the class-balanced weight is applied as a
        # scale afterwards (Cui et al. 2019). Folding the weight into pt would
        # distort the focal modulation for the rare classes it targets.
        ce = F.cross_entropy(logits, targets, reduction="none")   # unweighted
        pt = torch.exp(-ce)
        w = self.cb_weights[targets]
        species_loss = (w * (1.0 - pt) ** self.gamma * ce).mean()

        # Marginalize species probabilities up the taxonomy.
        p_species = F.softmax(logits, dim=1)
        p_genus = self._marginalize(p_species, self.species_to_genus, self.num_genera)
        p_family = self._marginalize(p_species, self.species_to_family, self.num_families)

        genus_targets = self.species_to_genus[targets]
        family_targets = self.species_to_family[targets]

        genus_loss = F.nll_loss(torch.log(p_genus.clamp_min(self.eps)), genus_targets)
        family_loss = F.nll_loss(torch.log(p_family.clamp_min(self.eps)), family_targets)

        total = (
            self.w_species * species_loss
            + self.w_genus * genus_loss
            + self.w_family * family_loss
        )

        self.last_components = {
            "species": float(species_loss.detach()),
            "genus": float(genus_loss.detach()),
            "family": float(family_loss.detach()),
            "total": float(total.detach()),
        }
        return total
