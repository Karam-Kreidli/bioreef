"""
BioReef-Classify — fine-grained reef-fish classification benchmark on OzFish.

Classification only (crops are given; no detection/tracking). Packages:
    data/      leakage-safe deployment-grouped split, context crops, taxonomy
    model/     frozen ViT backbone -> MCEAM context fusion -> head, HSLM loss
    training/  seeding, losses, balanced sampler, EMA, DDP infra
    eval/      Hierarchical Distance + the full metric suite
"""

__version__ = "0.1.0"
