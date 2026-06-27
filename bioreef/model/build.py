"""Assemble the classifier (backbone -> MCEAM -> head) from flat config so the
trainer and evaluator build identical models and every ablation is a flag.

The reference model (paper C09): DINOv3 frozen, MCEAM 1 block / 3 scales, linear
head. Ablations flip one field:
    A1  backbone="dinov2"
    A2  context_levels=0                 (no MCEAM — head on pooled ROI)
    A3  context_levels=1                 (ROI-scale only)
    A4  attention_depth=2                (A4b: 4)
"""

from dataclasses import dataclass

import torch.nn as nn

from .backbone import ViTBackbone, BACKBONES
from .mceam import MCEAM


@dataclass
class ModelConfig:
    backbone: str = "dinov3"          # dinov3 | dinov2
    context_levels: int = 3           # 0=no context, 1=ROI-scale, 3=ROI/social/habitat
    attention_depth: int = 1          # cross-attention blocks per stream
    embed_dim: int = 768
    output_dim: int = 256
    num_heads: int = 8
    unfreeze_blocks: int = 0          # 0 = fully frozen backbone


class Classifier(nn.Module):
    """Frozen backbone (no grad) + optional MCEAM + linear head. The backbone is
    held but never returned as a trainable param; only MCEAM+head (and optional
    last-N blocks) train.

    When context_levels==0 there is no MCEAM: the ROI [CLS] is projected straight
    to output_dim, giving the 'context off' ablation a clean path.
    """

    def __init__(self, cfg: ModelConfig, num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.backbone = ViTBackbone(
            pretrained_model_name=BACKBONES[cfg.backbone], freeze=True
        )
        if cfg.unfreeze_blocks > 0:
            self.backbone.unfreeze_blocks(cfg.unfreeze_blocks)

        if cfg.context_levels > 0:
            self.mceam = MCEAM(
                embed_dim=cfg.embed_dim, num_heads=cfg.num_heads,
                output_dim=cfg.output_dim, num_context_levels=cfg.context_levels,
                attention_depth=cfg.attention_depth, use_checkpointing=True,
            )
        else:
            # No context: project the ROI [CLS] directly (linear-probe-like, but
            # keeps HSLM + long-tail handling — this is ablation A2, not C01).
            self.mceam = None
            self.roi_only_proj = nn.Sequential(
                nn.LayerNorm(cfg.embed_dim),
                nn.Linear(cfg.embed_dim, cfg.output_dim),
                nn.GELU(),
            )

        self.head = nn.Linear(cfg.output_dim, num_classes)

    def trainable_modules(self):
        """The modules whose parameters the optimizer should step (everything
        except the frozen backbone)."""
        mods = [self.head]
        mods.append(self.mceam if self.mceam is not None else self.roi_only_proj)
        return mods

    def embed(self, streams):
        """streams dict -> (B, output_dim) embedding (no head)."""
        feats = self.backbone(streams)
        if self.mceam is not None:
            return self.mceam(feats)["embedding"]
        roi_cls, _ = feats["roi"]
        return self.roi_only_proj(roi_cls)

    def forward(self, streams):
        return self.head(self.embed(streams))
