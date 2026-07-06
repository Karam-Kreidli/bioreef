"""Model: frozen ViT backbone -> MCEAM context fusion -> classifier head, with
the HSLM hierarchical loss for training."""

from .backbone import ViTBackbone, BACKBONES
from .mceam import MCEAM, CrossAttentionBlock
from .hslm_loss import HSLMLoss
from .build import Classifier, ModelConfig
from .timm_baseline import TimmClassifier
from .factory import build_model, trainable_parameters, backbone_is_frozen

__all__ = [
    "ViTBackbone", "BACKBONES", "MCEAM", "CrossAttentionBlock", "HSLMLoss",
    "Classifier", "ModelConfig", "TimmClassifier",
    "build_model", "trainable_parameters", "backbone_is_frozen",
]
