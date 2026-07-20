"""
Model factory — build a model from a RunConfig by dispatching on model_family.

Every family returns an nn.Module with the SAME interface: forward(streams) ->
logits, where `streams` is the dataset's context-crop dict. This uniformity is
what lets one training/eval loop serve all families and keeps the preprocessing
fairness rule intact (all families see the same crops from the same pipeline).

    dino    -> Classifier          (frozen ViT + optional MCEAM)   [config-only family]
    timm    -> TimmClassifier      (fine-tuned backbone on ROI crop)
    matanet -> run from the official repo — NOT built here (see build_matanet)
"""

import torch.nn as nn

from .build import Classifier, ModelConfig
from .timm_baseline import TimmClassifier


def build_model(run_cfg, num_classes: int) -> nn.Module:
    """Instantiate the model for a run. `run_cfg` is a bioreef.run_config.RunConfig."""
    fam = run_cfg.model_family
    if fam == "dino":
        mcfg = ModelConfig(
            backbone=run_cfg.backbone,
            context_levels=run_cfg.context_levels,
            attention_depth=run_cfg.attention_depth,
            unfreeze_blocks=run_cfg.unfreeze_blocks,
            probe=getattr(run_cfg, "probe", "mlp"),
        )
        return Classifier(mcfg, num_classes)
    if fam == "timm":
        return TimmClassifier(run_cfg.timm_name, num_classes, pretrained=run_cfg.pretrained)
    if fam == "matanet":
        return build_matanet(run_cfg, num_classes)
    raise SystemExit(f"unknown model_family '{fam}' (dino | timm | matanet)")


def trainable_parameters(model: nn.Module):
    """Parameters the optimizer should step. For the DINO family the frozen
    backbone is excluded via Classifier.trainable_modules(); for timm every
    parameter is trainable (full fine-tune)."""
    if isinstance(model, Classifier):
        params = []
        for m in model.trainable_modules():
            params += list(m.parameters())
        if model.cfg.unfreeze_blocks > 0:
            params += [p for p in model.backbone.parameters() if p.requires_grad]
        return params
    return [p for p in model.parameters() if p.requires_grad]


def backbone_is_frozen(model: nn.Module) -> bool:
    """True if the backbone is frozen (DINO family) — the training loop keeps it
    in eval/no-grad. timm baselines fine-tune everything."""
    return isinstance(model, Classifier) and model.cfg.unfreeze_blocks == 0


def build_matanet(run_cfg, num_classes: int):
    """MATANet (paper C08) shares this work's multi-context architecture, so the
    fair comparison is to run it FROM THE OFFICIAL REPO on the OzFish split, not
    to reimplement it here inside our Classifier.

    Official repo: https://github.com/dhlee-work/fathomnet-cvpr2025-ssl
    The full bridge + workflow lives in matanet/ (export_ozfish.py, patch_matanet.py,
    ingest_predictions.py); see matanet/README.md.
    """
    raise NotImplementedError(
        "C08 MATANet is run from the official repo "
        "(https://github.com/dhlee-work/fathomnet-cvpr2025-ssl), not built here. "
        "Use the bridge in matanet/ (see matanet/README.md)."
    )
