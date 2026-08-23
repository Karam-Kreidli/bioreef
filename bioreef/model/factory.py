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
        # != 0 (not > 0): -1 is the full-fine-tune sentinel, so its backbone params
        # must be included too. `> 0` silently dropped them for unfreeze_blocks=-1.
        if model.cfg.unfreeze_blocks != 0:
            params += [p for p in model.backbone.parameters() if p.requires_grad]
        return params
    return [p for p in model.parameters() if p.requires_grad]


def backbone_is_frozen(model: nn.Module) -> bool:
    """True if the backbone is frozen (DINO family) — the training loop keeps it
    in eval/no-grad. timm baselines fine-tune everything."""
    return isinstance(model, Classifier) and model.cfg.unfreeze_blocks == 0


def llrd_param_groups(model: nn.Module, base_lr: float, layer_decay: float):
    """Build AdamW param groups with LAYER-WISE LR DECAY (LLRD) for a fine-tuned
    DINO Classifier — the standard way to fine-tune a large pretrained transformer.

    Deeper (earlier) backbone layers carry generic features that transfer well and
    should barely move; later layers + the heads (MCEAM/probe/classifier) adapt
    fastest. Each transformer block b (0 = first/earliest) gets:
        lr(b) = base_lr * layer_decay ** (n_blocks - b)
    so the last block gets base_lr*decay, ... the first block gets the smallest lr;
    the pre-block backbone bits (patch embed, cls/pos/register tokens, embeddings)
    get the most-decayed lr of all (depth n_blocks+1); the heads get the full
    base_lr. Portable across depths: reads the block count from the backbone, so it
    works on ViT-B (12 blocks) and ViT-L (24) with no change.

    Returns a list of {params, lr} dicts for optim.AdamW. Only valid for a DINO
    Classifier with an unfrozen backbone (unfreeze_blocks != 0); the caller falls
    back to the flat-LR path otherwise.
    """
    assert isinstance(model, Classifier), "LLRD is only defined for the DINO Classifier"
    if not (0 < layer_decay < 1):
        raise ValueError(f"layer_decay must be in (0,1), got {layer_decay}")

    blocks = model.backbone._find_blocks()
    if blocks is None:
        raise RuntimeError("LLRD: could not locate the backbone transformer blocks")
    n_blocks = len(blocks)

    # Map each backbone parameter to its depth index (0 = earliest block ...
    # n_blocks-1 = last block; pre-block params -> depth -1 sentinel = deepest decay).
    block_param_ids = {}
    for b, blk in enumerate(blocks):
        for p in blk.parameters():
            block_param_ids[id(p)] = b

    groups, seen = [], set()

    # Backbone params, one group per depth so each gets its own decayed lr.
    for p in model.backbone.parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        b = block_param_ids.get(id(p), None)
        if b is None:                       # pre-block (patch embed / tokens / embeddings)
            depth_from_top = n_blocks + 1   # most decayed
        else:
            depth_from_top = n_blocks - b   # last block -> 1, first block -> n_blocks
        lr = base_lr * (layer_decay ** depth_from_top)
        groups.append({"params": [p], "lr": lr})

    # Heads (MCEAM/probe + classifier): full base_lr, no decay.
    head_params = []
    for m in model.trainable_modules():
        for p in m.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                head_params.append(p)
    groups.append({"params": head_params, "lr": base_lr})
    return groups


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
