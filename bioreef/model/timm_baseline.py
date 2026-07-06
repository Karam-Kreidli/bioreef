"""
timm baseline models (paper panel C03/C04/C05/C06/C07): a standard backbone
fine-tuned end-to-end as a flat classifier.

Uniform interface with the DINO family: forward takes the dataset's `streams`
dict and returns logits. A timm baseline only uses streams["roi"] — the SAME
ImageNet-normalized 224x224 ROI crop the DINO family gets — so the preprocessing
fairness rule (Section 5.1) holds automatically; the other context streams are
simply ignored. This is the fair single-crop comparison to context-aware MCEAM.
"""

import torch.nn as nn


class TimmClassifier(nn.Module):
    """Fine-tuned timm backbone on the ROI crop. `name` is any timm model id
    (resnet50, convnext_tiny, tf_efficientnetv2_s, vit_base_patch16_224,
    swin_base_patch4_window7_224, ...)."""

    def __init__(self, name: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        import timm
        self.name = name
        # num_classes drives timm's own classifier head — fully trainable.
        self.net = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, streams):
        # Single-crop model: take the ROI stream, ignore social/habitat/full_frame.
        return self.net(streams["roi"])
