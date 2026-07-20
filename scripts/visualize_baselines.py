"""
Saliency figures for the fine-tuned BASELINE panel (C03/C05/C06/C07/C04), the
companion to scripts/visualize.py (which handles the MCEAM/dino family).

Per family, the right saliency method differs — attention only exists for ViTs:
    ViT baseline (C06, vit_base)      -> attention rollout  (same method as the
                                         MCEAM figure, so C06-vs-C09 is like-for-like)
    CNN / Swin (C03/C05/C04/C07)      -> Grad-CAM on the last feature stage
                                         (rollout is ill-defined for windowed Swin
                                         attention; CAM is the standard choice)

Renders ROI + heatmap on the SAME held-out crops as visualize.py (pass the same
--seed / --n / --split), so the identical fish appear across every model and the
panel reads as one comparison: "where does each model look?". Visualization only
— never touches training/eval numbers, needs no re-runs.

    python scripts/visualize_baselines.py \
        --weights results/C05_convnext_tiny/seed0/checkpoint.pt \
        --seed 0 --n 8 --out_dir figures/attention_baselines

Requires a checkpoint saved with --save_checkpoint (timm baselines are not saved
by default; re-run the one you want to visualize with --save_checkpoint).
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bioreef.config import BenchmarkConfig
from bioreef.data import split_from_config, FishCropDataset
from bioreef.model import TimmClassifier
from bioreef.training import set_seed, resolve_device
# Reuse the exact display helpers from the MCEAM figure so both figures match.
from scripts.visualize import denorm, upsample_map, overlay, RES


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="a run.py-format timm checkpoint")
    p.add_argument("--csv", default=None)
    p.add_argument("--img_dir", default=None)
    p.add_argument("--gpu", default=None, help="GPU, e.g. 1 or cuda:1 or cpu")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--seed", type=int, default=0,
                   help="MATCH visualize.py's seed to get the same crops across models")
    p.add_argument("--out_dir", default="figures/attention_baselines")
    p.add_argument("--alpha", type=float, default=0.5)
    return p.parse_args()


def load_timm_checkpoint(path):
    """(timm_name, state_dict, num_classes, idx_to_sp, benchmark_kwargs) from a
    run.py checkpoint. Refuses non-timm checkpoints (use visualize.py for dino)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    rc = ckpt.get("run_config", {})
    if rc.get("model_family") != "timm":
        raise SystemExit(f"{path}: not a timm baseline (model_family="
                         f"{rc.get('model_family')}); use scripts/visualize.py")
    idx_to_sp = {int(k): v for k, v in (ckpt.get("idx_to_sp") or {}).items()}
    # num_classes from idx_to_sp if present, else inferred from the head weight.
    num_classes = len(idx_to_sp) if idx_to_sp else _infer_num_classes(ckpt["model"])
    return rc["timm_name"], ckpt["model"], num_classes, idx_to_sp, ckpt["benchmark_config"]


def _infer_num_classes(state):
    """Last 2D tensor's row count is the classifier head width (timm heads vary
    in name: fc / head / classifier), so infer instead of hardcoding."""
    for k in reversed(list(state.keys())):
        t = state[k]
        if t.dim() == 2:
            return t.shape[0]
    raise SystemExit("could not infer num_classes from checkpoint")


def is_vit(timm_name: str) -> bool:
    return timm_name.startswith("vit_")


# ----- attention rollout for a plain timm ViT (C06) --------------------------
def vit_rollout(model, roi):
    """Attention rollout (Abnar & Zuidema 2020) over a timm ViT's blocks.
    Captures each block's attention via a forward hook, averages heads, adds the
    residual identity, multiplies across layers, returns the CLS->patch row as a
    grid heatmap. Same algorithm as backbone.attention_rollout, adapted to timm's
    block API (blocks[i].attn)."""
    net = model.net
    attentions = []

    def hook(module, inp, out):
        # timm Attention returns the projected output, not the attn matrix, so we
        # recompute attn from the module's qkv on the captured input.
        x = inp[0]
        B, N, C = x.shape
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        attn = (q @ k.transpose(-2, -1)) * module.scale
        attentions.append(attn.softmax(dim=-1).detach())

    handles = [blk.attn.register_forward_hook(hook) for blk in net.blocks]
    with torch.no_grad():
        _ = net(roi)
    for h in handles:
        h.remove()

    result = torch.eye(attentions[0].size(-1), device=roi.device)
    for attn in attentions:
        a = attn.mean(1)                       # average heads -> (B,N,N)
        a = a + torch.eye(a.size(-1), device=a.device)[None]
        a = a / a.sum(-1, keepdim=True)
        result = a[0] @ result
    # CLS row (index 0) -> patch tokens (drop CLS + any register/dist tokens).
    n_patch = int(((result.size(-1) - 1)) ** 0.5) ** 2
    cls_to_patch = result[0, -n_patch:]
    return cls_to_patch.cpu().numpy()


# ----- Grad-CAM for CNN / Swin (C03/C04/C05/C07) -----------------------------
def grad_cam(model, roi, target_class, target_layer):
    """Grad-CAM (Selvaraju 2017): weight the target layer's activation maps by the
    global-avg-pooled gradient of the target logit, ReLU, normalize. Works for any
    conv/stage output shaped (B,C,H,W) OR (B,H,W,C) (Swin/ConvNeXt-v2 channels-last)."""
    acts, grads = {}, {}

    def fwd(m, i, o): acts["v"] = o
    def bwd(m, gi, go): grads["v"] = go[0]

    h1 = target_layer.register_forward_hook(fwd)
    h2 = target_layer.register_full_backward_hook(bwd)

    model.zero_grad(set_to_none=True)
    logits = model.net(roi)
    logits[0, target_class].backward()
    h1.remove(); h2.remove()

    a, g = acts["v"], grads["v"]
    # Layout must NOT be inferred from relative dim sizes: a ResNet stage is
    # (B,2048,7,7), where "channels <= spatial" is false and would send NCHW
    # activations down the channels-last branch, producing a garbage CAM.
    # Channels-last here means Swin/ViT-style (B,H,W,C) blocks, where the last
    # dim is the embedding and the two middle dims are a square spatial grid.
    if a.dim() != 4:
        raise RuntimeError(f"Grad-CAM expects a 4-D activation, got {tuple(a.shape)}")
    channels_last = a.shape[1] == a.shape[2] and a.shape[1] != a.shape[3]
    if channels_last:                                    # (B,H,W,C)
        weights = g.mean(dim=(1, 2), keepdim=True)
        cam = (weights * a).sum(-1)                      # (B,H,W)
    else:                                                # (B,C,H,W)
        weights = g.mean(dim=(2, 3), keepdim=True)       # GAP over H,W
        cam = (weights * a).sum(1)                       # (B,H,W)
    cam = F.relu(cam)[0]
    return cam.detach().cpu().numpy()


def pick_cam_layer(net):
    """Last feature stage for Grad-CAM, via timm's feature_info when available;
    else fall back to the last Conv2d / LayerNorm in the backbone."""
    fi = getattr(net, "feature_info", None)
    if fi is not None:
        try:
            name = fi[-1]["module"]                       # e.g. 'stages.3' / 'layer4'
            mod = net
            for part in name.split("."):
                mod = getattr(mod, part) if not part.isdigit() else mod[int(part)]
            return mod
        except Exception:
            pass
    last = None
    for m in net.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.LayerNorm)):
            last = m
    if last is None:
        raise SystemExit("could not find a Grad-CAM target layer")
    return last


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    timm_name, state, num_classes, idx_to_sp, bench_kw = load_timm_checkpoint(args.weights)
    bench = BenchmarkConfig(**bench_kw).apply_overrides(csv_path=args.csv, img_dir=args.img_dir)
    if not bench.csv_path:
        raise SystemExit("no dataset CSV: checkpoint stored none; pass --csv")
    device = resolve_device(args.gpu, bench.device)
    method = "rollout" if is_vit(timm_name) else "grad-cam"
    print(f"[device] {device}  [model] {timm_name}  [method] {method}")

    model = TimmClassifier(timm_name, num_classes, pretrained=False).to(device).eval()
    model.load_state_dict(state)

    set_seed(args.seed)
    splits = split_from_config(bench.csv_path, bench.img_dir, bench)
    samples = {"train": splits[0], "val": splits[1], "test": splits[2]}[args.split]

    rng = np.random.default_rng(args.seed)      # SAME rng as visualize.py -> same crops
    idxs = rng.choice(len(samples), size=min(args.n, len(samples)), replace=False)
    ds = FishCropDataset([samples[i] for i in idxs], is_train=False)
    cam_layer = None if method == "rollout" else pick_cam_layer(model.net)

    for j in range(len(ds)):
        item = ds[j]
        streams = {k: v.unsqueeze(0).to(device) for k, v in item["streams"].items()}
        roi = streams["roi"]

        with torch.no_grad():
            pred_idx = int(model.net(roi).argmax(1).item())

        if method == "rollout":
            heat = upsample_map(vit_rollout(model, roi))
        else:
            heat = upsample_map(grad_cam(model, roi, pred_idx, cam_layer))

        pred_sp = idx_to_sp.get(pred_idx, str(pred_idx)) if idx_to_sp else str(pred_idx)
        true_sp = item["species"]
        mark = "OK" if pred_sp == true_sp else "MISS"

        fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.4))
        overlay(ax, denorm(item["streams"]["roi"]), heat,
                f"{timm_name} [{method}]\ntrue:{true_sp} pred:{pred_sp} [{mark}]", args.alpha)
        out_path = os.path.join(args.out_dir,
                                f"{timm_name}_{j:02d}_{true_sp.replace(' ', '_')}.png")
        fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  wrote {out_path}  (true:{true_sp} pred:{pred_sp} [{mark}])")

    print(f"[done] {len(ds)} panels -> {args.out_dir}/")


if __name__ == "__main__":
    main()
