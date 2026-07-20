"""
Attention figures for the proposed model (paper qualitative figure).

For each of a handful of held-out crops, render a two-part panel:
    left    backbone saliency  — attention rollout over the frozen ViT on the
            ROI crop ('does it localize the fish?')
    right   MCEAM cross-attention — where the fish [CLS] attends across each
            context stream (social / habitat / full_frame) ('what does my
            module pool?')

This is a visualization tool only — it never touches training/eval numbers and
needs no re-runs. Point it at any checkpoint (train.py or run.py format).

    python scripts/visualize.py --weights results/C09_proposed/seed0/checkpoint.pt \
        --n 8 --out_dir figures/attention

The model must have context (MCEAM) for the right-hand panels; the A2 ablation
(context_levels=0) has no cross-attention, so only the backbone column renders.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless: no display needed on the VM
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bioreef.config import BenchmarkConfig
from bioreef.data import split_from_config, FishCropDataset
from bioreef.model import Classifier, ModelConfig
from bioreef.training import set_seed, resolve_device

# ImageNet stats used by ContextHarvester._normalize — invert to display crops.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Context streams that MCEAM attends over, in figure order.
CONTEXT_ORDER = ("social", "habitat", "full_frame")
RES = 224


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="checkpoint (train.py or run.py format)")
    p.add_argument("--csv", default=None, help="override checkpoint/config data path")
    p.add_argument("--img_dir", default=None, help="override checkpoint/config data path")
    p.add_argument("--gpu", default=None, help="GPU, e.g. 1 or cuda:1 or cpu")
    p.add_argument("--n", type=int, default=8, help="number of test crops to render")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--seed", type=int, default=0, help="sampling seed for which crops")
    p.add_argument("--out_dir", default="figures/attention")
    p.add_argument("--alpha", type=float, default=0.5, help="heatmap overlay opacity")
    return p.parse_args()


def load_checkpoint(path):
    """Return (ModelConfig, state_dict, num_classes, idx_to_sp, BenchmarkConfig-kwargs)
    from either checkpoint format (train.py stores model_config/args; run.py
    stores run_config/benchmark_config)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model_config" in ckpt:                 # train.py format
        mcfg = ModelConfig(**ckpt["model_config"])
        bench_kw = ckpt["benchmark_config"]
        idx_to_sp = {int(k): v for k, v in ckpt["idx_to_sp"].items()}
        return mcfg, ckpt["model"], ckpt["num_classes"], idx_to_sp, bench_kw
    if "run_config" in ckpt:                   # run.py format
        rc = ckpt["run_config"]
        mcfg = ModelConfig(
            backbone=rc.get("backbone", "dinov3"),
            context_levels=rc.get("context_levels", 3),
            attention_depth=rc.get("attention_depth", 1),
            unfreeze_blocks=rc.get("unfreeze_blocks", 0),
        )
        idx_to_sp = {int(k): v for k, v in (ckpt.get("idx_to_sp") or {}).items()}
        head_w = ckpt["model"]["head.weight"]
        return mcfg, ckpt["model"], head_w.shape[0], idx_to_sp, ckpt["benchmark_config"]
    raise SystemExit(f"{path}: unrecognized checkpoint (no model_config/run_config)")


def denorm(stream_tensor):
    """(3,H,W) ImageNet-normalized tensor -> (H,W,3) float [0,1] RGB for display.
    ContextHarvester._normalize already converts BGR->RGB, so only the Z-score is
    undone here (no channel flip — flipping again would show false colours)."""
    img = stream_tensor.detach().cpu().numpy().transpose(1, 2, 0)  # (H,W,3)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def upsample_map(weights_1d_or_grid, size=RES):
    """N patch weights (or a grid) -> (size,size) heatmap in [0,1], bilinear."""
    t = torch.as_tensor(weights_1d_or_grid, dtype=torch.float32)
    if t.dim() == 1:
        grid = int(t.numel() ** 0.5)
        t = t.reshape(grid, grid)
    t = t[None, None]  # (1,1,g,g)
    up = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    up = up[0, 0]
    up = (up - up.min()) / (up.max() - up.min() + 1e-8)
    return up.numpy()


def overlay(ax, rgb, heat, title, alpha):
    ax.imshow(rgb)
    if heat is not None:
        ax.imshow(heat, cmap="jet", alpha=alpha)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def render_panel(streams, backbone_sal, mceam_attn, species, out_path, alpha):
    """One row of subplots: [ROI+rollout] then one column per context stream
    with its MCEAM attention. mceam_attn is {} for the no-context ablation."""
    cols = 1 + (len(CONTEXT_ORDER) if mceam_attn else 0)
    fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3.2))
    if cols == 1:
        axes = [axes]

    overlay(axes[0], denorm(streams["roi"]), backbone_sal,
            f"{species}\nROI · backbone rollout", alpha)

    for i, name in enumerate(CONTEXT_ORDER, start=1):
        if not mceam_attn:
            break
        rgb = denorm(streams[name])
        heat = upsample_map(mceam_attn[name]) if name in mceam_attn else None
        overlay(axes[i], rgb, heat, f"{name}\nMCEAM cross-attn", alpha)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    mcfg, state, num_classes, idx_to_sp, bench_kw = load_checkpoint(args.weights)
    bench = BenchmarkConfig(**bench_kw).apply_overrides(csv_path=args.csv, img_dir=args.img_dir)
    if not bench.csv_path:
        raise SystemExit("no dataset CSV: checkpoint stored none; pass --csv")
    device = resolve_device(args.gpu, bench.device)
    print(f"[device] {device}  [context_levels] {mcfg.context_levels}")

    set_seed(args.seed)
    splits = split_from_config(bench.csv_path, bench.img_dir, bench)
    train_s, val_s, test_s = splits[0], splits[1], splits[2]
    samples = {"train": train_s, "val": val_s, "test": test_s}[args.split]

    model = Classifier(mcfg, num_classes).to(device).eval()
    model.load_state_dict(state)
    if mcfg.context_levels == 0:
        print("[warn] context_levels=0 (A2): no MCEAM — only backbone rollout will render.")

    # Deterministic subset of crops for the figure.
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(samples), size=min(args.n, len(samples)), replace=False)
    ds = FishCropDataset([samples[i] for i in idxs], is_train=False)

    for j in range(len(ds)):
        item = ds[j]
        streams_cpu = item["streams"]
        streams = {k: v.unsqueeze(0).to(device) for k, v in streams_cpu.items()}

        with torch.no_grad():
            logits, attn = model.forward_with_attention(streams)
            backbone_sal = model.backbone.attention_rollout(streams["roi"])[0].cpu().numpy()

        pred_idx = int(logits.argmax(1).item())
        pred_sp = idx_to_sp.get(pred_idx, str(pred_idx)) if idx_to_sp else str(pred_idx)
        true_sp = item["species"]
        # Head-average MCEAM attention -> one weight vector per stream.
        mceam = {k: v[0].mean(0).squeeze(0).detach().cpu().numpy() for k, v in attn.items()}

        mark = "OK" if pred_sp == true_sp else "MISS"
        title = f"true:{true_sp}  pred:{pred_sp} [{mark}]"
        out_path = os.path.join(args.out_dir, f"attn_{j:02d}_{true_sp.replace(' ', '_')}.png")
        render_panel(streams_cpu, upsample_map(backbone_sal), mceam, title, out_path, args.alpha)
        print(f"  wrote {out_path}  ({title})")

    print(f"[done] {len(ds)} panels -> {args.out_dir}/")


if __name__ == "__main__":
    main()
