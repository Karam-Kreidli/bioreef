"""
Run configuration — one YAML per benchmark-panel / ablation config (paper K.1,
K.2). A run config fully describes ONE experiment: which model (backbone, context,
hierarchy, loss, sampler) and its training budget. Combined with a --seed, it
pins exactly one result.

This is the reproducibility unit: a reviewer reads configs/runs/C09_proposed.yaml
to see precisely what produced a table row, and re-runs it with
    python scripts/run.py C09 --seed 0

The fields mirror scripts/train.py's CLI flags 1:1, so run.py just translates a
RunConfig into a train.py invocation — the training loop stays the single source
of truth.
"""

import os
from dataclasses import dataclass, field, fields
from typing import List, Optional

RUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "configs", "runs")


@dataclass
class RunConfig:
    # Identity
    run_id: str = ""                 # short id, e.g. "C09" — also the yaml stem prefix
    name: str = ""                   # human label, e.g. "proposed"
    priority: str = "core"           # core | optional
    description: str = ""

    # Architecture family — decides which builder + input the run uses:
    #   dino    : frozen ViT + optional MCEAM (config-only family; C01, C09, A1-A9)
    #   timm    : a from-scratch/pretrained timm backbone, fine-tuned (C03/C05/C07)
    #   matanet : the official MATANet repo, adapted (C08)
    model_family: str = "dino"

    # --- dino-family model (maps to train.py model flags) ---
    backbone: str = "dinov3"
    context_levels: int = 3
    attention_depth: int = 1
    unfreeze_blocks: int = 0
    # Head shape when context_levels == 0: "mlp" (A2) | "linear" (C01, a strict
    # single-Linear probe). No effect when context_levels > 0.
    probe: str = "mlp"

    # --- timm-family model ---
    timm_name: str = "resnet50"      # any timm model id
    pretrained: bool = True          # fine-tune from ImageNet weights

    # Loss / sampler (long-tail handling)
    hslm: bool = True                # False -> --no_hslm
    loss: str = "cbfocal"            # cbfocal | ce
    sampler: str = "balanced"        # balanced | random
    family_weight: float = 3.0
    genus_weight: float = 2.0
    species_weight: float = 1.0

    # Training budget
    epochs: int = 30
    warmup_epochs: int = 3
    batch_size: int = 32
    lr: float = 1e-4

    # Frozen-feature cache: for a frozen backbone with no context (C01/C02/A2/A9)
    # the ROI [CLS] never changes, so compute it once and train only the head on
    # cached vectors (minutes instead of hours). Ignored for any other run. The
    # cache is un-augmented (standard linear-probe protocol).
    cache_features: bool = False

    # Marine training augmentation on/off. On (default) for the trained models.
    # Off makes the run train on clean crops — matches the un-augmented linear
    # probe, and is the augmentation ablation (strong aug can hurt a FROZEN
    # backbone by widening the train/test feature-distribution gap).
    augment: bool = True

    # Seeds this config is meant to run at (the campaign plan).
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])

    @property
    def slug(self) -> str:
        """Folder/file-safe id, e.g. 'C09_proposed'."""
        base = self.run_id or "run"
        return f"{base}_{self.name}" if self.name else base

    def to_serializable_dict(self) -> dict:
        """Full config with fields that are INERT for this run's model_family
        dropped, so the reviewer-facing run_config.yaml records only what actually
        shaped the run. The timm baseline (timm_name/pretrained) and the dino
        family (backbone/context/attention/unfreeze) are mutually exclusive — a
        dino run ignores timm_name entirely, so emitting 'timm_name: resnet50' in
        a DINOv3 config is misleading provenance. Keep everything for matanet (it
        may consult either)."""
        d = dict(self.__dict__)
        dino_only = ("backbone", "context_levels", "attention_depth",
                     "unfreeze_blocks", "probe")
        timm_only = ("timm_name", "pretrained")
        if self.model_family == "dino":
            for k in timm_only:
                d.pop(k, None)
        elif self.model_family == "timm":
            for k in dino_only:
                d.pop(k, None)
        return d

    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg = cls()
        valid = {f.name for f in fields(cls)}
        # Unknown keys are fatal, not ignored: a typo like `samplar: balanced`
        # would otherwise vanish and the run would use the default sampler.
        unknown = [k for k in data if k not in valid]
        if unknown:
            raise ValueError(f"{path}: unknown config key(s) {unknown}. "
                             f"Valid keys: {sorted(valid)}")
        for k, v in data.items():
            if v is not None:
                setattr(cfg, k, v)
        cfg._validate(path)
        return cfg

    def _validate(self, path: str = "<config>") -> None:
        """Reject invalid ENUM VALUES. A misspelled value (loss: cbfoca1) would
        otherwise fall through to a default (cross-entropy) and silently run a
        different experiment than the config names."""
        checks = {
            "model_family": ("dino", "timm", "matanet"),
            "loss": ("cbfocal", "ce"),
            "sampler": ("balanced", "random"),
            "probe": ("mlp", "linear"),
        }
        for field_name, allowed in checks.items():
            val = getattr(self, field_name)
            if val not in allowed:
                raise ValueError(f"{path}: {field_name}={val!r} invalid; "
                                 f"must be one of {allowed}")
        if self.context_levels not in (0, 1, 3):
            raise ValueError(f"{path}: context_levels={self.context_levels} "
                             "must be 0, 1, or 3")
        if self.epochs <= 0 or self.batch_size <= 0 or self.lr <= 0:
            raise ValueError(f"{path}: epochs/batch_size/lr must be positive")
        if self.warmup_epochs >= self.epochs:
            raise ValueError(f"{path}: warmup_epochs ({self.warmup_epochs}) "
                             f"must be < epochs ({self.epochs})")

    @classmethod
    def find(cls, run_id: str, runs_dir: Optional[str] = None) -> "RunConfig":
        """Resolve a run by its id prefix (e.g. 'C09' -> C09_proposed.yaml)."""
        runs_dir = runs_dir or RUNS_DIR
        matches = [
            f for f in os.listdir(runs_dir)
            if f.endswith((".yaml", ".yml")) and
            (f == f"{run_id}.yaml" or f.startswith(f"{run_id}_"))
        ]
        if not matches:
            raise SystemExit(f"no run config for id '{run_id}' in {runs_dir}")
        if len(matches) > 1:
            raise SystemExit(f"ambiguous run id '{run_id}': {matches}")
        return cls.from_yaml(os.path.join(runs_dir, matches[0]))

    def train_flags(self) -> List[str]:
        """Translate this config into scripts/train.py CLI flags (excludes
        --seed / --csv / --img_dir / --out, which run.py supplies per invocation)."""
        f = [
            "--backbone", self.backbone,
            "--context_levels", str(self.context_levels),
            "--attention_depth", str(self.attention_depth),
            "--unfreeze_blocks", str(self.unfreeze_blocks),
            "--loss", self.loss,
            "--sampler", self.sampler,
            "--family_weight", str(self.family_weight),
            "--genus_weight", str(self.genus_weight),
            "--species_weight", str(self.species_weight),
            "--epochs", str(self.epochs),
            "--warmup_epochs", str(self.warmup_epochs),
            "--batch_size", str(self.batch_size),
            "--lr", str(self.lr),
        ]
        if not self.hslm:
            f.append("--no_hslm")
        return f
