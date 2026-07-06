"""
Benchmark configuration — the single source of truth for the inclusion rules and
split parameters that DEFINE the benchmark.

Layering (lowest to highest precedence):
    1. dataclass defaults here
    2. configs/benchmark.yaml (or any --config path)
    3. explicit CLI overrides

Every entry point loads this so train/test/export/make_subset agree on what the
benchmark is. See configs/benchmark.yaml for the shipped values.
"""

import os
from dataclasses import dataclass, field, fields
from typing import List, Optional

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "benchmark.yaml"
)


@dataclass
class BenchmarkConfig:
    # Inclusion rules.
    min_samples: int = 20
    min_deployments: int = 3
    filter_placeholders: bool = True
    # Split.
    ratios: List[float] = field(default_factory=lambda: [0.70, 0.15, 0.15])
    split_seed: int = 0
    # Data location — set ONCE here so no script needs --csv/--img_dir per run.
    csv_path: str = ""
    img_dir: str = ""
    # Extra image dirs searched (in order) when a frame isn't in img_dir — for
    # datasets spread across folders (e.g. frames_1/ + frames_2/).
    extra_img_dirs: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Optional[str] = None) -> "BenchmarkConfig":
        """Load from a nested YAML (inclusion:/split:) into the flat dataclass.
        Missing file or fields fall back to the dataclass defaults."""
        cfg = cls()
        path = path or DEFAULT_CONFIG_PATH
        if not os.path.exists(path):
            return cfg
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        inc = data.get("inclusion", {})
        spl = data.get("split", {})
        dat = data.get("data", {})
        flat = {**inc, **spl, **dat}
        for fld in fields(cls):
            if fld.name in flat and flat[fld.name] is not None:
                setattr(cfg, fld.name, flat[fld.name])
        return cfg

    def apply_overrides(self, **kwargs) -> "BenchmarkConfig":
        """Override fields with any non-None values (CLI precedence)."""
        for k, v in kwargs.items():
            if v is not None and hasattr(self, k):
                setattr(self, k, v)
        return self
