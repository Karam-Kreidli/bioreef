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
    # Missing crops: rows whose image is absent are skipped BEFORE the species
    # filters and split run, so an incomplete frame dir silently produces a
    # DIFFERENT benchmark (species count, class indices, split sizes all shift).
    # True = hard error instead. Turn this ON for the final campaign; leave False
    # for exploratory work on partial data.
    strict_images: bool = False
    # Default compute device (e.g. "cuda:0", "cuda:1", "cpu"). Empty -> auto
    # (cuda:0 if available, else cpu). --gpu on a script overrides this.
    device: str = ""

    @classmethod
    def from_yaml(cls, path: Optional[str] = None, require: bool = None) -> "BenchmarkConfig":
        """Load from a nested YAML (inclusion:/split:/data:) into the flat dataclass.

        `require`: if True, an EXPLICITLY-passed config path that doesn't exist is a
        hard error (only a None path silently uses defaults). Defaults to True when a
        path is given, False when it isn't — so a typo'd --config can't silently fall
        back to defaults and quietly change the benchmark definition.

        Unknown keys inside inclusion:/split:/data: are a hard error, not silently
        ignored: a typo like `min_deploymets: 3` would otherwise vanish and the run
        would use the default threshold — a different benchmark with no warning."""
        cfg = cls()
        explicit = path is not None
        path = path or DEFAULT_CONFIG_PATH
        if require is None:
            require = explicit
        if not os.path.exists(path):
            if require:
                raise FileNotFoundError(
                    f"benchmark config not found: {path}. Refusing to silently fall "
                    f"back to dataclass defaults for an explicitly-requested config."
                )
            return cfg
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        inc = data.get("inclusion", {})
        spl = data.get("split", {})
        dat = data.get("data", {})
        flat = {**inc, **spl, **dat}
        valid = {f.name for f in fields(cls)}
        unknown = sorted(set(flat) - valid)
        if unknown:
            raise ValueError(
                f"unknown field(s) in {path}: {unknown}. Check for typos — an "
                f"unrecognised key would otherwise be ignored and the benchmark would "
                f"silently use the default. Known fields: {sorted(valid)}"
            )
        for fld in fields(cls):
            if fld.name in flat and flat[fld.name] is not None:
                setattr(cfg, fld.name, flat[fld.name])
        cfg._validate(path)
        return cfg

    def _validate(self, source: str = "<config>") -> None:
        """Sanity-check the benchmark definition. Ratios must be exactly three,
        positive, summing to 1 (within tolerance); thresholds must be sane."""
        r = self.ratios
        if len(r) != 3:
            raise ValueError(f"{source}: split ratios must be exactly 3 "
                             f"(train/val/test), got {len(r)}: {r}")
        if any(x <= 0 for x in r):
            raise ValueError(f"{source}: all split ratios must be positive, got {r}")
        if abs(sum(r) - 1.0) > 1e-6:
            raise ValueError(f"{source}: split ratios must sum to 1.0, "
                             f"got {sum(r)} from {r}")
        if self.min_samples < 1 or self.min_deployments < 1:
            raise ValueError(f"{source}: min_samples ({self.min_samples}) and "
                             f"min_deployments ({self.min_deployments}) must be >= 1")

    def apply_overrides(self, **kwargs) -> "BenchmarkConfig":
        """Override fields with any non-None values (CLI precedence), then
        re-validate so an override can't produce an invalid benchmark definition."""
        for k, v in kwargs.items():
            if v is not None and hasattr(self, k):
                setattr(self, k, v)
        self._validate("<cli overrides>")
        return self
