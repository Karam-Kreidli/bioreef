"""Config loading + override precedence (dataclass -> YAML -> CLI)."""

import os

from bioreef.config import BenchmarkConfig, DEFAULT_CONFIG_PATH


def test_defaults():
    c = BenchmarkConfig()
    assert c.min_samples == 20 and c.min_deployments == 3
    assert c.ratios == [0.70, 0.15, 0.15] and c.split_seed == 0
    print("test_defaults OK")


def test_yaml_load():
    assert os.path.exists(DEFAULT_CONFIG_PATH), DEFAULT_CONFIG_PATH
    c = BenchmarkConfig.from_yaml(DEFAULT_CONFIG_PATH)
    # The shipped benchmark definition (locked decision).
    assert c.min_samples == 20 and c.min_deployments == 3
    assert c.filter_placeholders is True
    print("test_yaml_load OK")


def test_missing_file_falls_back_to_defaults():
    c = BenchmarkConfig.from_yaml("does/not/exist.yaml")
    assert c.min_samples == 20
    print("test_missing_file_falls_back_to_defaults OK")


def test_cli_override_precedence():
    c = BenchmarkConfig.from_yaml(DEFAULT_CONFIG_PATH)
    c.apply_overrides(min_samples=25, min_deployments=None, split_seed=2)
    assert c.min_samples == 25          # overridden
    assert c.min_deployments == 3       # None override ignored -> keeps config value
    assert c.split_seed == 2            # overridden
    print("test_cli_override_precedence OK")


if __name__ == "__main__":
    test_defaults()
    test_yaml_load()
    test_missing_file_falls_back_to_defaults()
    test_cli_override_precedence()
    print("\nALL CONFIG TESTS PASSED")
