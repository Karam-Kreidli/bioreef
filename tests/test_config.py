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


def test_missing_explicit_config_errors():
    # An EXPLICITLY-requested config path that doesn't exist must hard-error, not
    # silently fall back to defaults (which would quietly change the benchmark).
    import pytest
    with pytest.raises(FileNotFoundError):
        BenchmarkConfig.from_yaml("does/not/exist.yaml")
    print("test_missing_explicit_config_errors OK")


def test_unknown_field_errors():
    # A typo'd key inside a config section must be rejected, not ignored.
    import tempfile, os, pytest
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("inclusion:\n  min_deploymets: 3\n")   # typo: min_deploymets
        tmp = f.name
    try:
        with pytest.raises(ValueError):
            BenchmarkConfig.from_yaml(tmp)
        print("test_unknown_field_errors OK")
    finally:
        os.unlink(tmp)


def test_bad_ratios_error():
    import pytest
    c = BenchmarkConfig()
    with pytest.raises(ValueError):
        c.apply_overrides(ratios=[0.5, 0.3])          # only 2
    with pytest.raises(ValueError):
        c.apply_overrides(ratios=[0.5, 0.3, 0.3])     # sum != 1
    print("test_bad_ratios_error OK")


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
    test_missing_explicit_config_errors()
    test_unknown_field_errors()
    test_bad_ratios_error()
    test_cli_override_precedence()
    print("\nALL CONFIG TESTS PASSED")
