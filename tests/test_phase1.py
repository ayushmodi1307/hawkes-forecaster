"""Phase 1 gate: verify project structure and config loading."""
import os
import yaml


def test_config_loads():
    """config.yaml must load and contain all required top-level keys."""
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["project"]["name"] == "hawkes_attack_forecaster"
    assert "seeds" in cfg
    assert "hawkes_fit" in cfg
    assert cfg["hawkes_fit"]["n_starts"] >= 5
    assert cfg["splits"]["embargo_hours"] == 4.0
    assert isinstance(cfg["forecasting"]["horizons_h"], list)
    assert len(cfg["forecasting"]["horizons_h"]) == 2


def test_directories_exist():
    """All required directories must be present."""
    for d in ["data/raw", "data/processed", "src", "app", "tests", "reports"]:
        assert os.path.isdir(d), f"Missing directory: {d}"


def test_requirements_parseable():
    """requirements.txt must list at least 9 packages."""
    with open("requirements.txt") as f:
        lines = f.readlines()
    pkgs = [l.strip() for l in lines if l.strip() and not l.startswith("#")]
    assert len(pkgs) >= 9, f"Only {len(pkgs)} packages listed"


def test_key_packages_importable():
    """Core dependencies must be importable."""
    import numpy  # noqa: F401
    import scipy  # noqa: F401
    import pandas  # noqa: F401
    import matplotlib  # noqa: F401
    import plotly  # noqa: F401
    import yaml  # noqa: F401
    import pytest  # noqa: F401
    import hypothesis  # noqa: F401
