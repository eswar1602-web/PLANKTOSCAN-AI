"""Load and validate config.yaml; resolve project-relative paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG: Path = PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read the YAML config, validate it and return it as a nested dict."""
    with open(path, "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    """Raise ValueError if the config is internally inconsistent."""
    split = cfg["data"]["split"]
    total = split["train"] + split["val"] + split["test"]
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")
    if not isinstance(cfg["project"]["seed"], int):
        raise ValueError("project.seed must be an integer")
    thr = cfg["uncertainty"]["target_accepted_accuracy"]
    if not 0.0 <= thr <= 1.0:
        raise ValueError("uncertainty.target_accepted_accuracy must be in [0, 1]")


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Return an absolute path for a key in cfg['paths'] (e.g. 'raw_data')."""
    return PROJECT_ROOT / cfg["paths"][key]
