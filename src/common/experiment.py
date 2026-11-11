"""Experiment logging: one folder per run with config, seed, git commit and versions."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

from src.common.config import PROJECT_ROOT, resolve_path

_TRACKED_PACKAGES = (
    "torch", "torchvision", "timm", "ultralytics", "opencv-python",
    "numpy", "scikit-learn", "pandas",
)


def _git_commit() -> str | None:
    """Return the current git commit hash, or None if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_versions() -> dict[str, str | None]:
    """Installed versions of the libraries that affect results."""
    versions: dict[str, str | None] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def start_run(cfg: dict[str, Any], name: str) -> Path:
    """Create runs/<timestamp>_<name>/ and write config.yaml + environment.json into it."""
    run_dir = resolve_path(cfg, "runs") / f"{datetime.now():%Y%m%d_%H%M%S}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    env = {
        "run_name": name,
        "started": datetime.now().isoformat(timespec="seconds"),
        "seed": cfg["project"]["seed"],
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _package_versions(),
        "command": sys.argv,
    }
    with open(run_dir / "environment.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)
    return run_dir
