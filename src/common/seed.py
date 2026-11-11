"""Global seeding so every experiment is reproducible."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and (if installed) torch, optionally forcing deterministic cuDNN."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # torch is optional for data/preprocess-only steps
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> str:
    """Return 'cuda' or 'cpu' based on preference and availability (GPU-optional)."""
    if preference == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if preference == "cuda":
        raise RuntimeError("device=cuda requested but CUDA is not available")
    return "cpu"
