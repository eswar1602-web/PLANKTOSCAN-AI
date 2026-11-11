"""Smoke tests for config loading and seeding (Phase 1)."""

import copy
import random

import numpy as np
import pytest

from src.common.config import load_config, validate_config
from src.common.seed import set_seed


def test_config_loads_and_is_valid() -> None:
    cfg = load_config()
    assert cfg["project"]["seed"] == 42
    assert cfg["data"]["dataset"] == "zoolake"


def test_bad_split_ratios_rejected() -> None:
    cfg = copy.deepcopy(load_config())
    cfg["data"]["split"]["test"] = 0.5
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_seed_is_reproducible() -> None:
    set_seed(123)
    a = (random.random(), np.random.rand(3).tolist())
    set_seed(123)
    b = (random.random(), np.random.rand(3).tolist())
    assert a == b


def test_torch_seed_is_reproducible() -> None:
    """Exercises the torch branch of set_seed (skipped only if torch is missing)."""
    torch = pytest.importorskip("torch")
    set_seed(7, deterministic=True)
    a = torch.rand(4)
    set_seed(7, deterministic=True)
    b = torch.rand(4)
    assert torch.equal(a, b)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    if torch.cuda.is_available():
        set_seed(7)
        c = torch.rand(4, device="cuda")
        set_seed(7)
        d = torch.rand(4, device="cuda")
        assert torch.equal(c, d)
