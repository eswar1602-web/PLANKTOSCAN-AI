"""Classifier construction and loading (EfficientNet via timm, no network access)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # never download weights implicitly

import timm
import torch
import torch.nn as nn


def build_model(backbone: str, num_classes: int, pretrained: bool = False,
                weights_file: str | Path | None = None) -> nn.Module:
    """EfficientNet (or any timm model) with a fresh head of num_classes outputs.

    pretrained=True loads ImageNet weights from a LOCAL file only (weights_file, fetched and
    SHA-256-verified by scripts.fetch_pretrained); the classifier head is re-initialised.
    """
    if not pretrained:
        return timm.create_model(backbone, pretrained=False, num_classes=num_classes)
    if weights_file is None or not Path(weights_file).exists():
        raise FileNotFoundError(f"pretrained weights not found locally: {weights_file} "
                                "(run python -m scripts.fetch_pretrained)")
    return timm.create_model(backbone, pretrained=True, num_classes=num_classes,
                             pretrained_cfg_overlay={"file": str(weights_file)})


def load_checkpoint(path: Path, device: str) -> tuple[nn.Module, dict[str, Any]]:
    """Load a best/last checkpoint saved by train.py; returns (model in eval mode, metadata)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    model = build_model(meta["backbone"], len(meta["classes"]), pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), meta
