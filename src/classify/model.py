"""Classifier construction and loading (EfficientNet via timm, no network access)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # never download weights implicitly

import timm
import torch
import torch.nn as nn


def build_model(backbone: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """EfficientNet (or any timm model) with a fresh head of num_classes outputs."""
    return timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)


def load_checkpoint(path: Path, device: str) -> tuple[nn.Module, dict[str, Any]]:
    """Load a best/last checkpoint saved by train.py; returns (model in eval mode, metadata)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    model = build_model(meta["backbone"], len(meta["classes"]), pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), meta
