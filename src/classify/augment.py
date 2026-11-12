"""Batched GPU augmentation and normalisation (no DataLoader workers needed on Windows)."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_float(batch_u8: torch.Tensor) -> torch.Tensor:
    """(B, H, W, 3) uint8 -> (B, 3, H, W) float in [0, 1]."""
    return batch_u8.permute(0, 3, 1, 2).float().div_(255.0)


def normalize(x: torch.Tensor) -> torch.Tensor:
    """ImageNet mean/std normalisation."""
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def augment(x: torch.Tensor, acfg: dict[str, Any], gen: torch.Generator) -> torch.Tensor:
    """Random flips, rotation + scale (one affine warp, black fill), brightness/contrast jitter."""
    b, dev = x.shape[0], x.device

    def u(lo: float, hi: float) -> torch.Tensor:
        return torch.rand(b, generator=gen, device=dev) * (hi - lo) + lo

    if acfg.get("hflip"):
        m = torch.rand(b, generator=gen, device=dev) < 0.5
        x = torch.where(m.view(-1, 1, 1, 1), x.flip(3), x)
    if acfg.get("vflip"):
        m = torch.rand(b, generator=gen, device=dev) < 0.5
        x = torch.where(m.view(-1, 1, 1, 1), x.flip(2), x)

    ang = u(-1, 1) * math.radians(acfg.get("rotation_deg", 0))
    lo, hi = acfg.get("scale", [1.0, 1.0])
    inv_s = 1.0 / u(lo, hi)
    cos, sin = torch.cos(ang) * inv_s, torch.sin(ang) * inv_s
    theta = torch.stack([torch.stack([cos, -sin, torch.zeros_like(cos)], 1),
                         torch.stack([sin, cos, torch.zeros_like(cos)], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    br, ct = acfg.get("brightness", 0.0), acfg.get("contrast", 0.0)
    x = x * (1 + u(-br, br)).view(-1, 1, 1, 1)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * (1 + u(-ct, ct)).view(-1, 1, 1, 1) + mean
    return x.clamp_(0, 1)
