"""Image enhancement for dark-field plankton images: illumination correction, denoising, CLAHE.

All functions take and return uint8 RGB arrays (H, W, 3) and preserve the shape, so the same
`preprocess` call can be used on training crops and on crops cut at inference time.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def border_median(img: np.ndarray, width: int = 2) -> np.ndarray:
    """Per-channel median of the outer `width` pixels: the background level of a dark-field crop."""
    w = max(1, min(width, min(img.shape[:2]) // 2))
    border = np.concatenate([
        img[:w].reshape(-1, 3), img[-w:].reshape(-1, 3),
        img[:, :w].reshape(-1, 3), img[:, -w:].reshape(-1, 3),
    ])
    return np.median(border, axis=0)


def correct_illumination(img: np.ndarray, method: str = "border_median", sigma_frac: float = 0.25) -> np.ndarray:
    """Remove the background level (dark field: subtract, never divide by a near-black background).

    border_median: subtract the constant background level estimated on the crop border.
    gaussian_subtract: subtract a heavy Gaussian blur (spatially varying glow), for full frames.
    """
    f = img.astype(np.float32)
    if method == "border_median":
        bg = border_median(img).astype(np.float32)[None, None, :]
    elif method == "gaussian_subtract":
        sigma = max(1.0, sigma_frac * max(img.shape[:2]))
        bg = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma)
    else:
        raise ValueError(f"unknown illumination method: {method}")
    return np.clip(f - bg, 0, 255).astype(np.uint8)


def denoise(img: np.ndarray, method: str = "nl_means", h: float = 5) -> np.ndarray:
    """Edge-preserving noise reduction with small, size-independent parameters."""
    if method == "nl_means":
        return cv2.fastNlMeansDenoisingColored(img, None, h, h, 3, 7)
    if method == "bilateral":
        return cv2.bilateralFilter(img, 5, 2 * h * 5, 3)
    if method == "none":
        return img
    raise ValueError(f"unknown denoise method: {method}")


def clahe(img: np.ndarray, clip_limit: float = 2.0, tile_px: int = 32) -> np.ndarray:
    """CLAHE on the L channel of LAB; tile grid adapts to image size (tile_px pixels per tile)."""
    h, w = img.shape[:2]
    grid = (max(1, round(w / tile_px)), max(1, round(h / tile_px)))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess(img: np.ndarray, pcfg: dict[str, Any]) -> np.ndarray:
    """Apply the enabled steps in a fixed order: illumination -> denoise -> CLAHE."""
    out = img
    il, dn, cl = pcfg["illumination"], pcfg["denoise"], pcfg["clahe"]
    if il["enabled"]:
        out = correct_illumination(out, il["method"], il.get("sigma_frac", 0.25))
    if dn["enabled"]:
        out = denoise(out, dn["method"], dn["h"])
    if cl["enabled"]:
        out = clahe(out, cl["clip_limit"], cl["tile_px"])
    return out
