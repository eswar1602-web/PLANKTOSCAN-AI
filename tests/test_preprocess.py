"""Unit tests for the preprocessing steps."""

import copy

import numpy as np
import pytest

from src.common.config import load_config
from src.preprocess.enhance import border_median, clahe, correct_illumination, denoise, preprocess


def _blob(h: int = 72, w: int = 76, bg: int = 20) -> np.ndarray:
    """Dark-field-like crop: constant background with a bright ellipse in the middle."""
    yy, xx = np.mgrid[:h, :w]
    mask = ((yy - h / 2) / (h / 4)) ** 2 + ((xx - w / 2) / (w / 4)) ** 2 <= 1
    img = np.full((h, w, 3), bg, np.uint8)
    img[mask] = (180, 160, 120)
    return img


def test_border_median_reads_background() -> None:
    assert border_median(_blob(bg=20)).tolist() == [20, 20, 20]


def test_illumination_border_median_removes_offset_keeps_object() -> None:
    out = correct_illumination(_blob(bg=20), "border_median")
    assert out[0, 0].tolist() == [0, 0, 0]                  # background -> 0
    assert out[36, 38].tolist() == [160, 140, 100]          # object minus the offset


def test_illumination_gaussian_subtract_flattens_gradient() -> None:
    h, w = 200, 200
    grad = np.tile(np.linspace(10, 60, w, dtype=np.float32), (h, 1))
    img = np.repeat(grad[..., None], 3, axis=2).astype(np.uint8)
    out = correct_illumination(img, "gaussian_subtract", sigma_frac=0.25)
    # the glow spans 50 gray levels; after correction the center region is nearly flat
    center = out[50:150, 50:150, 0].astype(int)
    assert center.max() - center.min() < 10


def test_illumination_unknown_method() -> None:
    with pytest.raises(ValueError):
        correct_illumination(_blob(), "divide")


def test_denoise_reduces_noise() -> None:
    rng = np.random.default_rng(0)
    clean = np.full((64, 64, 3), 100, np.uint8)
    noisy = np.clip(clean + rng.normal(0, 10, clean.shape), 0, 255).astype(np.uint8)
    for method in ("nl_means", "bilateral"):
        out = denoise(noisy, method, h=7)
        assert out.shape == noisy.shape and out.dtype == np.uint8
        assert out.astype(float).std() < noisy.astype(float).std()


def test_clahe_increases_contrast_on_low_contrast_image() -> None:
    img = _blob(bg=40)
    img[img > 40] = 60                                      # faint object: 40 vs 60
    out = clahe(img, clip_limit=4.0, tile_px=32)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.ptp(out[..., 0].astype(int)) > np.ptp(img[..., 0].astype(int))


def test_clahe_handles_tiny_image() -> None:
    out = clahe(_blob(16, 20), tile_px=32)                  # grid falls back to 1x1
    assert out.shape == (16, 20, 3)


def test_preprocess_disabled_is_identity_and_deterministic() -> None:
    pcfg = copy.deepcopy(load_config()["preprocess"])
    img = _blob()
    a, b = preprocess(img, pcfg), preprocess(img, pcfg)
    assert np.array_equal(a, b) and a.shape == img.shape
    for step in ("illumination", "denoise", "clahe"):
        pcfg[step]["enabled"] = False
    assert np.array_equal(preprocess(img, pcfg), img)
