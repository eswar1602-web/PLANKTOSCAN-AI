"""Crop loading, aspect-preserving letterbox, and cached uint8 arrays per split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.preprocess.enhance import preprocess


def load_rgb(path: Path) -> np.ndarray:
    """Read an image file as uint8 RGB (H, W, 3)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def letterbox(img: np.ndarray, size: int) -> np.ndarray:
    """Resize so the longer side == size (aspect preserved), pad with black to size x size.

    Black padding matches the dark-field background, so padding adds no artificial edge.
    """
    h, w = img.shape[:2]
    s = size / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    out = np.zeros((size, size, 3), np.uint8)
    y, x = (size - nh) // 2, (size - nw) // 2
    out[y:y + nh, x:x + nw] = r
    return out


def prepare_crop(img: np.ndarray, size: int, pcfg: dict[str, Any] | None) -> np.ndarray:
    """THE single crop transform used in training and inference: [preprocess] -> letterbox."""
    if pcfg is not None:
        img = preprocess(img, pcfg)
    return letterbox(img, size)


def cache_key(size: int, pcfg: dict[str, Any] | None) -> str:
    """Short hash identifying a (size, preprocessing) combination."""
    blob = json.dumps({"size": size, "pre": pcfg}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def build_array(
    df: pd.DataFrame, images_root: Path, size: int, pcfg: dict[str, Any] | None,
    cache_dir: Path | None = None, name: str = "",
) -> np.ndarray:
    """Stack prepared crops into a (N, size, size, 3) uint8 array, cached on disk by content key."""
    path = None
    if cache_dir is not None:
        key = cache_key(size, pcfg)
        ids = hashlib.sha1("\n".join(df["path"]).encode()).hexdigest()[:10]
        path = cache_dir / f"{name}_{key}_{ids}.npy"
        if path.exists():
            return np.load(path)
    arr = np.stack([
        prepare_crop(load_rgb(images_root / p), size, pcfg)
        for p in tqdm(df["path"], desc=f"prepare {name}", leave=False)
    ])
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)
    return arr
