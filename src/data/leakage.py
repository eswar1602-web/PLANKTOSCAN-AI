"""Perceptual-hash near-duplicate check across splits (second safeguard after time blocks)."""

from __future__ import annotations

from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def phash_int(path: Path) -> int:
    """64-bit perceptual hash of an image as a Python int."""
    with Image.open(path) as im:
        return int(str(imagehash.phash(im.convert("L"))), 16)


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between two uint64 hash arrays (len(a) x len(b))."""
    return np.bitwise_count(a[:, None] ^ b[None, :]).astype(np.uint8)


def cross_split_pairs(
    df: pd.DataFrame, hashes: np.ndarray, max_distance: int, chunk: int = 512
) -> pd.DataFrame:
    """All image pairs from *different* splits with pHash distance <= max_distance."""
    rows: list[dict] = []
    splits = df["split"].to_numpy()
    for sa, sb in (("train", "val"), ("train", "test"), ("val", "test")):
        ia, ib = np.flatnonzero(splits == sa), np.flatnonzero(splits == sb)
        for start in range(0, len(ia), chunk):
            blk = ia[start:start + chunk]
            d = hamming_matrix(hashes[blk], hashes[ib])
            r, c = np.nonzero(d <= max_distance)
            for i, j in zip(blk[r], ib[c]):
                rows.append({"idx_a": int(i), "idx_b": int(j), "split_a": sa, "split_b": sb,
                             "distance": int(hashes[i] ^ hashes[j]).bit_count()})
    return pd.DataFrame(rows, columns=["idx_a", "idx_b", "split_a", "split_b", "distance"])


def pixel_correlation(path_a: Path, path_b: Path, size: int = 64) -> float:
    """Pearson correlation of two images after grayscale + resize to size x size."""
    def load(p: Path) -> np.ndarray:
        with Image.open(p) as im:
            return np.asarray(im.convert("L").resize((size, size)), dtype=float).ravel()
    a, b = load(path_a), load(path_b)
    if a.std() == 0 or b.std() == 0:
        return float(np.array_equal(a, b))
    return float(np.corrcoef(a, b)[0, 1])


def compute_hashes(df: pd.DataFrame, images_root: Path) -> np.ndarray:
    """pHash for every row of df (column 'path' relative to images_root)."""
    return np.array(
        [phash_int(images_root / p) for p in tqdm(df["path"], desc="pHash")], dtype=np.uint64
    )
