"""Tests for duplicate-file handling and the pHash / pixel-correlation leakage check."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.data.dedupe import drop_duplicate_files
from src.data.leakage import cross_split_pairs, hamming_matrix, pixel_correlation


def test_duplicates_same_label_keep_one_conflicts_drop_all() -> None:
    df = pd.DataFrame({"target": ["rotifer", "rotifer", "diatom", "dinobryon", "copepod"],
                       "path": list("abcde")})
    md5 = pd.Series(["h1", "h1", "h2", "h2", "h3"])
    clean, removed = drop_duplicate_files(df, md5)
    assert list(clean["path"]) == ["a", "e"]
    assert sorted(removed["reason"]) == ["duplicate", "label_conflict", "label_conflict"]


def test_hamming_matrix() -> None:
    a = np.array([0b0000, 0b1111], dtype=np.uint64)
    b = np.array([0b0001, 0b1111], dtype=np.uint64)
    assert hamming_matrix(a, b).tolist() == [[1, 4], [3, 0]]


def test_cross_split_pairs_ignores_within_split() -> None:
    df = pd.DataFrame({"split": ["train", "train", "test", "val"]})
    # 0~0 same split (ignored), 0~1 cross split (1 bit), all-ones is 63-64 bits from everything
    hashes = np.array([0, 0, 1, 2**64 - 1], dtype=np.uint64)
    pairs = cross_split_pairs(df, hashes, max_distance=2)
    got = {(r.idx_a, r.idx_b) for r in pairs.itertuples()}
    assert got == {(0, 2), (1, 2)}


def test_pixel_correlation(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    img = (rng.random((40, 50)) * 255).astype(np.uint8)
    Image.fromarray(img).save(tmp_path / "a.png")
    Image.fromarray(img).save(tmp_path / "b.png")
    Image.fromarray(255 - img).save(tmp_path / "c.png")
    assert pixel_correlation(tmp_path / "a.png", tmp_path / "b.png") > 0.999
    assert pixel_correlation(tmp_path / "a.png", tmp_path / "c.png") < -0.9
