"""Find byte-identical image files filed under more than one raw class."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def file_md5(path: Path) -> str:
    """MD5 of a file's bytes (identity check only, not security)."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def drop_duplicate_files(df: pd.DataFrame, md5: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve byte-identical files: return (cleaned df, log of removed rows).

    Copies that agree on 'target' -> keep the first, drop the rest (reason 'duplicate').
    Copies that disagree on 'target' -> drop all (reason 'label_conflict'): the ground
    truth is ambiguous, so the image is not used at all.
    """
    df = df.assign(md5=md5.values)
    groups = df.groupby("md5")
    drop_idx: list[int] = []
    reasons: list[str] = []
    for _, g in groups:
        if len(g) < 2:
            continue
        if g["target"].nunique() > 1:
            drop_idx += list(g.index)
            reasons += ["label_conflict"] * len(g)
        else:
            drop_idx += list(g.index[1:])
            reasons += ["duplicate"] * (len(g) - 1)
    removed = df.loc[drop_idx].assign(reason=reasons)
    return df.drop(index=drop_idx).drop(columns="md5").reset_index(drop=True), removed
