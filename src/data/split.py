"""Leakage-safe, class-stratified train/val/test split of whole time blocks."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

SPLITS: tuple[str, ...] = ("train", "val", "test")
_FILENAME_US = re.compile(r"SPC-EAWAG-0P5X-(\d{16})-")


def capture_time(inv: pd.DataFrame) -> pd.Series:
    """Naive local capture time: features.tsv timestamp, else the microsecond stamp in the filename."""
    ts = pd.to_datetime(inv["timestamp"], errors="coerce")
    missing = ts.isna()
    if missing.any():
        us = inv.loc[missing, "path"].str.extract(_FILENAME_US)[0].astype(float)
        fn = pd.to_datetime(us, unit="us", utc=True).dt.tz_convert("Europe/Zurich").dt.tz_localize(None)
        ts = ts.where(~missing, fn)
    if ts.isna().any():
        raise ValueError(f"{int(ts.isna().sum())} images have no usable timestamp")
    return ts


def time_block(times: pd.Series, block: str) -> pd.Series:
    """Block id per image: 'day' -> YYYY-MM-DD, 'hour' -> YYYY-MM-DD HH."""
    fmt = {"day": "%Y-%m-%d", "hour": "%Y-%m-%d %H"}[block]
    return times.dt.strftime(fmt)


def assign_blocks(
    labels: pd.Series | pd.DataFrame, blocks: pd.Series, ratios: dict[str, float], seed: int
) -> pd.Series:
    """Assign whole blocks to splits so every class gets close to the target ratios.

    labels may be one column (e.g. target class) or several (e.g. target + raw subtype);
    with several, the balance of every column's classes is optimised jointly.
    Greedy: blocks are visited from the one holding the largest share of any single
    class downwards (ties shuffled by seed); each goes to the split whose per-class
    deficit it fills best, i.e. minimising sum_s,c (alloc[s,c]/total[c] - ratio[s])^2.
    Big blocks (e.g. a bloom day) are therefore placed first, into train.
    """
    cols = labels.to_frame() if isinstance(labels, pd.Series) else labels
    tables = [pd.crosstab(blocks, cols[c]) for c in cols.columns]
    counts = pd.concat(tables, axis=1).fillna(0)
    mat = counts.to_numpy(dtype=float)
    totals = mat.sum(axis=0)
    frac = mat / totals                                    # block share of each class
    target = np.array([ratios[s] for s in SPLITS])[:, None]

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(counts))
    order = order[np.argsort(-frac[order].max(axis=1), kind="stable")]

    alloc = np.zeros((len(SPLITS), mat.shape[1]))
    block_split: dict[str, str] = {}
    for i in order:
        # adding block i to split s changes only row s: delta = f^2 + 2 f (alloc_s - target_s)
        delta = (frac[i] ** 2 + 2 * frac[i] * (alloc - target)).sum(axis=1)
        s = int(np.argmin(delta))
        alloc[s] += frac[i]
        block_split[counts.index[i]] = SPLITS[s]
    return blocks.map(block_split)


def _min_fill(splits: pd.Series, ratios: dict[str, float]) -> float:
    """Smallest (achieved share / target share) over val and test for one class."""
    share = splits.value_counts(normalize=True)
    return min(share.get(s, 0.0) / ratios[s] for s in ("val", "test"))


def split_dataset(
    df: pd.DataFrame,
    ratios: dict[str, float],
    seed: int,
    block: str = "day",
    fallback_block: str | None = "hour",
    fallback_min_fill: float = 0.5,
    stratify_on: tuple[str, ...] = ("target",),
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Add 'time', 'block' and 'split' columns; return (df, fallback log per class).

    stratify_on: label columns balanced jointly, e.g. ("target", "raw_class") also keeps
    each raw subtype near the target ratios. A class is re-split on its own with finer
    fallback blocks only if its val or test fill is below fallback_min_fill, and the
    finer split is kept only if it actually raises that fill (otherwise reverted).
    """
    out = df.copy()
    out["time"] = capture_time(out)
    out["block"] = block + ":" + time_block(out["time"], block)
    out["split"] = assign_blocks(out[list(stratify_on)], out["block"], ratios, seed)

    log: dict[str, str] = {}
    if fallback_block:
        for cls, grp in out.groupby("target"):
            before = _min_fill(grp["split"], ratios)
            if before >= fallback_min_fill:
                continue
            idx = grp.index
            fine = fallback_block + ":" + time_block(out.loc[idx, "time"], fallback_block)
            new = assign_blocks(out.loc[idx, list(stratify_on)], fine, ratios, seed)
            after = _min_fill(new, ratios)
            if after > before:
                out.loc[idx, "block"], out.loc[idx, "split"] = fine, new
                log[str(cls)] = f"applied (min val/test fill {before:.2f} -> {after:.2f})"
            else:
                log[str(cls)] = f"tried, reverted (min val/test fill {before:.2f} -> {after:.2f}, no gain)"
    return out, log


def check_no_block_overlap(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Return (class, block) pairs whose images ended up in more than one split (should be empty)."""
    n = df.groupby(["target", "block"])["split"].nunique()
    return [tuple(k) for k in n[n > 1].index]


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per class: images and distinct days per split, plus total and % of the dataset."""
    day = df["time"].dt.strftime("%Y-%m-%d")
    imgs = pd.crosstab(df["target"], df["split"]).reindex(columns=list(SPLITS), fill_value=0)
    days = (
        df.assign(day=day).groupby(["target", "split"])["day"].nunique()
        .unstack(fill_value=0).reindex(columns=list(SPLITS), fill_value=0)
    )
    out = imgs.add_suffix("_images").join(days.add_suffix("_days"))
    out["total_images"] = imgs.sum(axis=1)
    out["percent_of_dataset"] = (100 * out["total_images"] / out["total_images"].sum()).round(2)
    for s in SPLITS:
        out[f"{s}_share"] = (out[f"{s}_images"] / out["total_images"]).round(3)
    return out.sort_values("total_images", ascending=False)
