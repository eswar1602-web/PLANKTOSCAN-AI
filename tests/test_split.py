"""Tests for the time-block, class-stratified split and its fallback."""

import numpy as np
import pandas as pd
import pytest

from src.data.split import (
    assign_blocks, capture_time, check_no_block_overlap, split_dataset, split_summary,
)

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _synthetic(seed: int = 0, n_days: int = 80) -> pd.DataFrame:
    """3 classes, two raw subtypes for 'a', images spread over n_days with random hours."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        day = pd.Timestamp("2019-01-01") + pd.Timedelta(days=d)
        for cls, raw, lam in (("a", "a1", 6), ("a", "a2", 3), ("b", "b1", 4), ("c", "c1", 2)):
            for _ in range(rng.poisson(lam)):
                t = day + pd.Timedelta(hours=int(rng.integers(0, 24)), minutes=int(rng.integers(0, 60)))
                rows.append({"target": cls, "raw_class": raw, "timestamp": str(t), "path": f"x{len(rows)}.jpeg"})
    return pd.DataFrame(rows)


def test_every_image_assigned_and_blocks_not_shared() -> None:
    out, _ = split_dataset(_synthetic(), RATIOS, seed=42)
    assert out["split"].isin(RATIOS).all()
    assert check_no_block_overlap(out) == []
    # whole days: no calendar day appears in two splits
    assert (out.groupby(out["time"].dt.date)["split"].nunique() == 1).all()


def test_ratios_close_to_target_per_class() -> None:
    out, _ = split_dataset(_synthetic(), RATIOS, seed=42)
    share = pd.crosstab(out["target"], out["split"], normalize="index")
    for s, r in RATIOS.items():
        assert (share[s] - r).abs().max() < 0.05, share


def test_deterministic_for_fixed_seed() -> None:
    df = _synthetic()
    a, _ = split_dataset(df, RATIOS, seed=7)
    b, _ = split_dataset(df, RATIOS, seed=7)
    pd.testing.assert_series_equal(a["split"], b["split"])


def test_dominant_block_goes_to_train() -> None:
    """A 'bloom day' larger than the val/test share must land in train."""
    df = _synthetic()
    bloom = pd.DataFrame({"target": "a", "raw_class": "a1", "path": [f"bloom{i}.jpeg" for i in range(400)],
                          "timestamp": "2019-02-10 12:00:00"})
    out, _ = split_dataset(pd.concat([df, bloom], ignore_index=True), RATIOS, seed=42)
    assert (out.loc[out["path"].str.startswith("bloom"), "split"] == "train").all()


def test_two_level_stratification_balances_subtypes() -> None:
    out, _ = split_dataset(_synthetic(), RATIOS, seed=42, stratify_on=("target", "raw_class"))
    share = pd.crosstab(out["raw_class"], out["split"], normalize="index")
    assert (share["test"] - 0.15).abs().max() < 0.06, share


def test_fallback_applied_when_days_cannot_fill_and_hours_help() -> None:
    """Class 'r' lives on 2 days only (but many hours): day blocks cannot fill val+test."""
    df = _synthetic()
    rare = [{"target": "r", "raw_class": "r1", "path": f"r{d}{h}.jpeg",
             "timestamp": f"2019-03-0{d} {h:02d}:30:00"} for d in (1, 2) for h in range(24)]
    out, log = split_dataset(pd.concat([df, pd.DataFrame(rare)], ignore_index=True), RATIOS, seed=42)
    assert log["r"].startswith("applied")
    r = out[out["target"] == "r"]
    assert set(r["split"]) == {"train", "val", "test"}
    assert r["block"].str.startswith("hour:").all()
    assert check_no_block_overlap(out) == []


def test_fallback_reverted_when_hours_do_not_help() -> None:
    """Class 'r' captured in a single hour: hour blocks cannot help, so day blocks are kept."""
    df = _synthetic()
    rare = pd.DataFrame({"target": "r", "raw_class": "r1", "path": [f"r{i}.jpeg" for i in range(20)],
                         "timestamp": "2019-03-01 10:15:00"})
    out, log = split_dataset(pd.concat([df, rare], ignore_index=True), RATIOS, seed=42)
    assert log["r"].startswith("tried, reverted")
    assert out.loc[out["target"] == "r", "block"].str.startswith("day:").all()


def test_capture_time_falls_back_to_filename() -> None:
    df = pd.DataFrame({
        "timestamp": [None, "2018-08-06 03:20:40"],
        "path": ["c/training_data/SPC-EAWAG-0P5X-1533518440410452-5232375038104-048299-021-1104-1610-216-116.jpeg",
                 "c/training_data/other.jpeg"],
    })
    t = capture_time(df)
    assert t.iloc[0] == pd.Timestamp("2018-08-06 03:20:40.410452")   # CEST, naive local time
    assert t.iloc[1] == pd.Timestamp("2018-08-06 03:20:40")


def test_capture_time_raises_when_unusable() -> None:
    with pytest.raises(ValueError):
        capture_time(pd.DataFrame({"timestamp": [None], "path": ["no_stamp.jpeg"]}))


def test_split_summary_totals() -> None:
    out, _ = split_dataset(_synthetic(), RATIOS, seed=42)
    s = split_summary(out)
    assert s["total_images"].sum() == len(out)
    assert (s[["train_images", "val_images", "test_images"]].sum(axis=1) == s["total_images"]).all()


def test_assign_blocks_accepts_multiple_label_columns() -> None:
    df = _synthetic()
    blocks = pd.Series(["b" + str(i // 10) for i in range(len(df))])
    out = assign_blocks(df[["target", "raw_class"]], blocks, RATIOS, seed=1)
    assert out.notna().all()
