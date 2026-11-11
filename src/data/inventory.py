"""Inventory the raw ZooLake folder: images per class, integrity check, timestamps."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def is_readable_image(path: Path) -> bool:
    """True if PIL can verify and fully decode the image (catches truncated/corrupt files)."""
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:  # verify() invalidates the handle; reopen to decode
            im.load()
            return im.size[0] > 0 and im.size[1] > 0
    except Exception:
        return False


def _load_timestamps(class_dir: Path) -> dict[str, str]:
    """Map image filename -> timestamp string from the class's features.tsv (empty if absent)."""
    tsv = class_dir / "features.tsv"
    if not tsv.exists():
        return {}
    # Select by name: column order differs between classes (e.g. brachionus starts with 'area').
    df = pd.read_csv(tsv, sep="\t", usecols=["url", "timestamp"])
    names = df["url"].astype(str).map(lambda u: Path(u.replace("\\", "/")).name)
    return dict(zip(names, df["timestamp"].astype(str)))


def build_inventory(images_root: Path, check_integrity: bool = True) -> pd.DataFrame:
    """One row per image: raw_class, path, readable, width, height, timestamp."""
    rows: list[dict] = []
    class_dirs = sorted(d for d in images_root.iterdir() if d.is_dir())
    for class_dir in tqdm(class_dirs, desc="classes"):
        stamps = _load_timestamps(class_dir)
        for p in sorted(class_dir.rglob("*")):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            ok, w, h = True, None, None
            if check_integrity:
                ok = is_readable_image(p)
                if ok:
                    with Image.open(p) as im:
                        w, h = im.size
            rows.append({
                "raw_class": class_dir.name,
                "path": str(p.relative_to(images_root)),
                "readable": ok,
                "width": w,
                "height": h,
                "timestamp": stamps.get(p.name),
            })
    return pd.DataFrame(rows)


def class_counts(inv: pd.DataFrame) -> pd.DataFrame:
    """Per-class image counts (readable images only), sorted descending, with share in %."""
    good = inv[inv["readable"]]
    counts = good.groupby("raw_class").size().sort_values(ascending=False).rename("images")
    out = counts.to_frame()
    out["percent"] = (100 * out["images"] / out["images"].sum()).round(2)
    return out


def imbalance_ratio(counts: pd.Series) -> float:
    """Largest class count divided by smallest class count."""
    return float(counts.max() / counts.min())
