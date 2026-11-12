"""Real-image pilot: compare pipeline counts with manual counts. Never used for tuning anything.

counts.csv: column `image` (file name, with or without extension), column `total_count`, and optionally one
column per organism class (copepod, diatom, ...). Empty cells mean "not counted" and are skipped.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.pipeline.counting import ORGANISM, REJECT, UNCERTAIN, Detection

TAG = "real pilot, not used for tuning"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def list_images(folder: Path) -> list[Path]:
    """Image files directly in `folder`, sorted by name."""
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_manual_counts(path: Path, organisms: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Read counts.csv. Returns (table indexed by image stem, class columns present, ignored columns)."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if "image" not in df.columns or "total_count" not in df.columns:
        raise ValueError("counts.csv needs columns 'image' and 'total_count'")
    df["image"] = df["image"].astype(str).str.strip().map(lambda s: Path(s).stem)
    if df["image"].duplicated().any():
        raise ValueError(f"duplicate images in counts.csv: {sorted(df.loc[df['image'].duplicated(), 'image'])}")
    class_cols = [c for c in organisms if c in df.columns]
    ignored = [c for c in df.columns if c not in {"image", "total_count", *class_cols}]
    for c in ["total_count", *class_cols]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("image")[["total_count", *class_cols]], class_cols, ignored


def predicted_counts(dets: list[Detection], organisms: list[str]) -> dict[str, int]:
    """Per-image predicted counts.

    pred_total_organisms_plus_uncertain: confident organisms + uncertain (an uncertain item is most likely an
    organism the classifier is unsure of); this is the main number compared with a manual total.
    pred_total_confident: confident organisms only (a lower bound). Rejects are never counted as organisms.
    """
    out = {f"pred_{c}": sum(d.status == ORGANISM and d.label == c for d in dets) for c in organisms}
    out["pred_uncertain"] = sum(d.status == UNCERTAIN for d in dets)
    out["pred_reject"] = sum(d.status == REJECT for d in dets)
    out["pred_total_confident"] = sum(out[f"pred_{c}"] for c in organisms)
    out["pred_total_organisms_plus_uncertain"] = out["pred_total_confident"] + out["pred_uncertain"]
    out["detections"] = len(dets)
    return out


def error_summary(pred: pd.Series, true: pd.Series) -> dict[str, float]:
    """MAE and mean signed error (pred - true) over images where both are known."""
    ok = pred.notna() & true.notna()
    e = (pred[ok] - true[ok]).astype(float)
    return {"n_images": int(ok.sum()),
            "MAE": float(e.abs().mean()) if len(e) else float("nan"),
            "mean_signed_error": float(e.mean()) if len(e) else float("nan"),
            "mean_true": float(true[ok].mean()) if len(e) else float("nan"),
            "mean_pred": float(pred[ok].mean()) if len(e) else float("nan")}


def compare(per_image: pd.DataFrame, manual: pd.DataFrame | None, class_cols: list[str]
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join predictions (indexed by image stem) with manual counts; return (per-image table, summary table)."""
    table = per_image.copy()
    rows = []
    if manual is not None:
        table = table.join(manual.add_prefix("manual_"), how="left")
        for name, col in (("total (organisms + uncertain)", "pred_total_organisms_plus_uncertain"),
                          ("total (confident organisms only)", "pred_total_confident")):
            table[f"error_{col}"] = table[col] - table["manual_total_count"]
            rows.append({"quantity": name, **error_summary(table[col], table["manual_total_count"])})
        for c in class_cols:
            rows.append({"quantity": c, **error_summary(table[f"pred_{c}"], table[f"manual_{c}"])})
    table.insert(0, "note", TAG)
    summary = pd.DataFrame(rows)
    if len(summary):
        summary.insert(0, "note", TAG)
    return table, summary


def unmatched(images: list[str], manual: pd.DataFrame | None) -> tuple[list[str], list[str]]:
    """(images without a manual count, manual rows without an image)."""
    if manual is None:
        return [], []
    have = set(images)
    return sorted(have - set(manual.index)), sorted(set(manual.index) - have)

