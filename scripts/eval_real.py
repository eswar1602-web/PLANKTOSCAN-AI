"""Real-image pilot: run the frozen pipeline on a folder of real images and compare with manual counts.

Usage:  python -m scripts.eval_real --images <folder> [--counts <counts.csv>] [--max-examples 12]
counts.csv: `image`, `total_count`, optional per-class columns (copepod, diatom, ...). Total counts alone work.
Outputs (results/real_pilot/, every one tagged "real pilot, not used for tuning"):
  per_image.csv, summary.csv, summary.json, summary.md, scatter_total.png, annotated/<image>.jpg
Nothing is tuned here: models and thresholds are the frozen ones from models/ and results/thresholds.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.classify.data import load_rgb
from src.common.config import load_config, resolve_path
from src.common.experiment import start_run
from src.evaluation import BLUE, GRID, INK, INK2, ORANGE, SURFACE
from src.pipeline.pipeline import Pipeline, annotate
from src.pipeline.real_eval import TAG, compare, list_images, load_manual_counts, predicted_counts, unmatched


def tag_image(img: np.ndarray) -> np.ndarray:
    """Add a banner with the pilot tag to the top of an annotated RGB image."""
    h = max(18, img.shape[0] // 30)
    band = np.full((h, img.shape[1], 3), 30, np.uint8)
    cv2.putText(band, TAG.upper(), (5, int(h * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, h / 32, (255, 200, 60), 1,
                cv2.LINE_AA)
    return np.vstack([band, img])


def scatter(table: pd.DataFrame, path: Path) -> None:
    """Predicted vs manual total count, both prediction variants, with the y = x line."""
    ok = table["manual_total_count"].notna()
    t = table[ok]
    fig, ax = plt.subplots(figsize=(5.5, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    hi = max(1.0, float(np.nanmax(t[["manual_total_count", "pred_total_organisms_plus_uncertain"]].to_numpy())))
    ax.plot([0, hi * 1.05], [0, hi * 1.05], color=INK2, lw=1, ls="--", label="perfect (y = x)")
    ax.scatter(t["manual_total_count"], t["pred_total_organisms_plus_uncertain"], s=40, color=BLUE,
               edgecolor=SURFACE, lw=1.5, label="organisms + uncertain", zorder=3)
    ax.scatter(t["manual_total_count"], t["pred_total_confident"], s=40, color=ORANGE, marker="s",
               edgecolor=SURFACE, lw=1.5, label="confident organisms only", zorder=3)
    ax.set_xlabel("manual total count", color=INK2)
    ax.set_ylabel("predicted count", color=INK2)
    ax.grid(color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"Real images: predicted vs manual (n = {len(t)})", loc="left", color=INK, fontsize=11)
    fig.text(0.01, 0.01, TAG, color=ORANGE, fontsize=8)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--counts", type=Path)
    ap.add_argument("--max-examples", type=int, default=12)
    ap.add_argument("--out", type=Path, help="default results/real_pilot")
    a = ap.parse_args()
    cfg = load_config()
    run_dir = start_run(cfg, "eval_real")
    out = a.out or resolve_path(cfg, "results") / "real_pilot"
    (out / "annotated").mkdir(parents=True, exist_ok=True)

    images = list_images(a.images)
    if not images:
        raise SystemExit(f"no images in {a.images}")
    pipe = Pipeline(cfg)
    manual, class_cols, ignored = (load_manual_counts(a.counts, pipe.organisms) if a.counts else (None, [], []))

    rows, det_rows = [], []
    for i, p in enumerate(images):
        img = load_rgb(p)
        dets = pipe.run(img)
        rows.append({"image": p.stem, "file": p.name, "height": img.shape[0], "width": img.shape[1],
                     **predicted_counts(dets, pipe.organisms)})
        det_rows += [{"note": TAG, "image": p.stem, **d.to_row()} for d in dets]
        if i < a.max_examples:
            cv2.imwrite(str(out / "annotated" / f"{p.stem}.jpg"),
                        cv2.cvtColor(tag_image(annotate(img, dets)), cv2.COLOR_RGB2BGR))
    per_image = pd.DataFrame(rows).set_index("image")
    table, summary = compare(per_image, manual, class_cols)
    no_count, no_image = unmatched(list(per_image.index), manual)

    table.to_csv(out / "per_image.csv")
    pd.DataFrame(det_rows).to_csv(out / "detections.csv", index=False)
    if len(summary):
        summary.to_csv(out / "summary.csv", index=False)
    has_counts = manual is not None and table["manual_total_count"].notna().any()
    if has_counts:
        scatter(table, out / "scatter_total.png")

    info = {"note": TAG, "images": len(images), "images_folder": str(a.images),
            "counts_file": str(a.counts) if a.counts else None, "per_class_columns": class_cols,
            "ignored_columns": ignored, "images_without_manual_count": no_count,
            "manual_rows_without_image": no_image, "thresholds": {"det_conf": pipe.det_conf,
            "nms_iou": pipe.nms_iou, "uncertainty": pipe.uncertainty}, "run": run_dir.name,
            "summary": summary.drop(columns="note").to_dict("records") if len(summary) else []}
    for d in (out, run_dir):
        (d / "summary.json").write_text(json.dumps(info, indent=2))

    lines = [f"# Real-image pilot ({TAG})", "",
             f"_{len(images)} images from `{a.images}`; frozen models and VAL-chosen thresholds "
             f"(detector conf {pipe.det_conf:.2f}, NMS IoU {pipe.nms_iou}, uncertainty {pipe.uncertainty:.2f}). "
             f"Run `{run_dir.name}`._", ""]
    if len(summary):
        lines += ["Error = predicted - manual. 'organisms + uncertain' treats every uncertain item as an organism; "
                  "'confident organisms only' is a lower bound.", "",
                  summary.drop(columns="note").round(3).to_markdown(index=False), ""]
    else:
        lines += ["No manual counts given: predictions only (see per_image.csv).", ""]
    if ignored:
        lines.append(f"Ignored columns in counts.csv (not one of the organism classes): {', '.join(ignored)}.")
    if no_count:
        lines.append(f"{len(no_count)} image(s) have no manual count and are excluded from the errors.")
    if no_image:
        lines.append(f"{len(no_image)} counts.csv row(s) have no matching image: {', '.join(no_image[:10])}.")
    lines += ["", "Caveat: the models were trained on ZooLake dark-field crops; on other microscopes the domain "
              "shift can be large. These numbers were not used to choose any model or threshold."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
