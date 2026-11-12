"""Save a before/after grid of the preprocessing steps to results/preprocess_comparison.png.

Usage:  python -m scripts.compare_preprocess
Uses TRAIN-split crops only (one random crop per class, fixed seed).
"""

from __future__ import annotations

import copy

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.config import load_config, resolve_path
from src.preprocess.enhance import preprocess


def _letterbox(img: np.ndarray, size: int = 128) -> np.ndarray:
    h, w = img.shape[:2]
    s = size / max(h, w)
    r = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((size, size, 3), np.uint8)
    y, x = (size - r.shape[0]) // 2, (size - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def main() -> None:
    cfg = load_config()
    root = resolve_path(cfg, "raw_data") / "zoolake" / cfg["data"]["zoolake"]["images_subdir"]
    df = pd.read_csv(resolve_path(cfg, "processed_data") / "split_manifest.csv")
    train = df[df["split"] == "train"]
    picks = train.groupby("target").sample(1, random_state=cfg["project"]["seed"]).sort_values("target")

    base = cfg["preprocess"]
    steps = [("original", None), ("illumination", ["illumination"]),
             ("+ denoise", ["illumination", "denoise"]), ("+ CLAHE (full)", ["illumination", "denoise", "clahe"])]
    fig, axes = plt.subplots(len(picks), len(steps), figsize=(2.2 * len(steps), 2.2 * len(picks)), facecolor="#fcfcfb")
    for r, row in enumerate(picks.itertuples()):
        img = cv2.cvtColor(cv2.imread(str(root / row.path)), cv2.COLOR_BGR2RGB)
        for c, (name, enabled) in enumerate(steps):
            if enabled is None:
                out = img
            else:
                p = copy.deepcopy(base)
                for k in ("illumination", "denoise", "clahe"):
                    p[k]["enabled"] = k in enabled
                out = preprocess(img, p)
            ax = axes[r, c]
            ax.imshow(_letterbox(out))
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=10, color="#0b0b0b")
            if c == 0:
                ax.set_ylabel(f"{row.target}\n{img.shape[1]}x{img.shape[0]}px", fontsize=9, color="#0b0b0b")
    fig.suptitle("Preprocessing steps on train-split crops (shown letterboxed to 128 px)", color="#0b0b0b")
    fig.tight_layout()
    out_path = resolve_path(cfg, "results") / "preprocess_comparison.png"
    fig.savefig(out_path, dpi=110, facecolor="#fcfcfb")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
