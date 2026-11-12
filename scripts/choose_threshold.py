"""Choose the 'uncertain' threshold on the VAL split for the final classifier.

Rule: the smallest max-softmax threshold whose accepted val predictions reach
uncertainty.target_accepted_accuracy. Writes results/thresholds.json (key uncertainty_threshold)
and results/uncertainty_curve.png / .csv.
Usage:  python -m scripts.choose_threshold
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.classify.model import load_checkpoint
from src.classify.train import load_split_arrays, predict_logits
from src.common.config import load_config, resolve_path
from src.common.seed import get_device


def selective_curve(probs: np.ndarray, y: np.ndarray, grid: np.ndarray) -> pd.DataFrame:
    """For each threshold: coverage (fraction accepted) and accuracy among accepted."""
    conf, pred = probs.max(1), probs.argmax(1)
    rows = []
    for t in grid:
        acc_mask = conf >= t
        cov = float(acc_mask.mean())
        acc = float((pred[acc_mask] == y[acc_mask]).mean()) if acc_mask.any() else float("nan")
        rows.append({"threshold": round(float(t), 3), "coverage": cov, "accepted_accuracy": acc})
    return pd.DataFrame(rows)


def pick_threshold(curve: pd.DataFrame, target: float) -> float:
    """Smallest threshold reaching the target accepted accuracy (fallback: the best one)."""
    ok = curve[curve["accepted_accuracy"] >= target]
    if len(ok):
        return float(ok["threshold"].min())
    return float(curve.loc[curve["accepted_accuracy"].idxmax(), "threshold"])


def main() -> None:
    cfg = load_config()
    device = get_device(cfg["project"]["device"])
    model, meta = load_checkpoint(resolve_path(cfg, "models") / "classifier_best.pt", device)
    x, y, _ = load_split_arrays(cfg, ("val",), meta["use_preprocess"], meta.get("grayscale", False))["val"]   # VAL only
    probs = torch.softmax(torch.from_numpy(predict_logits(model, x, device)), 1).numpy()
    curve = selective_curve(probs, y, np.round(np.arange(0.0, 1.0, 0.01), 2))
    target = cfg["uncertainty"]["target_accepted_accuracy"]
    t = pick_threshold(curve, target)
    row = curve[curve["threshold"] == t].iloc[0]
    out = resolve_path(cfg, "results")
    curve.to_csv(out / "uncertainty_curve.csv", index=False)
    path = out / "thresholds.json"
    th = json.loads(path.read_text()) if path.exists() else {}
    th.update({"uncertainty_threshold": t, "uncertainty_target_accuracy": target,
               "uncertainty_val_coverage": float(row["coverage"]),
               "uncertainty_val_accepted_accuracy": float(row["accepted_accuracy"])})
    path.write_text(json.dumps(th, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot(curve["threshold"], curve["accepted_accuracy"], color="#2a78d6", lw=2, label="accuracy of accepted")
    ax.plot(curve["threshold"], curve["coverage"], color="#eb6834", lw=2, label="coverage (not uncertain)")
    ax.axvline(t, color="#52514e", ls="--", lw=1)
    ax.text(t + 0.01, 0.05, f"chosen t = {t:.2f}", color="#52514e")
    ax.set_xlabel("max softmax threshold", color="#52514e"); ax.set_ylim(0, 1.02)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 0.12))
    ax.set_title("Uncertainty threshold chosen on VAL", loc="left", color="#0b0b0b")
    fig.tight_layout(); fig.savefig(out / "uncertainty_curve.png", dpi=130, facecolor="#fcfcfb")
    print(f"threshold {t:.2f}: val coverage {row['coverage']:.4f}, accepted accuracy {row['accepted_accuracy']:.4f}")


if __name__ == "__main__":
    main()
