"""Evaluation helpers: day-block bootstrap CIs, count errors, figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

SURFACE, INK, INK2, GRID, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"


def bootstrap_f1(y: np.ndarray, pred: np.ndarray, groups: np.ndarray, n_classes: int,
                 n_boot: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-class and macro F1 over n_boot resamples of whole groups (e.g. days).

    Returns (per_class (n_boot, n_classes), macro (n_boot,)). Resampling whole days respects the
    correlation between images captured on the same day, giving wider (more honest) intervals.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    per, mac = np.zeros((n_boot, n_classes)), np.zeros(n_boot)
    labels = list(range(n_classes))
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        _, _, f1, sup = precision_recall_fscore_support(y[idx], pred[idx], labels=labels, zero_division=0)
        f1 = np.where(sup > 0, f1, np.nan)          # class absent from the resample -> undefined
        per[b] = f1
        mac[b] = np.nanmean(f1)
    return per, mac


def count_errors(pred: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Per class: MAE, mean signed error (pred - gt), mean GT count per frame. Frames x classes inputs."""
    err = pred - gt
    return pd.DataFrame({"mean_true_per_frame": gt.mean(), "mean_pred_per_frame": pred.mean(),
                         "MAE": err.abs().mean(), "mean_signed_error": err.mean()})


def plot_confusion(cm: np.ndarray, classes: list[str], title: str, path) -> None:
    """Row-normalised confusion matrix (recall per true class), counts annotated."""
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.2, 6.2), facecolor=SURFACE)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", color=INK)
    ax.set_yticks(range(len(classes)), classes, color=INK)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if cmn[i, j] > 0.55 else INK)
    ax.set_xlabel("predicted", color=INK2); ax.set_ylabel("true", color=INK2)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130, facecolor=SURFACE); plt.close(fig)


def plot_f1_ci(df: pd.DataFrame, title: str, path) -> None:
    """Dot + 95% CI bar per class (df: index=class, columns f1, ci_low, ci_high)."""
    d = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(d) + 1.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = np.arange(len(d))
    ax.hlines(y, d["ci_low"], d["ci_high"], color=BLUE, lw=2)
    ax.plot(d["f1"], y, "o", color=BLUE, ms=8, mec=SURFACE, mew=2)
    for yi, (f, hi) in enumerate(zip(d["f1"], d["ci_high"])):
        ax.text(min(hi + 0.01, 1.0), yi, f"{f:.3f}", va="center", color=INK2, fontsize=9)
    ax.set_yticks(y, d.index, color=INK); ax.set_xlim(0, 1.08)
    ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("F1 (dot) with 95% day-block bootstrap CI", color=INK2)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130, facecolor=SURFACE); plt.close(fig)


def plot_count_errors(df: pd.DataFrame, title: str, path) -> None:
    """Signed count error per class (bars) with MAE markers."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(len(df))
    ax.bar(x - 0.18, df["mean_signed_error"], 0.36, color=BLUE, label="mean signed error (pred - true)")
    ax.bar(x + 0.18, df["MAE"], 0.36, color=ORANGE, label="MAE")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, df.index, rotation=30, ha="right", color=INK)
    ax.set_ylabel("objects per frame", color=INK2)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, ncols=2, loc="upper left", bbox_to_anchor=(0, -0.28))
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130, facecolor=SURFACE); plt.close(fig)
