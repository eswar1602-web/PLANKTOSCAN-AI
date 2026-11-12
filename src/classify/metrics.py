"""Classification metrics: per-class P/R/F1, macro F1 (all and organisms only), confusion matrix."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, classes: list[str], reject_class: str | None = "reject"
) -> dict[str, Any]:
    """All metrics as a JSON-serialisable dict. Labels are integer indices into `classes`."""
    labels = list(range(len(classes)))
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    org = [i for i, c in enumerate(classes) if c != reject_class]
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f1)),
        "macro_f1_organisms": float(np.mean(f1[org])),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]), "support": int(sup[i])}
            for i, c in enumerate(classes)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classes": classes,
    }


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro F1 over all classes (zero F1 for classes never predicted)."""
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0)
    return float(np.mean(f1))
