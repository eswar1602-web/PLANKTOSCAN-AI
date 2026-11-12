"""Ultralytics helpers: offline/no-telemetry setup, box matching, detection F1."""

from __future__ import annotations

import os

import numpy as np

# Must be set before ultralytics is imported anywhere in the process.
os.environ.setdefault("YOLO_OFFLINE", "1")        # is_online() -> False
os.environ.setdefault("YOLO_AUTOINSTALL", "0")    # never pip-install missing packages


def _blocked_download(*args, **kwargs):
    raise RuntimeError(f"download blocked by project policy: {args[:1]} {kwargs.get('url', '')}")


def ultralytics_offline() -> None:
    """Disable Ultralytics network side effects: analytics sync and the AMP self-check download.

    The AMP check downloads a pretrained YOLO model to compare FP16/FP32 outputs; downloads are
    not allowed in this project, and RTX 30xx GPUs support FP16 AMP, so the check is skipped.
    """
    from ultralytics import settings
    settings.update({"sync": False})
    import ultralytics.data.utils as data_utils
    import ultralytics.engine.trainer as trainer
    import ultralytics.utils.checks as checks
    trainer.check_amp = lambda *a, **k: True
    # check_font downloads Arial.ttf (the Windows font is 'arial.ttf'; the lookup is case-sensitive).
    # We never draw with Ultralytics (plots=False; annotations are drawn with OpenCV), so skip it.
    data_utils.check_font = lambda *a, **k: None
    checks.check_font = lambda *a, **k: None
    # Backstop: any remaining download attempt fails loudly instead of fetching something.
    import ultralytics.utils.downloads as downloads
    downloads.safe_download = _blocked_download
    original_asset = downloads.attempt_download_asset

    def local_only_asset(file, *args, **kwargs):
        """Allow loading files that already exist locally; block anything that would download."""
        from pathlib import Path
        if Path(str(file)).exists():
            return original_asset(file, *args, **kwargs)
        return _blocked_download(file)

    downloads.attempt_download_asset = local_only_asset
    data_utils.safe_download = _blocked_download
    checks.safe_download = _blocked_download


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (N, 4) and b (M, 4) in x0, y0, x1, y1."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0]); iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2]); iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def greedy_match(pred: np.ndarray, scores: np.ndarray, gt: np.ndarray, iou_thr: float) -> list[tuple[int, int]]:
    """Match predictions (highest score first) to unmatched ground truth with IoU >= iou_thr."""
    if len(pred) == 0 or len(gt) == 0:
        return []
    ious = iou_matrix(pred, gt)
    used: set[int] = set()
    pairs = []
    for i in np.argsort(-scores):
        j_order = np.argsort(-ious[i])
        for j in j_order:
            if ious[i, j] < iou_thr:
                break
            if j not in used:
                used.add(int(j)); pairs.append((int(i), int(j)))
                break
    return pairs


def detection_prf(tp: int, n_pred: int, n_gt: int) -> tuple[float, float, float]:
    """Precision, recall, F1 from counts."""
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gt if n_gt else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)
