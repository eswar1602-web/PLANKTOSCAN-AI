"""Measure detector box error on SYNTHETIC VAL frames, to set the frame-crop box jitter (VAL only).

For each detection matched to a ground-truth box (IoU >= detector.match_iou), the offset of each
edge divided by the GT box side. Reports mean and std per edge and pooled; the pooled std is the
value proposed for classifier.frame_crops.box_jitter. Writes results/box_jitter_val.json.
Usage:  python -m scripts.calibrate_box_jitter
"""

from __future__ import annotations

import json

import numpy as np

from src.classify.data import load_rgb
from src.common.config import load_config, resolve_path
from src.common.experiment import start_run
from src.common.seed import set_seed
from src.detect.yolo_utils import greedy_match
from src.pipeline.pipeline import Pipeline


def edge_offsets(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """(N, 4) edge offsets (pred - gt) divided by GT width (x edges) or height (y edges)."""
    w, h = gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]
    return (pred - gt) / np.stack([w, h, w, h], 1)


def main() -> None:
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    run_dir = start_run(cfg, "calibrate_box_jitter")
    pipe = Pipeline(cfg, classifier_path=resolve_path(cfg, "models") / "classifier_v1.pt")
    syn = resolve_path(cfg, "synthetic_data")
    gt = json.loads((syn / "gt_val.json").read_text())
    offs = []
    for fid, objs in gt.items():
        img = load_rgb(syn / "images" / "val" / f"{fid}.jpg")
        boxes, confs = pipe.detect(img, pipe.det_conf)
        g = np.array([o["box"] for o in objs]).reshape(-1, 4)
        for i, j in greedy_match(boxes, confs, g, cfg["detector"]["match_iou"]):
            offs.append(edge_offsets(boxes[i:i + 1], g[j:j + 1])[0])
    o = np.array(offs)
    res = {"split": "val (synthetic frames)", "matched": len(o),
           "edge_mean": dict(zip(["x0", "y0", "x1", "y1"], o.mean(0).round(4).tolist())),
           "edge_std": dict(zip(["x0", "y0", "x1", "y1"], o.std(0).round(4).tolist())),
           "pooled_std": round(float(o.std()), 4),
           "pooled_abs_p90": round(float(np.percentile(np.abs(o), 90)), 4)}
    text = json.dumps(res, indent=2)
    (resolve_path(cfg, "results") / "box_jitter_val.json").write_text(text)
    (run_dir / "box_jitter_val.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
