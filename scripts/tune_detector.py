"""Tune detector NMS IoU and confidence on SYNTHETIC VAL frames only.

1) NMS IoU: the grid value with the highest val mAP@0.5:0.95 (overlapping organisms need
   a high-enough IoU so neighbours are not suppressed; too high keeps duplicates).
2) Confidence: the value maximising detection F1 (greedy match at IoU >= match_iou) on val.
Writes results/detector_tuning_nms.csv, detector_tuning_conf.csv and updates results/thresholds.json.
Usage:  python -m scripts.tune_detector
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.common.config import load_config, resolve_path
from src.detect.yolo_utils import detection_prf, greedy_match, ultralytics_offline


def main() -> None:
    ultralytics_offline()
    from ultralytics import YOLO

    cfg = load_config()
    dcfg = cfg["detector"]
    synth = resolve_path(cfg, "synthetic_data")
    model = YOLO(str(resolve_path(cfg, "models") / "detector_best.pt"))
    out = resolve_path(cfg, "results")

    rows = []
    for iou in dcfg["nms_iou_grid"]:
        m = model.val(data=str(synth / "data.yaml"), split="val", iou=iou, conf=0.001, imgsz=dcfg["image_size"],
                      batch=dcfg["batch_size"], plots=False, verbose=False, workers=0)
        rows.append({"nms_iou": iou, "val_map50": float(m.box.map50), "val_map50_95": float(m.box.map)})
        print(rows[-1])
    nms = pd.DataFrame(rows)
    nms.to_csv(out / "detector_tuning_nms.csv", index=False)
    best_iou = float(nms.sort_values(["val_map50_95", "val_map50"], ascending=False).iloc[0]["nms_iou"])

    gt = json.loads((synth / "gt_val.json").read_text())
    preds = {}
    for name in gt:
        r = model.predict(str(synth / "images" / "val" / f"{name}.jpg"), conf=0.001, iou=best_iou,
                          imgsz=dcfg["image_size"], verbose=False)[0]
        preds[name] = (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())
    crow = []
    for conf in np.round(np.arange(0.05, 0.95, dcfg["conf_grid_step"]), 3):
        tp = n_pred = n_gt = 0
        for name, objs in gt.items():
            b, s = preds[name]
            keep = s >= conf
            g = np.array([o["box"] for o in objs]).reshape(-1, 4)
            tp += len(greedy_match(b[keep], s[keep], g, dcfg["match_iou"]))
            n_pred += int(keep.sum()); n_gt += len(g)
        p, r, f1 = detection_prf(tp, n_pred, n_gt)
        crow.append({"conf": float(conf), "precision": p, "recall": r, "f1": f1})
    conf_df = pd.DataFrame(crow)
    conf_df.to_csv(out / "detector_tuning_conf.csv", index=False)
    best = conf_df.loc[conf_df["f1"].idxmax()]

    path = out / "thresholds.json"
    th = json.loads(path.read_text()) if path.exists() else {}
    th.update({"nms_iou": best_iou, "det_conf": float(best["conf"]),
               "det_val_precision": float(best["precision"]), "det_val_recall": float(best["recall"]),
               "det_val_f1": float(best["f1"]), "det_match_iou": dcfg["match_iou"]})
    path.write_text(json.dumps(th, indent=2))
    print(f"chosen NMS IoU {best_iou}, conf {best['conf']:.2f} (val F1 {best['f1']:.4f}, "
          f"P {best['precision']:.4f}, R {best['recall']:.4f})")


if __name__ == "__main__":
    main()
