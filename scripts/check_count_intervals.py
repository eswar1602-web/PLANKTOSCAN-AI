"""Measure how often the 95% classification count interval contains the true count, on synthetic VAL frames.

Usage:  python -m scripts.check_count_intervals [--probabilities raw|calibrated]
Uses the deployed models and thresholds unchanged; VAL frames only (never test). Writes
results/count_intervals_val[_calibrated].{json,md} and a run folder in runs/.

Two coverages, per (frame, class):
- end to end: interval from ALL detections vs the true count in the frame (includes detector misses
  and false detections, which the interval does not model);
- classification only: interval from the detections matched to a ground-truth object vs the true
  labels of those objects (isolates the classifier's probability calibration).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.common.config import load_config, resolve_path
from src.common.experiment import start_run
from src.common.seed import set_seed
from src.detect.yolo_utils import greedy_match
from src.pipeline.statistics import LEVEL, classification_intervals
from src.service import AnalysisService, read_image_file


def coverage_rows(frame: str, dets: list, objs: list[dict], classes: list[str], match_iou: float,
                  calibrated: bool = False) -> list[dict]:
    gb = np.array([o["box"] for o in objs], dtype=float).reshape(-1, 4)
    pb = np.array([d.box for d in dets], dtype=float).reshape(-1, 4)
    pairs = greedy_match(pb, np.array([d.det_conf for d in dets]), gb, match_iou)
    full = classification_intervals(dets, classes, calibrated=calibrated)["per_class"]
    matched = classification_intervals([dets[i] for i, _ in pairs], classes, calibrated=calibrated)["per_class"]
    rows = []
    for c in classes:
        truth = sum(o["target"] == c for o in objs)
        truth_m = sum(objs[j]["target"] == c for _, j in pairs)
        rows.append({"frame": frame, "class": c, "true": truth, "expected": full[c]["expected"],
                     "low": full[c]["low"], "high": full[c]["high"], "covered": full[c]["low"] <= truth <= full[c]["high"],
                     "true_matched": truth_m, "low_matched": matched[c]["low"], "high_matched": matched[c]["high"],
                     "covered_matched": matched[c]["low"] <= truth_m <= matched[c]["high"],
                     "width": full[c]["high"] - full[c]["low"]})
    return rows


def main() -> None:
    cfg = load_config()
    set_seed(cfg["project"]["seed"], cfg["project"]["deterministic"])
    ap = argparse.ArgumentParser()
    ap.add_argument("--probabilities", choices=("raw", "calibrated"), default="raw")
    cal = ap.parse_args().probabilities == "calibrated"
    suffix = "_calibrated" if cal else ""
    run_dir = start_run(cfg, f"check_count_intervals_val{suffix}")
    syn = resolve_path(cfg, "synthetic_data")
    gt = json.loads((syn / "gt_val.json").read_text())
    th = json.loads((resolve_path(cfg, "results") / "thresholds.json").read_text())
    svc = AnalysisService(cfg)
    classes = list(svc.pipe.organisms) + [svc.pipe.reject]
    rows = []
    for name, objs in sorted(gt.items()):
        img, _ = read_image_file(syn / "images" / "val" / f"{name}.jpg", svc.scfg)
        rows += coverage_rows(name, svc.pipe.run(img), objs, classes, th["det_match_iou"], cal)
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "coverage_rows.csv", index=False)

    def summary(d: pd.DataFrame) -> dict:
        return {"cells": len(d), "coverage_end_to_end": round(float(d["covered"].mean()), 4),
                "coverage_classification_only": round(float(d["covered_matched"].mean()), 4),
                "truth_below_interval": round(float((d["true"] < d["low"]).mean()), 4),
                "truth_above_interval": round(float((d["true"] > d["high"]).mean()), 4),
                "mean_width": round(float(d["width"].mean()), 3)}

    per_frame = df.groupby("frame").agg(all_end=("covered", "all"), all_cls=("covered_matched", "all"))
    out = {"level": LEVEL, "split": "synthetic VAL frames", "probabilities": "calibrated" if cal else "raw", "frames": int(df["frame"].nunique()),
           "classifier_run": svc.model_info["classifier"]["train_run"],
           "thresholds": {k: th[k] for k in ("det_conf", "nms_iou", "uncertainty_threshold")},
           "overall": summary(df),
           "frames_with_every_class_covered": {"end_to_end": round(float(per_frame["all_end"].mean()), 4),
                                               "classification_only": round(float(per_frame["all_cls"].mean()), 4)},
           "per_class": {c: summary(g) for c, g in df.groupby("class")},
           "run_dir": str(run_dir)}
    res = resolve_path(cfg, "results")
    (res / f"count_intervals_val{suffix}.json").write_text(json.dumps(out, indent=2))
    lines = [f"# Count interval coverage on synthetic VAL frames ({out['frames']} frames, nominal {LEVEL:.0%})", "",
             "| class | coverage end to end | coverage classification only | truth below | truth above | mean width |",
             "|---|---|---|---|---|---|"]
    for c, s in [("ALL", out["overall"])] + list(out["per_class"].items()):
        lines.append(f"| {c} | {s['coverage_end_to_end']:.3f} | {s['coverage_classification_only']:.3f} | "
                     f"{s['truth_below_interval']:.3f} | {s['truth_above_interval']:.3f} | {s['mean_width']:.2f} |")
    f = out["frames_with_every_class_covered"]
    lines += ["", f"Frames with every class inside its interval: end to end {f['end_to_end']:.3f}, "
                  f"classification only {f['classification_only']:.3f}.", ""]
    (res / f"count_intervals_val{suffix}.md").write_text("\n".join(lines), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(out, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
