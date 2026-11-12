"""Collect VAL results of all classifier and detector runs into results/ablation*.csv / .md.

Usage:  python -m scripts.collect_ablation
Classifier: runs/*/val_report.json (+ the run's saved config.yaml for init/grayscale).
Detector:   runs/*/yolo/results.csv (Ultralytics' per-epoch val metrics on SYNTHETIC val frames).
Uses VAL numbers only.
"""

from __future__ import annotations

import json

import pandas as pd
import yaml

from src.common.config import load_config, resolve_path


def collect() -> pd.DataFrame:
    """One row per classifier run."""
    cfg = load_config()
    rows = []
    for rep_path in sorted(resolve_path(cfg, "runs").glob("*/val_report.json")):
        rep = json.loads(rep_path.read_text())
        run_cfg = yaml.safe_load((rep_path.parent / "config.yaml").read_text(encoding="utf-8"))["classifier"]
        name = rep_path.parent.name.split("_", 2)[-1]
        row = {"run": rep_path.parent.name, "experiment": name,
               "init": "ImageNet" if run_cfg.get("pretrained") else "from scratch",
               "grayscale": bool(run_cfg.get("grayscale", False)),
               "preprocess": rep["use_preprocess"], "class_weighting": rep["class_weighting"],
               "epochs_budget": rep.get("epochs_budget"), "best_epoch": rep["best_epoch"],
               "val_accuracy": rep["accuracy"], "val_macro_f1": rep["macro_f1"],
               "val_macro_f1_organisms": rep["macro_f1_organisms"]}
        row.update({f"f1_{c}": v["f1"] for c, v in rep["per_class"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


def collect_detector() -> pd.DataFrame:
    """One row per YOLO run: epochs trained, best val mAP, and val mAP at a common epoch."""
    cfg = load_config()
    runs = sorted(resolve_path(cfg, "runs").glob("*/yolo/results.csv"))
    tables = {}
    for p in runs:
        r = pd.read_csv(p)
        r.columns = [c.strip() for c in r.columns]
        tables[p.parents[1].name] = r
    if not tables:
        return pd.DataFrame()
    common = min(int(r["epoch"].max()) for r in tables.values())
    rows = []
    for run, r in tables.items():
        args = yaml.safe_load((resolve_path(cfg, "runs") / run / "yolo" / "args.yaml").read_text(encoding="utf-8"))
        best = r.loc[r["metrics/mAP50-95(B)"].idxmax()]
        at = r[r["epoch"] == common].iloc[0]
        rows.append({"run": run, "init": "COCO (yolov8n.pt)" if str(args.get("model", "")).endswith(".pt")
                     else "from scratch", "time_cap_h": args.get("time"), "epochs_trained": int(r["epoch"].max()),
                     "stopped_early_by_user": run.endswith("_detector"),
                     "best_epoch": int(best["epoch"]), "best_val_mAP50": float(best["metrics/mAP50(B)"]),
                     "best_val_mAP50_95": float(best["metrics/mAP50-95(B)"]), "common_epoch": common,
                     "val_mAP50_at_common_epoch": float(at["metrics/mAP50(B)"]),
                     "val_mAP50_95_at_common_epoch": float(at["metrics/mAP50-95(B)"])})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    out = resolve_path(cfg, "results")
    df = collect()
    df.to_csv(out / "ablation.csv", index=False)
    show = df[["experiment", "init", "grayscale", "preprocess", "class_weighting", "epochs_budget", "best_epoch",
               "val_accuracy", "val_macro_f1", "val_macro_f1_organisms"]].round(4)
    det = collect_detector()
    det.to_csv(out / "ablation_detector.csv", index=False)
    text = "# Ablations (VAL only, single seed)\n\n## Classifier (EfficientNet-B0)\n\n" + show.to_markdown(index=False)
    if len(det):
        text += ("\n\n## Detector (YOLOv8n, SYNTHETIC val frames, Ultralytics default val: NMS IoU 0.7)\n\n"
                 + det.drop(columns=["run"]).round(4).to_markdown(index=False))
    (out / "ablation.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
