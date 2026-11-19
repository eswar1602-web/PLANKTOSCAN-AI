"""ONE-SHOT comparison of classifier v1 vs v2 on FRESH synthetic test frames + the test split crops.

Usage:  python -m scripts.evaluate_v2
Inputs (all fixed before this runs): models/classifier_v1.pt, models/classifier_v2.pt,
models/detector_best.pt, results/thresholds.json (v1 and v2 thresholds, both chosen on VAL),
data/synthetic_test_v2/ (new seed, never used before), data/processed/split_manifest.csv (test crops).
Writes results/v1_vs_v2.{json,md}, results/count_errors_test_v2_{v1,v2}_{strict,estimated}.csv.
Refuses to run a second time (the fresh test frames would otherwise become a tuning set).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from src.classify.data import load_rgb
from src.classify.metrics import classification_metrics
from src.classify.model import load_checkpoint
from src.classify.train import load_split_arrays, predict_logits
from src.common.config import load_config, resolve_path
from src.common.experiment import start_run
from src.common.seed import get_device, set_seed
from src.detect.yolo_utils import ultralytics_offline
from src.evaluation import frame_record, summarize_frames
from src.pipeline.counting import assign_status

SENSITIVITY_T = (0.85, 0.89, 0.90)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-run (only after deleting outputs on purpose)")
    ap.add_argument("--dry-run-val", action="store_true",
                    help="same code on synthetic VAL frames + VAL crops; writes only into its run folder")
    a = ap.parse_args()
    cfg = load_config()
    seed = cfg["project"]["seed"]
    set_seed(seed, cfg["project"]["deterministic"])
    results = resolve_path(cfg, "results")
    dry = a.dry_run_val
    split = "val" if dry else "test"
    if not dry and (results / "v1_vs_v2.json").exists() and not a.force:
        raise SystemExit("results/v1_vs_v2.json exists: this evaluation is run once.")
    run_dir = start_run(cfg, "evaluate_v2_dryrun_val" if dry else "evaluate_v2")
    out = run_dir if dry else results          # the dry run never writes to results/
    device = get_device(cfg["project"]["device"])
    th = json.loads((results / "thresholds.json").read_text())
    models = resolve_path(cfg, "models")
    reject = cfg["data"]["reject_class"]
    spec = {"v1": (models / "classifier_v1.pt", th["uncertainty_threshold_v1"]),
            "v2": (models / "classifier_v2.pt", th["uncertainty_threshold"])}
    _, m2 = load_checkpoint(spec["v2"][0], "cpu")
    assert th["uncertainty_classifier_run"] == m2["run"], "v2 threshold was chosen for a different checkpoint"

    # 1. classifier on the TEST split (original crops)
    res: dict = {"run": run_dir.name, "mode": "DRY RUN on VAL (not a result)" if dry else "fresh test",
                 "thresholds": {k: v[1] for k, v in spec.items()}}
    crop_probs: dict = {}
    for name, (path, t) in spec.items():
        model, meta = load_checkpoint(path, device)
        x, y, _ = load_split_arrays(cfg, (split,), meta["use_preprocess"], meta.get("grayscale", False))[split]
        pr = torch.softmax(torch.from_numpy(predict_logits(model, x, device)), 1).numpy()
        r = classification_metrics(y, pr.argmax(1), meta["classes"], reject)
        acc = pr.max(1) >= t
        crop_probs[name] = (pr, y)
        res[name] = {"checkpoint": path.name, "train_run": meta["run"], "uncertainty_threshold": t,
                     "test_crops": {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"],
                                    "macro_f1_organisms": r["macro_f1_organisms"],
                                    "uncertain_fraction": float(1 - acc.mean()),
                                    "per_class_f1": {c: v["f1"] for c, v in r["per_class"].items()}}}

    # 2. detector mAP on the fresh frames (same detector for both)
    ultralytics_offline()
    from ultralytics import YOLO
    from src.pipeline.pipeline import Pipeline
    syn = resolve_path(cfg, "synthetic_data" if dry else "synthetic_test_v2")
    det = YOLO(str(models / "detector_best.pt"))
    m = det.val(data=str(syn / "data.yaml"), split=split, iou=th["nms_iou"], conf=0.001,
                imgsz=cfg["detector"]["image_size"], batch=cfg["detector"]["batch_size"], plots=False,
                verbose=False, workers=0, project=str(run_dir), name="det_val")
    res["detector_fresh_test"] = {"mAP@0.5": float(m.box.map50), "mAP@0.5:0.95": float(m.box.map)}

    # 3. end-to-end on the fresh frames
    gt = json.loads((syn / f"gt_{split}.json").read_text())
    pipes = {k: Pipeline(cfg, classifier_path=p) for k, (p, _) in spec.items()}
    organisms = pipes["v2"].organisms
    recs: dict[str, list] = {k: [] for k in spec}
    raw_v2: list = []                                  # v2 detections, re-thresholded for the sensitivity table
    for fid, objs in gt.items():
        img = load_rgb(syn / "images" / split / f"{fid}.jpg")
        for k, (_, t) in spec.items():
            dets = pipes[k].run(img, uncertainty=t)
            recs[k].append(frame_record(dets, objs, organisms, reject, cfg["detector"]["match_iou"]))
            if k == "v2":
                raw_v2.append((dets, objs))
    # threshold sensitivity for v2: REPORT ONLY, no selection (the threshold stays the VAL-chosen one)
    sens = []
    pr2, y2 = crop_probs["v2"]
    for t in SENSITIVITY_T:
        rr = [frame_record([replace(d, status=assign_status(d.label, d.prob, t, reject)) for d in dets], objs,
                           organisms, reject, cfg["detector"]["match_iou"]) for dets, objs in raw_v2]
        s, _, _ = summarize_frames(rr, organisms, reject)
        sens.append({"threshold": t, "chosen on VAL": t == spec["v2"][1],
                     "matched_uncertain_pct": s["matched_uncertain_pct"],
                     "count MAE (strict)": s["strict_count_MAE"], "signed error (strict)": s["strict_count_signed_error"],
                     "confident accuracy (matched)": s["matched_confident_accuracy"],
                     f"{split} crops uncertain": float((pr2.max(1) < t).mean()),
                     f"{split} crops accepted acc": float((pr2.argmax(1) == y2)[pr2.max(1) >= t].mean())})
    res["v2_threshold_sensitivity"] = sens
    for k in spec:
        s, e_strict, e_est = summarize_frames(recs[k], organisms, reject)
        res[k]["end_to_end_fresh_test"] = s
        e_strict.round(4).to_csv(out / f"count_errors_test_v2_{k}_strict.csv")
        e_est.round(4).to_csv(out / f"count_errors_test_v2_{k}_estimated.csv")

    text = json.dumps(res, indent=2)
    (out / "v1_vs_v2.json").write_text(text)
    if not dry:
        (run_dir / "v1_vs_v2.json").write_text(text)
    rows = {}
    for k in spec:
        e, c = res[k]["end_to_end_fresh_test"], res[k]["test_crops"]
        rows[k] = {"uncertainty t (VAL)": res[k]["uncertainty_threshold"],
                   "matched_uncertain_pct": e["matched_uncertain_pct"],
                   "count MAE (strict)": e["strict_count_MAE"], "signed error (strict)": e["strict_count_signed_error"],
                   "count MAE (estimated)": e["estimated_count_MAE"],
                   "signed error (estimated)": e["estimated_count_signed_error"],
                   "detection F1": e["det_f1"], "confident accuracy (matched)": e["matched_confident_accuracy"],
                   f"classifier {split} macro F1": c["macro_f1"], f"classifier {split} acc": c["accuracy"],
                   f"classifier {split} uncertain": c["uncertain_fraction"]}
    tab = pd.DataFrame(rows).round(4)
    md = (f"<!-- generated by scripts/evaluate_v2.py (run {run_dir.name}) -->\n"
          + ("**DRY RUN on synthetic VAL frames and VAL crops: a code check, not a result.** " if dry else "")
          + f"{'VAL' if dry else 'Fresh SYNTHETIC test'} frames: {e['frames']} frames, {e['gt_objects']} objects, mean true organisms per "
          f"frame {e['mean_true_per_frame']:.2f}. Detector mAP@0.5 {res['detector_fresh_test']['mAP@0.5']:.4f}, "
          f"mAP@0.5:0.95 {res['detector_fresh_test']['mAP@0.5:0.95']:.4f} (same detector for v1 and v2).\n\n"
          + tab.to_markdown() + "\n\n"
          "v2 threshold sensitivity (report only; the threshold used above is the VAL-chosen one):\n\n"
          + pd.DataFrame(sens).round(4).to_markdown(index=False) + "\n")
    (out / "v1_vs_v2.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
