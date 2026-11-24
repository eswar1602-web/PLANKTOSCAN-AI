"""Write the model manifest: SHA-256, size, training run and role of every deployed weight file.

Usage:  python -m scripts.package_models [--check]
Writes models/MANIFEST.json (next to the weights, which are not in git) and results/model_manifest.json
(committed copy). --check only verifies the files against an existing models/MANIFEST.json (exit 1 on mismatch).
No weights are copied or modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from importlib import metadata

import torch

from src.common.config import load_config, resolve_path
from src.service import file_sha256

ROLES = {
    "detector_best.pt": "deployed detector (YOLOv8n, COCO-pretrained, fine-tuned on synthetic ZooLake frames)",
    "classifier_best.pt": "deployed classifier (= classifier_v2.pt)",
    "classifier_v2.pt": "classifier v2: EfficientNet-B0, original + synthetic-frame crops (Phase A)",
    "classifier_v1.pt": "classifier v1: EfficientNet-B0, original crops only (kept for comparison)",
    "confidence.npz": "temperature + unknown-flag statistics for the deployed classifier (scripts/fit_confidence.py)",
}


def describe(name: str, path) -> dict:
    entry = {"file": name, "role": ROLES[name], "bytes": path.stat().st_size, "sha256": file_sha256(path)}
    if name.startswith("classifier"):
        meta = torch.load(path, map_location="cpu", weights_only=False)["meta"]
        entry.update(train_run=meta["run"], backbone=meta["backbone"], classes=meta["classes"],
                     image_size=meta["image_size"], best_epoch=meta.get("epoch"),
                     val_macro_f1=round(meta["val_macro_f1"], 4), frame_crops=bool(meta.get("frame_crops")))
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    models = resolve_path(cfg, "models")
    if a.check:
        man = json.loads((models / "MANIFEST.json").read_text())
        bad = [e["file"] for e in man["files"] if file_sha256(models / e["file"]) != e["sha256"]]
        print("all checksums match" if not bad else f"MISMATCH: {', '.join(bad)}")
        return 1 if bad else 0
    files = [describe(n, models / n) for n in ROLES if (models / n).is_file()]
    det_run = (models / "detector_run.txt").read_text().strip() if (models / "detector_run.txt").exists() else None
    for e in files:
        if e["file"] == "detector_best.pt":
            e["train_run"] = det_run
    th = json.loads((resolve_path(cfg, "results") / "thresholds.json").read_text())
    man = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": files,
           "thresholds": {k: th[k] for k in ("det_conf", "nms_iou", "uncertainty_threshold",
                                             "uncertainty_threshold_v1", "uncertainty_classifier_run") if k in th},
           "packages": {p: metadata.version(p) for p in ("torch", "torchvision", "timm", "ultralytics",
                                                        "opencv-python", "numpy", "fastapi", "reportlab")}}
    text = json.dumps(man, indent=2)
    (models / "MANIFEST.json").write_text(text)
    (resolve_path(cfg, "results") / "model_manifest.json").write_text(text)
    for e in files:
        print(f"{e['file']:20s} {e['bytes']:>10,d} B  sha256 {e['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
