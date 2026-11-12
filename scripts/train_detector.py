"""Train the single-class YOLOv8n detector on SYNTHETIC train frames (val frames for monitoring).

Usage:  python -m scripts.train_detector            (new run)
        python -m scripts.train_detector --resume runs/<run>/yolo/weights/last.pt
Built from yolov8n.yaml (no pretrained weights: none cached and downloads are not allowed).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.common.config import load_config, resolve_path
from src.common.experiment import start_run
from src.common.seed import set_seed
from src.detect.yolo_utils import ultralytics_offline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--pretrained", type=int, choices=[0, 1], default=0)
    ap.add_argument("--time-hours", type=float)
    ap.add_argument("--name", default="detector")
    a = ap.parse_args()
    ultralytics_offline()
    from ultralytics import YOLO

    cfg = load_config()
    dcfg = cfg["detector"]
    if a.resume:
        YOLO(str(a.resume)).train(resume=True)
        return
    set_seed(cfg["project"]["seed"])
    if a.pretrained:
        dcfg["model"] = str(resolve_path(cfg, "models") / "pretrained" / dcfg["pretrained_weights"])
    if a.time_hours:
        dcfg["time_hours"] = a.time_hours
    run_dir = start_run(cfg, a.name)
    model = YOLO(dcfg["model"])
    model.train(
        data=str(resolve_path(cfg, "synthetic_data") / "data.yaml"),
        imgsz=dcfg["image_size"], epochs=dcfg["epochs"], time=dcfg["time_hours"],
        patience=dcfg["patience"], batch=dcfg["batch_size"], workers=dcfg["workers"],
        amp=dcfg["amp"], seed=cfg["project"]["seed"], deterministic=False, single_cls=True,
        pretrained=bool(a.pretrained), plots=False, project=str(run_dir), name="yolo", exist_ok=True, verbose=False,
    )
    best = run_dir / "yolo" / "weights" / "best.pt"
    models = resolve_path(cfg, "models")
    models.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, models / "detector_best.pt")
    (models / "detector_run.txt").write_text(run_dir.name)
    print(f"best detector -> {models / 'detector_best.pt'}")


if __name__ == "__main__":
    main()
