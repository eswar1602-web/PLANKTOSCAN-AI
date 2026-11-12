"""Train the crop classifier (train split), selecting the checkpoint by val macro F1.

Usage examples (from the project root):
  python -m scripts.train_classifier --name cls_pre0_w0 --use-preprocess 0 --weighting none --epochs 25
  python -m scripts.train_classifier --resume runs/<run_dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.common.config import load_config
from src.classify.train import train_classifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="classifier")
    ap.add_argument("--use-preprocess", type=int, choices=[0, 1])
    ap.add_argument("--weighting", choices=["none", "inverse_sqrt_freq"])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--pretrained", type=int, choices=[0, 1])
    ap.add_argument("--grayscale", type=int, choices=[0, 1])
    ap.add_argument("--resume", type=Path, help="run directory to resume (uses its saved config)")
    a = ap.parse_args()
    if a.resume:
        cfg = yaml.safe_load(open(a.resume / "config.yaml", encoding="utf-8"))
    else:
        cfg = load_config()
        c = cfg["classifier"]
        if a.use_preprocess is not None:
            c["use_preprocess"] = bool(a.use_preprocess)
        if a.weighting:
            c["class_weighting"] = a.weighting
        if a.epochs:
            c["epochs"] = a.epochs
        if a.pretrained is not None:
            c["pretrained"] = bool(a.pretrained)
        if a.grayscale is not None:
            c["grayscale"] = bool(a.grayscale)
    train_classifier(cfg, a.name, resume=a.resume)


if __name__ == "__main__":
    main()
