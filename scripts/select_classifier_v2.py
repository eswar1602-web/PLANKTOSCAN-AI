"""Choose between classifier v1 and a v2 run by macro F1 on the MIXED VAL set (original + frame crops).

v1 (models/classifier_v1.pt, the cls_pt_rgb model) is never overwritten. The winner is copied to
models/classifier_best.pt; the v2 checkpoint is also kept as models/classifier_v2.pt.
Writes results/classifier_selection_v2.json. VAL only.
Usage:  python -m scripts.select_classifier_v2 --run runs/<timestamp>_cls_v2_frames
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from src.classify.model import load_checkpoint
from src.classify.train import load_frame_crop_arrays, load_split_arrays, mixed_val_report, predict_logits
from src.common.config import load_config, resolve_path
from src.common.seed import get_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    a = ap.parse_args()
    cfg = load_config()
    device = get_device(cfg["project"]["device"])
    models = resolve_path(cfg, "models")
    v1_path, v2_path = models / "classifier_v1.pt", a.run.resolve() / "best.pt"
    _, meta2 = load_checkpoint(v2_path, device)
    assert meta2.get("frame_crops"), "the v2 run must be trained with --frame-crops 1"
    run_cfg = {**cfg, "classifier": {**cfg["classifier"], "frame_crops": meta2["frame_crops"]}}
    x, y, _ = load_split_arrays(cfg, ("val",), meta2["use_preprocess"], meta2["grayscale"])["val"]
    xf, yf, _ = load_frame_crop_arrays(run_cfg, ("val",), meta2["use_preprocess"], meta2["grayscale"])["val"]
    xm, ym = np.concatenate([x, xf]), np.concatenate([y, yf])
    res = {}
    for name, path in (("v1", v1_path), ("v2", v2_path)):
        model, meta = load_checkpoint(path, device)
        assert (meta["use_preprocess"], meta.get("grayscale", False), meta["image_size"]) == \
               (meta2["use_preprocess"], meta2["grayscale"], meta2["image_size"]), "same input transform"
        rep = mixed_val_report(ym, predict_logits(model, xm, device).argmax(1), len(y), meta["classes"],
                               cfg["data"]["reject_class"])
        res[name] = {"checkpoint": str(path.relative_to(resolve_path(cfg, "results").parent)), "run": meta["run"],
                     "mixed_val_macro_f1": rep["macro_f1"], "mixed_val_accuracy": rep["accuracy"],
                     "original_crops": rep["original_crops"], "frame_crops": rep["frame_crops"],
                     "per_class_f1": {c: v["f1"] for c, v in rep["per_class"].items()}}
    winner = max(res, key=lambda k: res[k]["mixed_val_macro_f1"])
    shutil.copy(v2_path, models / "classifier_v2.pt")
    shutil.copy(v1_path if winner == "v1" else v2_path, models / "classifier_best.pt")
    sel = {"criterion": "macro F1 on VAL original + frame crops", "selected": winner, **res}
    (resolve_path(cfg, "results") / "classifier_selection_v2.json").write_text(json.dumps(sel, indent=2))
    print(json.dumps({k: (v if k in ("criterion", "selected") else
                          {m: v[m] for m in ("mixed_val_macro_f1", "original_crops", "frame_crops")})
                      for k, v in sel.items()}, indent=2))


if __name__ == "__main__":
    main()
