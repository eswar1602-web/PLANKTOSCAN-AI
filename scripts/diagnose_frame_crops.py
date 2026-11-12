"""Diagnose crop-vs-frame domain shift on SYNTHETIC VAL frames only (no test data).

Classifies ground-truth-box crops cut from val frames at several paddings, and the original ZooLake
crops of the same objects, and reports uncertain fraction + accuracy for each. Writes
results/frame_crop_diagnostic.csv. Usage:  python -m scripts.diagnose_frame_crops
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.classify.data import load_rgb
from src.common.config import load_config, resolve_path
from src.pipeline.pipeline import Pipeline


def main() -> None:
    cfg = load_config()
    pipe = Pipeline(cfg)
    syn = resolve_path(cfg, "synthetic_data")
    root = resolve_path(cfg, "raw_data") / "zoolake" / cfg["data"]["zoolake"]["images_subdir"]
    gt = json.loads((syn / "gt_val.json").read_text())
    idx = {c: i for i, c in enumerate(pipe.classes)}
    rows, y, crops = [], [], {p: [] for p in (0.0, 0.1, 0.3, 0.6, 1.0)}
    orig = []
    for fid, objs in gt.items():
        img = load_rgb(syn / "images" / "val" / f"{fid}.jpg")
        for o in objs:
            y.append(idx[o["target"]])
            for p in crops:
                pipe.cfg["pipeline"]["crop_padding"] = p
                crops[p].append(pipe.crops(img, np.array([o["box"]]))[0])
            orig.append(load_rgb(root / o["path"]))
    y = np.array(y)
    t = pipe.uncertainty
    for name, cs in [*((f"frame crop, padding {p}", c) for p, c in crops.items()), ("original ZooLake crop", orig)]:
        pr = pipe.classify(cs)
        conf, pred = pr.max(1), pr.argmax(1)
        acc = conf >= t
        rows.append({"input": name, "n": len(y), "accuracy": float((pred == y).mean()),
                     "uncertain_fraction": float((~acc).mean()),
                     "accepted_accuracy": float((pred[acc] == y[acc]).mean()), "median_max_prob": float(np.median(conf))})
    df = pd.DataFrame(rows)
    df.to_csv(resolve_path(cfg, "results") / "frame_crop_diagnostic.csv", index=False)
    print(f"threshold {t:.2f} (VAL frames, ground-truth boxes)\n" + df.round(4).to_markdown(index=False))


if __name__ == "__main__":
    main()
