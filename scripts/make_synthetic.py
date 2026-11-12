"""Generate synthetic detector frames per split (train/val/test use only their own crops).

Usage:  python -m scripts.make_synthetic
Writes data/synthetic/{images,labels}/{split}/, gt_{split}.json, data.yaml, and
results/synthetic_stats.csv + results/synthetic_examples.png (train frames only).
"""

from __future__ import annotations

import shutil

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.common.config import load_config, resolve_path
from src.common.experiment import start_run
from src.common.seed import set_seed
from src.detect.synth import generate_split

SPLIT_SEED_OFFSET = {"train": 0, "val": 1, "test": 2}


def main() -> None:
    cfg = load_config()
    seed = cfg["project"]["seed"]
    set_seed(seed)
    run_dir = start_run(cfg, "synthetic")
    scfg = cfg["synthetic"]
    root = resolve_path(cfg, "raw_data") / "zoolake" / cfg["data"]["zoolake"]["images_subdir"]
    out = resolve_path(cfg, "synthetic_data")
    if out.exists():
        shutil.rmtree(out)
    manifest = pd.read_csv(resolve_path(cfg, "processed_data") / "split_manifest.csv")
    stats = [generate_split(manifest, s, scfg["frames"][s], root, out, scfg, seed + SPLIT_SEED_OFFSET[s])
             for s in ("train", "val", "test")]
    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: object\n")
    df = pd.DataFrame(stats)
    df.to_csv(resolve_path(cfg, "results") / "synthetic_stats.csv", index=False)
    df.to_csv(run_dir / "synthetic_stats.csv", index=False)
    print(df.to_string(index=False))

    # example figure: 3 TRAIN frames with their labels drawn
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3), facecolor="#fcfcfb")
    for ax, i in zip(axes, (0, 1, 2)):
        name = f"train_{i:05d}"
        img = cv2.cvtColor(cv2.imread(str(out / "images" / "train" / f"{name}.jpg")), cv2.COLOR_BGR2RGB)
        s = scfg["image_size"]
        for line in (out / "labels" / "train" / f"{name}.txt").read_text().splitlines():
            _, cx, cy, w, h = (float(v) for v in line.split())
            x0, y0, x1, y1 = (int((cx - w / 2) * s), int((cy - h / 2) * s), int((cx + w / 2) * s), int((cy + h / 2) * s))
            cv2.rectangle(img, (x0, y0), (x1, y1), (42, 120, 214), 1)
        ax.imshow(img)
        ax.set_title(f"{name}: {len((out / 'labels' / 'train' / f'{name}.txt').read_text().splitlines())} labelled objects",
                     fontsize=10, color="#0b0b0b")
        ax.axis("off")
    fig.suptitle("Synthetic training frames with auto-generated YOLO boxes (blue)", color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(resolve_path(cfg, "results") / "synthetic_examples.png", dpi=110, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
