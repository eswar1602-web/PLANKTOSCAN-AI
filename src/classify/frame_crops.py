"""Classifier crops re-cut from SYNTHETIC frames (closes the native-crop vs frame-crop domain gap).

The pipeline classifies crops cut from a frame around a detector box. A ZooLake crop pasted into a
frame loses its native background and is resampled, so the classifier sees a different input at
inference than in training. Here we cut the same kind of crop from the synthetic TRAIN / VAL frames
around the ground-truth box, with a random box jitter that mimics detector box error, using the
exact cropping function the pipeline uses (`pad_crop`). Per-split only: a split's frame crops come
from that split's frames, whose pasted crops all belong to that split (asserted).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.classify.data import load_rgb, prepare_crop

Box = tuple[float, float, float, float]


def pad_crop(img: np.ndarray, box: Box | np.ndarray, pad: float) -> np.ndarray:
    """THE crop cut used in training (frame crops) and inference: box grown by pad*side, clipped to the image."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in box)
    px, py = pad * (x1 - x0), pad * (y1 - y0)
    a0, b0 = int(max(0, x0 - px)), int(max(0, y0 - py))
    a1, b1 = int(min(w, x1 + px)), int(min(h, y1 + py))
    a0, b0 = min(a0, w - 1), min(b0, h - 1)
    return img[b0:max(b1, b0 + 1), a0:max(a1, a0 + 1)]


def jitter_box(box: Box, rng: np.random.Generator, jitter: float, img_size: tuple[int, int]) -> Box:
    """Move each edge independently by N(0, jitter * side); keep >= 2 px and inside the image."""
    h, w = img_size
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    d = rng.normal(0.0, jitter, 4) * np.array([bw, bh, bw, bh])
    nx0, ny0, nx1, ny1 = x0 + d[0], y0 + d[1], x1 + d[2], y1 + d[3]
    nx0, nx1 = np.clip(min(nx0, nx1 - 2), 0, w - 2), np.clip(max(nx1, nx0 + 2), 2, w)
    ny0, ny1 = np.clip(min(ny0, ny1 - 2), 0, h - 2), np.clip(max(ny1, ny0 + 2), 2, h)
    return float(nx0), float(ny0), float(nx1), float(ny1)


def sample_objects(gt: dict[str, list[dict]], n: int, rng: np.random.Generator) -> list[tuple[str, int]]:
    """n (frame id, object index) pairs drawn without replacement (all of them if n >= available)."""
    pool = [(fid, k) for fid in sorted(gt) for k in range(len(gt[fid]))]
    if n >= len(pool):
        return pool
    pick = np.sort(rng.choice(len(pool), size=n, replace=False))
    return [pool[i] for i in pick]


def frame_crop_key(split: str, n: int, fcfg: dict[str, Any], size: int, pcfg: dict | None,
                   grayscale: bool, seed: int, gt_digest: str) -> str:
    """Cache key: changes whenever any input that affects the crops changes."""
    blob = json.dumps({"split": split, "n": n, "f": fcfg, "size": size, "pre": pcfg, "gray": grayscale,
                       "seed": seed, "gt": gt_digest}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def build_frame_crops(
    synth_root: Path, split: str, n: int, split_of: dict[str, str], classes: list[str], size: int,
    pcfg: dict | None, fcfg: dict[str, Any], seed: int, grayscale: bool = False, cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """(uint8 prepared crops (n, size, size, 3), labels, metadata) cut from synthetic `split` frames."""
    gt_path = synth_root / f"gt_{split}.json"
    gt_text = gt_path.read_text()
    gt = json.loads(gt_text)
    for objs in gt.values():                                   # leakage guard: per-split only
        for o in objs:
            if split_of.get(o["path"]) != split or o.get("split") != split:
                raise AssertionError(f"{split} frame contains crop {o['path']} of split {split_of.get(o['path'])}")
    key = frame_crop_key(split, n, fcfg, size, pcfg, grayscale, seed, hashlib.sha1(gt_text.encode()).hexdigest())
    paths = None
    if cache_dir is not None:
        paths = cache_dir / f"frame_{split}_{key}.npy", cache_dir / f"frame_{split}_{key}.csv"
        if paths[0].exists() and paths[1].exists():
            meta = pd.read_csv(paths[1])
            return np.load(paths[0]), meta["label"].to_numpy(), meta
    rng = np.random.default_rng(seed)
    idx = {c: i for i, c in enumerate(classes)}
    picks = sample_objects(gt, n, rng)
    by_frame: dict[str, list[int]] = {}
    for fid, k in picks:
        by_frame.setdefault(fid, []).append(k)
    xs, rows = [], []
    for fid in tqdm(sorted(by_frame), desc=f"frame crops {split}", leave=False):
        img = load_rgb(synth_root / "images" / split / f"{fid}.jpg")
        for k in by_frame[fid]:
            o = gt[fid][k]
            box = jitter_box(tuple(o["box"]), rng, fcfg["box_jitter"], img.shape[:2])
            xs.append(prepare_crop(pad_crop(img, box, fcfg["padding"]), size, pcfg, grayscale))
            rows.append({"frame": fid, "obj": k, "path": o["path"], "target": o["target"],
                         "label": idx[o["target"]], "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3]})
    x = np.stack(xs)
    meta = pd.DataFrame(rows)
    if paths is not None:
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        np.save(paths[0], x)
        meta.to_csv(paths[1], index=False)
    return x, meta["label"].to_numpy(), meta
