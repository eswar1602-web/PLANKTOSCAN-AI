"""Synthetic dark-field multi-organism frames with automatic YOLO labels.

Crops are pasted with a soft alpha mask derived from their own brightness (dark-field: the
organism is bright, the crop background dark), so no rectangular seam exists for the detector
to learn. Each split's frames are built ONLY from that split's crops (asserted).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.classify.data import load_rgb

Box = tuple[float, float, float, float]   # x0, y0, x1, y1 in pixels


def make_background(rng: np.random.Generator, size: int, bcfg: dict[str, Any]) -> np.ndarray:
    """Dark background with low-frequency uneven illumination (float32, H x W x 3)."""
    base = rng.uniform(*bcfg["base_level"])
    bg = np.full((size, size), base, np.float32)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    strength = rng.uniform(*bcfg["glow_strength"])
    for _ in range(int(rng.integers(1, 4))):
        cx, cy = rng.uniform(-0.2, 1.2, 2) * size
        sigma = rng.uniform(0.3, 0.8) * size
        bg += strength * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    tint = rng.uniform(0.9, 1.1, 3).astype(np.float32)
    return bg[..., None] * tint[None, None, :]


def crop_alpha(crop: np.ndarray, low: float, high: float, feather: float) -> np.ndarray:
    """Soft mask in [0, 1]: ramps from (border level + low) to (border level + high) gray levels."""
    gray = crop.astype(np.float32).mean(axis=2)
    w = max(1, min(2, min(gray.shape) // 2))
    border = np.concatenate([gray[:w].ravel(), gray[-w:].ravel(), gray[:, :w].ravel(), gray[:, -w:].ravel()])
    a = np.clip((gray - np.median(border) - low) / max(1e-6, high - low), 0, 1)
    if feather > 0:
        a = cv2.GaussianBlur(a, (0, 0), sigmaX=feather)
    a[0, :] = a[-1, :] = 0                      # crop edge is always fully transparent: no seams
    a[:, 0] = a[:, -1] = 0
    return a


def rotate_scale(img: np.ndarray, alpha: np.ndarray, angle: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate/scale crop and mask together on an expanded canvas (nothing is cut off)."""
    h, w = alpha.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(np.ceil(h * sin + w * cos)) + 2, int(np.ceil(h * cos + w * sin)) + 2
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    img_r = cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    a_r = cv2.warpAffine(alpha, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
    return img_r, a_r


def box_from_alpha(alpha: np.ndarray, thr: float = 0.25) -> Box | None:
    """Tight box (x0, y0, x1, y1) around mask pixels >= thr; None if the mask is empty."""
    ys, xs = np.nonzero(alpha >= thr)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def yolo_line(box: Box, size: int, cls: int = 0) -> str:
    """YOLO label line: class cx cy w h, all normalised to [0, 1]."""
    x0, y0, x1, y1 = box
    return f"{cls} {(x0 + x1) / 2 / size:.6f} {(y0 + y1) / 2 / size:.6f} {(x1 - x0) / size:.6f} {(y1 - y0) / size:.6f}"


def paste(frame: np.ndarray, img: np.ndarray, alpha: np.ndarray, x: int, y: int) -> None:
    """Alpha-blend img into frame (float32) with its top-left corner at (x, y), in place."""
    h, w = alpha.shape
    a = alpha[..., None]
    region = frame[y:y + h, x:x + w]
    frame[y:y + h, x:x + w] = a * img.astype(np.float32) + (1 - a) * region


def generate_frame(
    rng: np.random.Generator, crops: pd.DataFrame, images_root: Path, scfg: dict[str, Any]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """One frame: background + n alpha-blended crops (overlap-controlled) + noise + debris specks."""
    size = scfg["image_size"]
    frame = make_background(rng, size, scfg["background"])
    objects: list[dict[str, Any]] = []
    n = int(rng.integers(scfg["organisms_per_frame"][0], scfg["organisms_per_frame"][1] + 1))
    for _ in range(n):
        row = crops.iloc[int(rng.integers(len(crops)))]
        img = load_rgb(images_root / row["path"])
        ms = max(img.shape[:2])
        if ms > scfg["max_crop_side_px"]:
            f = scfg["max_crop_side_px"] / ms
            img = cv2.resize(img, (max(1, round(img.shape[1] * f)), max(1, round(img.shape[0] * f))),
                             interpolation=cv2.INTER_AREA)
        alpha = crop_alpha(img, scfg["alpha_low"], scfg["alpha_high"], scfg["mask_feather_px"])
        img_r, a_r = rotate_scale(img, alpha, rng.uniform(-scfg["rotation_deg"], scfg["rotation_deg"]),
                                  rng.uniform(*scfg["scale_range"]))
        local = box_from_alpha(a_r)
        if local is None or min(local[2] - local[0], local[3] - local[1]) < scfg["min_box_px"]:
            continue
        h, w = a_r.shape
        if h >= size or w >= size:
            continue
        for _attempt in range(30):
            x, y = int(rng.integers(0, size - w)), int(rng.integers(0, size - h))
            box = (local[0] + x, local[1] + y, local[2] + x, local[3] + y)
            if all(iou(box, o["box"]) <= scfg["max_overlap_iou"] for o in objects):
                paste(frame, img_r, a_r, x, y)
                objects.append({"box": box, "target": row["target"], "raw_class": row["raw_class"],
                                "path": row["path"], "split": row["split"]})
                break
    bcfg = scfg["background"]
    frame += rng.normal(0, rng.uniform(*bcfg["noise_std"]), frame.shape).astype(np.float32)
    for _ in range(int(rng.integers(bcfg["debris_specks"][0], bcfg["debris_specks"][1] + 1))):
        cx, cy = (int(v) for v in rng.integers(0, size, 2))
        cv2.circle(frame, (cx, cy), int(rng.integers(0, 2)), float(rng.uniform(10, 40)), -1)
    return np.clip(frame, 0, 255).astype(np.uint8), objects


def assert_same_split(objects: list[dict[str, Any]], split: str, split_of: dict[str, str]) -> None:
    """Leakage guard: every pasted crop must belong (per the split manifest) to this frame's split."""
    for o in objects:
        if split_of.get(o["path"]) != split:
            raise AssertionError(f"crop {o['path']} belongs to split {split_of.get(o['path'])}, not {split}")


def generate_split(
    manifest: pd.DataFrame, split: str, n_frames: int, images_root: Path, out_root: Path,
    scfg: dict[str, Any], seed: int,
) -> dict[str, Any]:
    """Write n_frames frames + YOLO labels + ground-truth JSON for one split; return stats."""
    crops = manifest[manifest["split"] == split].reset_index(drop=True)
    assert len(crops) and (crops["split"] == split).all()
    split_of = dict(zip(manifest["path"], manifest["split"]))
    rng = np.random.default_rng(seed)
    img_dir, lbl_dir = out_root / "images" / split, out_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    gt: dict[str, list] = {}
    for i in tqdm(range(n_frames), desc=f"frames {split}"):
        frame, objs = generate_frame(rng, crops, images_root, scfg)
        assert_same_split(objs, split, split_of)
        name = f"{split}_{i:05d}"
        cv2.imwrite(str(img_dir / f"{name}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        (lbl_dir / f"{name}.txt").write_text("\n".join(yolo_line(o["box"], scfg["image_size"]) for o in objs))
        gt[name] = [{k: (list(v) if k == "box" else v) for k, v in o.items()} for o in objs]
    (out_root / f"gt_{split}.json").write_text(json.dumps(gt))
    n_obj = sum(len(v) for v in gt.values())
    return {"split": split, "frames": n_frames, "objects": n_obj, "crops_available": len(crops),
            "unique_crops_used": len({o["path"] for v in gt.values() for o in v})}
