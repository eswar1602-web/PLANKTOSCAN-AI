"""Tests for synthetic frame generation and YOLO label writing."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.common.config import load_config
from src.detect.synth import (
    assert_same_split, box_from_alpha, crop_alpha, generate_split, iou, paste, rotate_scale, yolo_line,
)


def _crop(h: int = 40, w: int = 60, bg: int = 10) -> np.ndarray:
    img = np.full((h, w, 3), bg, np.uint8)
    img[10:30, 20:40] = 200                 # bright 20x20 organism
    return img


def test_alpha_is_zero_on_background_one_on_object_and_zero_at_edges() -> None:
    a = crop_alpha(_crop(), low=6, high=30, feather=0)
    assert a[0].max() == 0 and a[:, -1].max() == 0   # seam-free border
    assert a[5, 5] == 0 and a[20, 30] == 1


def test_box_from_alpha_is_tight() -> None:
    a = crop_alpha(_crop(), low=6, high=30, feather=0)
    assert box_from_alpha(a) == (20.0, 10.0, 40.0, 30.0)
    assert box_from_alpha(np.zeros((5, 5), np.float32)) is None


def test_rotation_keeps_whole_object() -> None:
    img, a = rotate_scale(_crop(), crop_alpha(_crop(), 6, 30, 0), angle=45, scale=1.0)
    x0, y0, x1, y1 = box_from_alpha(a)
    assert x0 > 0 and y0 > 0 and x1 < a.shape[1] and y1 < a.shape[0]
    side = 20 * np.sqrt(2)                   # rotated 20x20 square: diagonal-sized box
    assert abs((x1 - x0) - side) < 3 and abs((y1 - y0) - side) < 3


def test_yolo_line_normalised() -> None:
    assert yolo_line((100, 200, 164, 232), 640) == "0 0.206250 0.337500 0.100000 0.050000"


def test_iou() -> None:
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9


def test_paste_blends_without_touching_outside_mask() -> None:
    frame = np.full((80, 80, 3), 7.0, np.float32)
    img, a = _crop(), crop_alpha(_crop(), 6, 30, 0)
    paste(frame, img, a, 5, 5)
    assert frame[5 + 20, 5 + 30].tolist() == [200, 200, 200]   # inside the organism
    assert frame[5, 5].tolist() == [7, 7, 7]                   # crop background: frame untouched


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> tuple[pd.DataFrame, Path]:
    rows = []
    for split in ("train", "val"):
        for i in range(3):
            p = f"{split}_{i}.png"
            cv2.imwrite(str(tmp_path / p), _crop())
            rows.append({"path": p, "split": split, "target": "copepod", "raw_class": "cyclops"})
    return pd.DataFrame(rows), tmp_path


def test_generate_split_labels_match_ground_truth_and_split(tiny_dataset, tmp_path: Path) -> None:
    manifest, root = tiny_dataset
    scfg = load_config()["synthetic"]
    out = tmp_path / "synth"
    stats = generate_split(manifest, "val", 3, root, out, scfg, seed=0)
    gt = json.loads((out / "gt_val.json").read_text())
    assert stats["frames"] == 3 and len(gt) == 3
    for name, objs in gt.items():
        lines = (out / "labels" / "val" / f"{name}.txt").read_text().splitlines()
        assert len(lines) == len(objs)                      # one label per pasted object
        assert all(o["path"].startswith("val_") and o["split"] == "val" for o in objs)
        for line in lines:
            c, *xywh = line.split()
            assert c == "0" and all(0 <= float(v) <= 1 for v in xywh)
        img = cv2.imread(str(out / "images" / "val" / f"{name}.jpg"))
        assert img.shape == (scfg["image_size"], scfg["image_size"], 3)


def test_split_guard_rejects_foreign_crops() -> None:
    split_of = {"a.png": "train", "b.png": "val"}
    assert_same_split([{"path": "b.png"}], "val", split_of)            # ok
    with pytest.raises(AssertionError, match="belongs to split train"):
        assert_same_split([{"path": "b.png"}, {"path": "a.png"}], "val", split_of)
    with pytest.raises(AssertionError):
        assert_same_split([{"path": "unknown.png"}], "val", split_of)
