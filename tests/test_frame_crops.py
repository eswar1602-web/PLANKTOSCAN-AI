"""Tests for frame-crop extraction (v2 classifier training data)."""

import json

import cv2
import numpy as np
import pytest

from src.classify.frame_crops import build_frame_crops, jitter_box, pad_crop, sample_objects

CLASSES = ["copepod", "diatom", "reject"]


def _old_pipeline_crop(img, box, pad):
    """The v1 Pipeline.crops body, kept here to prove pad_crop did not change inference."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    px, py = pad * (x1 - x0), pad * (y1 - y0)
    a0, b0 = int(max(0, x0 - px)), int(max(0, y0 - py))
    a1, b1 = int(min(w, x1 + px)), int(min(h, y1 + py))
    return img[b0:max(b1, b0 + 1), a0:max(a1, a0 + 1)]


@pytest.mark.parametrize("box", [(10, 10, 50, 40), (0, 0, 20, 20), (90, 70, 100, 80), (33.7, 12.2, 61.9, 77.5)])
def test_pad_crop_matches_v1_pipeline(box) -> None:
    img = np.random.default_rng(0).integers(0, 255, (80, 100, 3), dtype=np.uint8)
    assert np.array_equal(pad_crop(img, box, 0.1), _old_pipeline_crop(img, box, 0.1))


def test_pad_crop_never_empty() -> None:
    img = np.zeros((50, 50, 3), np.uint8)
    for box in [(49.5, 49.5, 50, 50), (60, 60, 70, 70), (10, 10, 10, 10)]:
        c = pad_crop(img, box, 0.1)
        assert c.shape[0] >= 1 and c.shape[1] >= 1


def test_jitter_box_stays_inside_and_matches_std() -> None:
    rng = np.random.default_rng(1)
    box = (100.0, 100.0, 200.0, 150.0)
    out = np.array([jitter_box(box, rng, 0.06, (640, 640)) for _ in range(4000)])
    assert (out[:, 0] >= 0).all() and (out[:, 2] <= 640).all()
    assert (out[:, 2] - out[:, 0] >= 2).all() and (out[:, 3] - out[:, 1] >= 2).all()
    assert abs((out[:, 0] - 100).std() / 100 - 0.06) < 0.005      # x edges scale with width
    assert abs((out[:, 1] - 100).std() / 50 - 0.06) < 0.005       # y edges scale with height
    edge = jitter_box((0.0, 0.0, 5.0, 5.0), np.random.default_rng(2), 2.0, (20, 20))
    assert 0 <= edge[0] < edge[2] <= 20 and 0 <= edge[1] < edge[3] <= 20


def test_jitter_zero_is_identity() -> None:
    assert jitter_box((1.0, 2.0, 30.0, 40.0), np.random.default_rng(0), 0.0, (100, 100)) == (1.0, 2.0, 30.0, 40.0)


def test_sample_objects_without_replacement() -> None:
    gt = {"f0": [{}] * 3, "f1": [{}] * 4}
    s = sample_objects(gt, 5, np.random.default_rng(0))
    assert len(s) == 5 and len(set(s)) == 5
    assert sorted(sample_objects(gt, 99, np.random.default_rng(0))) == sorted(
        [("f0", i) for i in range(3)] + [("f1", i) for i in range(4)])


def _synth_root(tmp_path, split="train", crop_split="train"):
    (tmp_path / "images" / split).mkdir(parents=True)
    gt = {}
    for i in range(3):
        img = np.full((64, 64, 3), 8, np.uint8)
        img[10:30, 10:30] = 200
        cv2.imwrite(str(tmp_path / "images" / split / f"{split}_{i:05d}.jpg"), img)
        gt[f"{split}_{i:05d}"] = [{"box": [10, 10, 30, 30], "target": CLASSES[i % 3], "path": f"c{i}.jpg",
                                   "split": crop_split, "raw_class": "x"}]
    (tmp_path / f"gt_{split}.json").write_text(json.dumps(gt))
    return {f"c{i}.jpg": crop_split for i in range(3)}


def test_build_frame_crops_shapes_labels_and_cache(tmp_path) -> None:
    split_of = _synth_root(tmp_path)
    f = {"box_jitter": 0.06, "padding": 0.1}
    x, y, meta = build_frame_crops(tmp_path, "train", 3, split_of, CLASSES, 32, None, f, 0, cache_dir=tmp_path / "c")
    assert x.shape == (3, 32, 32, 3) and x.dtype == np.uint8
    assert sorted(y.tolist()) == [0, 1, 2] and set(meta["frame"]) == {"train_00000", "train_00001", "train_00002"}
    x2, y2, _ = build_frame_crops(tmp_path, "train", 3, split_of, CLASSES, 32, None, f, 0, cache_dir=tmp_path / "c")
    assert np.array_equal(x, x2) and np.array_equal(y, y2)
    assert len(list((tmp_path / "c").glob("*.npy"))) == 1


def test_build_frame_crops_refuses_cross_split_crops(tmp_path) -> None:
    split_of = _synth_root(tmp_path, split="train", crop_split="val")
    with pytest.raises(AssertionError):
        build_frame_crops(tmp_path, "train", 3, split_of, CLASSES, 32, None, {"box_jitter": 0, "padding": 0.1}, 0)


def test_frame_crop_loader_refuses_test() -> None:
    from src.classify.train import load_frame_crop_arrays
    from src.common.config import load_config
    with pytest.raises(AssertionError):
        load_frame_crop_arrays(load_config(), ("train", "test"), True)
