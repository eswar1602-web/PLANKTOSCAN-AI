"""ONNX deployment helpers: letterbox geometry, YOLO decoding/NMS, normalisation, export round trip."""

import numpy as np
import pytest
import torch

from src.classify.augment import normalize, to_float
from src.deploy.onnx_runtime import OnnxModel, classifier_input, decode_yolo, export_module, letterbox_frame


def test_letterbox_frame_geometry():
    img = np.zeros((100, 200, 3), np.uint8)
    img[40:60, 90:110] = 255
    x, r, (px, py) = letterbox_frame(img, 64)
    assert x.shape == (1, 3, 64, 64) and x.dtype == np.float32
    assert r == pytest.approx(0.32) and px == 0 and py == 16
    assert x[0, 0, 0, 0] == pytest.approx(114 / 255)       # grey padding, as in Ultralytics
    assert x[0, :, 32, 32].min() > 0.9                     # the white square lands in the centre


def _raw(boxes_cxcywh, scores):
    """Fake YOLO output (1, 5, N) for one class."""
    a = np.array(boxes_cxcywh, np.float32).T
    return np.concatenate([a, np.array(scores, np.float32)[None]], 0)[None]


def test_decode_yolo_threshold_nms_and_unletterbox():
    out = _raw([[50, 50, 20, 20], [51, 50, 20, 20], [150, 150, 10, 10], [10, 10, 4, 4]],
               [0.9, 0.8, 0.7, 0.1])
    boxes, scores = decode_yolo(out, conf=0.5, iou=0.5, ratio=2.0, pad=(10, 0), shape=(200, 200))
    assert len(boxes) == 2                                  # duplicate suppressed, low score dropped
    assert scores.tolist() == pytest.approx([0.9, 0.7])
    assert boxes[0].tolist() == pytest.approx([15, 20, 25, 30])   # ((40-10)/2, 40/2, (60-10)/2, 60/2)


def test_decode_yolo_empty():
    boxes, scores = decode_yolo(_raw([[5, 5, 2, 2]], [0.01]), 0.5, 0.5, 1.0, (0, 0), (10, 10))
    assert boxes.shape == (0, 4) and scores.shape == (0,)


def test_classifier_input_matches_training_normalisation():
    crops = np.random.default_rng(0).integers(0, 256, (2, 8, 8, 3), dtype=np.uint8)
    ref = normalize(to_float(torch.from_numpy(crops))).numpy()
    assert np.abs(classifier_input(crops) - ref).max() < 1e-6


def test_export_round_trip(tmp_path):
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.BatchNorm2d(4), torch.nn.SiLU(),
                              torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 3)).eval()
    x = torch.randn(1, 3, 16, 16)
    path = export_module(net, x, tmp_path / "tiny.onnx")
    y = OnnxModel(path)(x.numpy())
    with torch.no_grad():
        assert np.abs(y - net(x).numpy()).max() < 1e-4
