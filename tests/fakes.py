"""Test doubles: a fake Pipeline (no models, no GPU) and helpers to write image files."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.common.config import load_config
from src.pipeline.counting import Detection, assign_status
from src.pipeline.pipeline import load_thresholds

CLASSES = ["copepod", "diatom", "nauplius", "cladoceran", "rotifer", "ceratium", "dinobryon", "reject"]


class FakePipeline:
    """Mimics src.pipeline.pipeline.Pipeline: returns preset (label, prob) detections, applies the real status rule."""

    def __init__(self, preds: list[tuple[str, float]] | None = None, error: Exception | None = None, run_id: str | None = None):
        th = load_thresholds(load_config())
        self.det_conf, self.uncertainty, self.nms_iou = th["det_conf"], th["uncertainty_threshold"], th["nms_iou"]
        self.classes, self.reject = CLASSES, "reject"
        self.organisms = [c for c in CLASSES if c != "reject"]
        self.device = "cpu"
        self.meta = {"run": run_id or th.get("uncertainty_classifier_run"), "backbone": "fake"}
        self.preds = preds if preds is not None else [("copepod", 0.97), ("diatom", 0.95), ("rotifer", 0.5),
                                                      ("reject", 0.99), ("dinobryon", 0.3)]
        self.error = error
        self.calls: list[tuple[float, float]] = []

    def run(self, img, det_conf=None, uncertainty=None, timings=None):
        if self.error:
            raise self.error
        det_conf = self.det_conf if det_conf is None else det_conf
        uncertainty = self.uncertainty if uncertainty is None else uncertainty
        self.calls.append((det_conf, uncertainty))
        import time
        from src.common.timing import stage
        with stage(timings, "detection"):
            time.sleep(0.001)
        with stage(timings, "classification"):
            time.sleep(0.002)
        out = []
        for i, (label, p) in enumerate(self.preds):
            probs = {c: (p if c == label else (1 - p) / (len(CLASSES) - 1)) for c in CLASSES}
            out.append(Detection(box=(10.0 + 50 * i, 10.0, 50.0 + 50 * i, 60.0), det_conf=0.9, label=label, prob=p,
                                 status=assign_status(label, p, uncertainty, "reject"), probs=probs))
        return out

    def classify(self, crops):
        return np.zeros((len(crops), len(CLASSES)))


def fake_service(tmp_path: Path, preds=None, error=None, run_id=None, **cfg_overrides):
    """AnalysisService around a FakePipeline; small dummy weight files so model checksums are cheap."""
    from src.service import AnalysisService
    cfg = load_config()
    for k, v in cfg_overrides.items():
        cfg["service"][k] = v
    det, cls = tmp_path / "det.pt", tmp_path / "cls.pt"
    det.write_bytes(b"det"), cls.write_bytes(b"cls")
    return AnalysisService(cfg, pipeline=FakePipeline(preds, error, run_id), detector_path=det,
                           classifier_path=cls, warmup=False)


def write_image(path: Path, h: int = 120, w: int = 160, channels: int = 3, dtype=np.uint8) -> Path:
    """A dark frame with two bright blobs, written with OpenCV (format from the extension)."""
    shape = (h, w) if channels == 1 else (h, w, channels)
    img = np.full(shape, 8, dtype)
    hi = 200 if dtype == np.uint8 else 50000
    img[20:50, 20:60] = hi
    img[70:100, 90:140] = hi
    assert cv2.imwrite(str(path), img)
    return path
