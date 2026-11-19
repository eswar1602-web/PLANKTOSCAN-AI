"""End-to-end: image -> [frame preprocess] -> detect -> crop -> [crop preprocess] -> classify -> status."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.classify.data import prepare_crop
from src.classify.frame_crops import pad_crop
from src.classify.model import load_checkpoint
from src.classify.train import predict_logits
from src.common.config import resolve_path
from src.common.timing import stage
from src.common.seed import get_device
from src.detect.yolo_utils import ultralytics_offline
from src.pipeline.counting import ORGANISM, REJECT, UNCERTAIN, Detection, assign_status
from src.preprocess.enhance import preprocess

STATUS_COLORS = {ORGANISM: (42, 120, 214), UNCERTAIN: (235, 104, 52), REJECT: (150, 150, 150)}  # RGB


def load_thresholds(cfg: dict[str, Any]) -> dict[str, float]:
    """Thresholds chosen on VAL data by the tuning scripts (results/thresholds.json)."""
    path = resolve_path(cfg, "results") / "thresholds.json"
    return json.loads(path.read_text()) if path.exists() else {}


class Pipeline:
    """Loads the detector + classifier once; `run` processes one RGB image."""

    def __init__(self, cfg: dict[str, Any], detector_path: Path | None = None, classifier_path: Path | None = None):
        ultralytics_offline()
        from ultralytics import YOLO
        self.cfg = cfg
        self.device = get_device(cfg["project"]["device"])
        models = resolve_path(cfg, "models")
        self.detector = YOLO(str(detector_path or models / "detector_best.pt"))
        self.classifier, self.meta = load_checkpoint(classifier_path or models / "classifier_best.pt", self.device)
        self.classes: list[str] = self.meta["classes"]
        self.reject = cfg["data"]["reject_class"]
        self.organisms = [c for c in self.classes if c != self.reject]
        th = load_thresholds(cfg)
        self.det_conf = th.get("det_conf", 0.25)
        self.nms_iou = th.get("nms_iou", 0.6)
        self.uncertainty = th.get("uncertainty_threshold", 0.5)

    def detect(self, img: np.ndarray, det_conf: float) -> tuple[np.ndarray, np.ndarray]:
        """Boxes (N, 4) x0 y0 x1 y1 and confidences (N,) from the YOLO detector."""
        frame = preprocess(img, self.cfg["preprocess"]) if self.cfg["detector"]["preprocess_frames"] else img
        res = self.detector.predict(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), conf=det_conf, iou=self.nms_iou,
                                    imgsz=self.cfg["detector"]["image_size"], verbose=False,
                                    device=0 if self.device == "cuda" else "cpu")[0]
        return res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()

    def crops(self, img: np.ndarray, boxes: np.ndarray) -> list[np.ndarray]:
        """Padded crops cut from the ORIGINAL image (classifier preprocessing is applied per crop)."""
        pad = self.cfg["pipeline"]["crop_padding"]
        return [pad_crop(img, b, pad) for b in boxes]

    def classify(self, crops: list[np.ndarray]) -> np.ndarray:
        """Softmax probabilities (N, n_classes); the same prepare_crop as in training."""
        if not crops:
            return np.zeros((0, len(self.classes)))
        x = np.stack([prepare_crop(c, self.meta["image_size"], self.meta["preprocess"], self.meta.get("grayscale", False)) for c in crops])
        bs = int(self.cfg["pipeline"].get("classifier_batch", 0))
        if bs > 0:   # fixed batch shape: cuDNN sets up each NEW input shape once (6-12 s on the RTX 3050),
            n = len(x)   # and the number of detections changes per image. Eval mode: padding never affects real crops.
            x = np.concatenate([x, np.zeros((-n % bs,) + x.shape[1:], x.dtype)])
            logits = predict_logits(self.classifier, x, self.device, batch=bs)[:n]
        else:
            logits = predict_logits(self.classifier, x, self.device)
        return torch.softmax(torch.from_numpy(logits), 1).numpy()

    def run(self, img: np.ndarray, det_conf: float | None = None, uncertainty: float | None = None,
            timings: dict[str, float] | None = None) -> list[Detection]:
        """Full pipeline on one uint8 RGB image. If `timings` is given, per-stage ms are added to it."""
        det_conf = self.det_conf if det_conf is None else det_conf
        uncertainty = self.uncertainty if uncertainty is None else uncertainty
        with stage(timings, "detection", self.device):
            boxes, confs = self.detect(img, det_conf)
            keep = np.array([(b[2] - b[0]) * (b[3] - b[1]) >= self.cfg["pipeline"]["min_box_area_px"]
                             for b in boxes], dtype=bool)
            boxes, confs = boxes[keep], confs[keep]
        with stage(timings, "classification", self.device):     # crop + crop preprocessing + classifier
            probs = self.classify(self.crops(img, boxes))
        dets = []
        for b, c, p in zip(boxes, confs, probs):
            k = int(p.argmax())
            label = self.classes[k]
            dets.append(Detection(box=tuple(float(v) for v in b), det_conf=float(c), label=label, prob=float(p[k]),
                                  status=assign_status(label, float(p[k]), uncertainty, self.reject),
                                  probs={cl: float(v) for cl, v in zip(self.classes, p)}))
        return dets


def annotate(img: np.ndarray, dets: list[Detection]) -> np.ndarray:
    """Draw boxes coloured by status with '#id label prob' text (OpenCV only, no font downloads).

    The ids are 1-based list positions, the same ids as in the reports' detection tables.
    """
    out = img.copy()
    scale = max(0.35, min(img.shape[:2]) / 1600)
    for i, d in enumerate(dets, start=1):
        x0, y0, x1, y1 = (int(v) for v in d.box)
        col = STATUS_COLORS[d.status]
        cv2.rectangle(out, (x0, y0), (x1, y1), col, max(1, int(2 * scale / 0.4)))
        text = f"#{i} {d.label} {d.prob:.2f}" if d.status == ORGANISM else f"#{i} {d.status}: {d.label}? {d.prob:.2f}"
        cv2.putText(out, text, (x0, max(10, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)
    return out
