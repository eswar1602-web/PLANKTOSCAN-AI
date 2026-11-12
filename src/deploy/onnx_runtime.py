"""Export the classifier and detector to ONNX and run them on CPU with OpenCV DNN.

Neither `onnx` nor `onnxruntime` is installed, and the project allows no further downloads. So:
- export uses PyTorch's TorchScript-based ONNX exporter. Its last step, `_add_onnxscript_fn`, only inserts
  custom onnx-script functions and needs the `onnx` package just to parse the bytes. Our models have no such
  functions, so the step is replaced by an identity (`_skip_onnxscript_step`). The written bytes are the same.
- inference uses `cv2.dnn` (OpenCV's built-in ONNX importer, CPU target). onnxruntime is used if installed.
YOLO decoding and NMS run in NumPy + cv2.dnn.NMSBoxes, mirroring Ultralytics (letterbox 114, class-agnostic).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _skip_onnxscript_step() -> None:
    """Make torch's onnx-script function insertion a no-op (no custom functions exist in these models)."""
    import torch.onnx._internal.torchscript_exporter.onnx_proto_utils as opu
    import torch.onnx._internal.torchscript_exporter.utils as tu
    opu._add_onnxscript_fn = lambda model_bytes, custom_opsets: model_bytes
    if hasattr(tu, "onnx_proto_utils"):
        tu.onnx_proto_utils._add_onnxscript_fn = opu._add_onnxscript_fn


def export_module(module, example, out: Path, opset: int = 17) -> Path:
    """Export a torch module (eval mode, CPU) to ONNX with a fixed input shape."""
    import torch
    _skip_onnxscript_step()
    out.parent.mkdir(parents=True, exist_ok=True)
    module = module.cpu().float().eval()
    with torch.no_grad():
        torch.onnx.export(module, example, str(out), dynamo=False, opset_version=opset,
                          input_names=["images"], output_names=["output"])
    return out


def export_yolo(pt_path: Path, out: Path, imgsz: int) -> Path:
    """Export the raw YOLOv8 network: output (1, 4 + n_classes, anchors) = cx, cy, w, h, scores."""
    import torch
    from src.detect.yolo_utils import ultralytics_offline
    ultralytics_offline()
    from ultralytics import YOLO
    net = YOLO(str(pt_path)).model.float().eval().fuse()
    for m in net.modules():
        if m.__class__.__name__ == "Detect":
            m.export, m.format = True, "onnx"
    return export_module(net, torch.zeros(1, 3, imgsz, imgsz), out)


class OnnxModel:
    """One ONNX model on CPU: onnxruntime if installed, else OpenCV DNN."""

    def __init__(self, path: Path, threads: int | None = None):
        self.path = Path(path)
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            if threads:
                so.intra_op_num_threads = threads
            self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
            self.backend = f"onnxruntime {ort.__version__} (CPU)"
        except ImportError:
            self.sess = None
            self.net = cv2.dnn.readNetFromONNX(str(path))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.backend = f"OpenCV DNN {cv2.__version__} (CPU)"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if self.sess is not None:
            return self.sess.run(None, {self.sess.get_inputs()[0].name: x})[0]
        self.net.setInput(x)
        return self.net.forward()


def letterbox_frame(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    """RGB uint8 -> (1, 3, size, size) float32 in [0, 1], resize ratio, (pad_x, pad_y). Ultralytics-style."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    px, py = (size - nw) / 2, (size - nh) / 2
    out = np.full((size, size, 3), 114, np.uint8)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else img
    top, left = int(round(py - 0.1)), int(round(px - 0.1))
    out[top:top + nh, left:left + nw] = resized
    return out.transpose(2, 0, 1)[None].astype(np.float32) / 255.0, r, (left, top)


def decode_yolo(out: np.ndarray, conf: float, iou: float, ratio: float, pad: tuple[float, float],
                shape: tuple[int, int], max_det: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Raw (1, 4+nc, N) output -> boxes (M, 4) x0 y0 x1 y1 in original pixels and scores (M,), after NMS."""
    p = out[0].T                                     # (N, 4 + nc)
    scores = p[:, 4:].max(1)
    keep = scores >= conf
    p, scores = p[keep], scores[keep]
    if not len(p):
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    xywh = np.stack([cx - w / 2, cy - h / 2, w, h], 1)
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), conf, iou, top_k=max_det)
    idx = np.array(idx, dtype=int).reshape(-1)[:max_det]
    b = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)[idx]
    b[:, [0, 2]] = (b[:, [0, 2]] - pad[0]) / ratio
    b[:, [1, 3]] = (b[:, [1, 3]] - pad[1]) / ratio
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, shape[1])
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, shape[0])
    return b.astype(np.float32), scores[idx].astype(np.float32)


def classifier_input(crops: np.ndarray) -> np.ndarray:
    """uint8 (N, H, W, 3) prepared crops -> ImageNet-normalised float32 (N, 3, H, W), NumPy only (as in training)."""
    from src.classify.augment import MEAN, STD
    mean = MEAN.cpu().numpy().reshape(1, 3, 1, 1)
    std = STD.cpu().numpy().reshape(1, 3, 1, 1)
    x = crops.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    return ((x - mean) / std).astype(np.float32)
