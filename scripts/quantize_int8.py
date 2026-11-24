"""INT8 post-training quantization with what is installed (PyTorch FX, oneDNN backend), measured on CPU.

Usage:  python -m scripts.quantize_int8 [--calib 512] [--runs 100]
- Classifier: static INT8 (per-channel weights, activations calibrated on TRAIN crops), then compared
  with FP32 on ALL VAL crops (original + frame crops): accuracy, macro F1, agreement, size, CPU latency.
- Detector: the same FX route is attempted; YOLOv8's detection head cannot be traced by torch.fx, and the
  failure is recorded instead of a number.
No test data. No weights are changed: the INT8 classifier is written to models/int8/ as a separate
TorchScript file and is NOT used by the app. Writes results/int8_quantization.{json,md} and a run folder.

onnxruntime (the usual INT8 route for CPU / edge, with QDQ models) is not installed, and the project
allows no new downloads; OpenCV 5.0 DNN has no quantizer. See the README for that route.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"        # CPU only

import argparse
import copy
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from src.classify.augment import normalize, to_float
from src.classify.metrics import classification_metrics
from src.classify.model import load_checkpoint
from src.classify.train import load_frame_crop_arrays, load_split_arrays
from src.common.config import load_config, resolve_path
from src.common.experiment import start_run
from src.common.seed import set_seed


MAX_F1_DROP = 0.01      # written after the first measurement; INT8 fails every criterion by a wide margin
DECISION_RULE = ("deploy INT8 only if it is faster than FP32 at 1 thread, VAL macro F1 drops by at most "
                 f"{MAX_F1_DROP}, and the detector can be quantized too")


def to_input(x_u8: np.ndarray) -> torch.Tensor:
    return normalize(to_float(torch.from_numpy(x_u8)))


@torch.no_grad()
def logits_of(model, x_u8: np.ndarray, batch: int = 128) -> np.ndarray:
    return torch.cat([model(to_input(x_u8[i:i + batch])) for i in range(0, len(x_u8), batch)]).numpy()


@torch.no_grad()
def median_ms(model, inputs: list[torch.Tensor], warmup: int = 10) -> float:
    for x in inputs[:warmup]:
        model(x)
    t = []
    for x in inputs:
        t0 = time.perf_counter()
        model(x)
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t))


def quantize_fx(model, example: torch.Tensor, calib: list[torch.Tensor]):
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    prepared = prepare_fx(copy.deepcopy(model).eval(), get_default_qconfig_mapping("onednn"), (example,))
    with torch.no_grad():
        for x in calib:
            prepared(x)
    return convert_fx(prepared)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", type=int, default=512, help="TRAIN crops used to calibrate activation ranges")
    ap.add_argument("--runs", type=int, default=100, help="VAL crops timed (batch 1)")
    a = ap.parse_args()
    assert not torch.cuda.is_available()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    cfg = load_config()
    seed = cfg["project"]["seed"]
    set_seed(seed, cfg["project"]["deterministic"])
    torch.backends.quantized.engine = "onednn"
    run_dir = start_run(cfg, "quantize_int8")
    models, results = resolve_path(cfg, "models"), resolve_path(cfg, "results")
    model, meta = load_checkpoint(models / "classifier_best.pt", "cpu")
    classes, reject = meta["classes"], cfg["data"]["reject_class"]
    gray = meta.get("grayscale", False)
    run_cfg = {**cfg, "classifier": {**cfg["classifier"], "frame_crops": meta["frame_crops"]}}

    # calibration: a fixed random subset of TRAIN crops (never VAL, never test)
    xtr, _, _ = load_split_arrays(cfg, ("train",), meta["use_preprocess"], gray)["train"]
    idx = np.random.default_rng(seed).choice(len(xtr), size=min(a.calib, len(xtr)), replace=False)
    calib = [to_input(xtr[idx[i:i + 64]]) for i in range(0, len(idx), 64)]
    example = calib[0][:1]

    t0 = time.perf_counter()
    q = quantize_fx(model, example, calib)
    quant_s = time.perf_counter() - t0

    # accuracy on ALL of VAL (original + frame crops)
    xva, yva, _ = load_split_arrays(cfg, ("val",), meta["use_preprocess"], gray)["val"]
    xvf, yvf, _ = load_frame_crop_arrays(run_cfg, ("val",), meta["use_preprocess"], gray)["val"]
    x, y = np.concatenate([xva, xvf]), np.concatenate([yva, yvf])
    torch.set_num_threads(os.cpu_count())
    z32, z8 = logits_of(model, x), logits_of(q, x)
    p32, p8 = z32.argmax(1), z8.argmax(1)
    th = json.loads((results / "thresholds.json").read_text())["uncertainty_threshold"]
    conf32 = torch.softmax(torch.from_numpy(z32), 1).max(1).values.numpy()
    conf8 = torch.softmax(torch.from_numpy(z8), 1).max(1).values.numpy()
    m32 = classification_metrics(y, p32, classes, reject)
    m8 = classification_metrics(y, p8, classes, reject)

    # size and latency (batch 1, VAL crops)
    out_dir = models / "int8"
    out_dir.mkdir(exist_ok=True)
    traced = torch.jit.trace(q, example)
    q_path = out_dir / "classifier_int8.ts"
    torch.jit.save(traced, str(q_path))
    singles = [to_input(x[i:i + 1]) for i in range(min(a.runs, len(x)))]
    lat = {}
    for label, n in (("all_threads", os.cpu_count()), ("1_thread", 1)):
        torch.set_num_threads(n)
        lat[label] = {"threads": n, "fp32_ms": round(median_ms(model, singles), 2), "int8_ms": round(median_ms(q, singles), 2)}
    torch.set_num_threads(os.cpu_count())

    # detector: attempt the same route
    det = {"attempted": "torch.ao FX static INT8 (onednn) on the fused YOLOv8n graph"}
    try:
        from src.detect.yolo_utils import ultralytics_offline
        ultralytics_offline()
        from ultralytics import YOLO
        net = YOLO(str(models / "detector_best.pt")).model.float().eval().fuse()
        for m in net.modules():
            if m.__class__.__name__ == "Detect":
                m.export, m.format = True, "onnx"
        quantize_fx(net, torch.zeros(1, 3, cfg["detector"]["image_size"], cfg["detector"]["image_size"]), [])
        det["result"] = "quantized (unexpected): not measured"
    except Exception as e:
        det["result"] = f"failed: {type(e).__name__}: {str(e).splitlines()[0][:200]}"

    def mb(p: Path) -> float:
        return round(p.stat().st_size / 1e6, 2)

    fp32_size = mb(models / "classifier_best.pt")
    res = {"note": "CPU only; PyTorch FX static INT8, oneDNN backend; calibration on TRAIN crops; accuracy on all VAL "
                   "crops (original + frame crops); latency = median over VAL crops, batch 1",
           "torch": torch.__version__, "engine": torch.backends.quantized.engine, "calibration_crops": len(idx),
           "quantize_seconds": round(quant_s, 1), "val_crops": int(len(y)),
           "classifier": {
               "size_mb": {"fp32_checkpoint": fp32_size, "int8_torchscript": mb(q_path)},
               "val_accuracy": {"fp32": round(m32["accuracy"], 4), "int8": round(m8["accuracy"], 4)},
               "val_macro_f1": {"fp32": round(m32["macro_f1"], 4), "int8": round(m8["macro_f1"], 4)},
               "argmax_agreement": round(float((p32 == p8).mean()), 4),
               "status_agreement_at_threshold": round(float(((conf32 >= th) == (conf8 >= th)).mean()), 4),
               "uncertain_share": {"fp32": round(float((conf32 < th).mean()), 4), "int8": round(float((conf8 < th).mean()), 4)},
               "latency_ms": lat},
           "detector": det, "file": str(q_path), "run": run_dir.name,
           "decision_rule": DECISION_RULE}
    faster = lat["1_thread"]["int8_ms"] < lat["1_thread"]["fp32_ms"]
    f1_drop = m32["macro_f1"] - m8["macro_f1"]
    det_ok = det["result"].startswith("quantized")
    reasons = [r for ok, r in ((faster, f"INT8 is not faster at 1 thread ({lat['1_thread']['int8_ms']} vs "
                                         f"{lat['1_thread']['fp32_ms']} ms)"),
                               (f1_drop <= MAX_F1_DROP, f"VAL macro F1 drops by {f1_drop:.3f} (> {MAX_F1_DROP})"),
                               (det_ok, "the detector could not be quantized with the installed tools")) if not ok]
    res["deploy"] = not reasons
    res["decision"] = "deploy" if not reasons else "not deployed: " + "; ".join(reasons)
    for d in (results, run_dir):
        (d / "int8_quantization.json").write_text(json.dumps(res, indent=2))
    c = res["classifier"]
    md = "\n".join([
        f"# INT8 quantization (PyTorch FX, {res['engine']}, CPU), run `{run_dir.name}`", "",
        f"Calibration: {len(idx)} TRAIN crops. Evaluation: {len(y)} VAL crops (original + frame crops). No test data.", "",
        "| classifier (EfficientNet-B0) | FP32 | INT8 |", "|---|---:|---:|",
        f"| size (MB) | {c['size_mb']['fp32_checkpoint']} | {c['size_mb']['int8_torchscript']} |",
        f"| VAL accuracy | {c['val_accuracy']['fp32']} | {c['val_accuracy']['int8']} |",
        f"| VAL macro F1 | {c['val_macro_f1']['fp32']} | {c['val_macro_f1']['int8']} |",
        f"| VAL uncertain share at {th} | {c['uncertain_share']['fp32']} | {c['uncertain_share']['int8']} |",
        f"| latency per crop, {lat['all_threads']['threads']} threads (ms) | {lat['all_threads']['fp32_ms']} | {lat['all_threads']['int8_ms']} |",
        f"| latency per crop, 1 thread (ms) | {lat['1_thread']['fp32_ms']} | {lat['1_thread']['int8_ms']} |", "",
        f"Predicted class agreement INT8 vs FP32: {c['argmax_agreement']:.2%}; confirmed/uncertain status agreement: "
        f"{c['status_agreement_at_threshold']:.2%}.", "",
        f"Detector: {det['result']}", "",
        f"Rule (written after the first measurement; the result fails every part of it by a wide margin): "
        f"{DECISION_RULE}.", "", f"Decision: {res['decision']}.", "",
        "Likely cause of the loss and the slowdown (not tested): post-training per-tensor activation ranges fit "
        "EfficientNet's SiLU and squeeze-excitation layers poorly, and oneDNN converts between INT8 and float "
        "around every op it cannot run in INT8. Quantization-aware training would be needed, which is retraining.",
        ""])
    (results / "int8_quantization.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
