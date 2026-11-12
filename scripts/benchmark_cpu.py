"""CPU-only (no GPU) benchmark of the final models exported to ONNX.

Usage:  python -m scripts.benchmark_cpu [--runs 100]
- Exports models/classifier_best.pt and models/detector_best.pt to models/onnx/*.onnx.
- Checks ONNX vs PyTorch outputs (max abs difference) and ONNX vs Ultralytics detections on VAL frames.
- Latency, median over --runs runs (each run a different SYNTHETIC VAL frame / VAL crop, never test data),
  after 10 warm-up runs, at all CPU threads and at 1 thread (a rough proxy for a small embedded CPU):
    detector per frame   = letterbox + ONNX forward + decode/NMS
    classifier per crop  = forward only, and with prepare_crop (illumination, NL-means, CLAHE, letterbox)
    full pipeline/frame  = detector + every crop (prepare + classify, batch 1)
- Writes results/benchmark_cpu.json and .md and the README section between EMBEDDED markers.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"        # CPU only: hide the GPU before torch is imported

import argparse
import json
import platform
import re
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src.classify.data import load_rgb, prepare_crop
from src.classify.model import load_checkpoint
from src.common.config import load_config, resolve_path
from src.common.experiment import start_run
from src.deploy.onnx_runtime import (OnnxModel, classifier_input, decode_yolo, export_module, export_yolo,
                                     letterbox_frame)
from src.detect.yolo_utils import greedy_match, ultralytics_offline


def cpu_name() -> str:
    """Processor name (Windows CIM), falling back to platform.processor()."""
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return out or platform.processor()
    except Exception:
        return platform.processor()


def median_ms(fn, inputs: list, warmup: int = 10) -> float:
    """Median wall time in ms of fn(x), one call per input, after `warmup` untimed calls."""
    for x in inputs[:warmup]:
        fn(x)
    times = []
    for x in inputs:
        t0 = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def set_threads(n: int) -> None:
    cv2.setNumThreads(n)
    torch.set_num_threads(n)


def crop_box(f: np.ndarray, box, pad: float) -> np.ndarray:
    """Padded crop, same rule as Pipeline.crops."""
    h, w = f.shape[:2]
    x0, y0, x1, y1 = box
    px, py = pad * (x1 - x0), pad * (y1 - y0)
    a0, b0 = int(max(0, x0 - px)), int(max(0, y0 - py))
    a1, b1 = int(min(w, x1 + px)), int(min(h, y1 + py))
    return f[b0:max(b1, b0 + 1), a0:max(a1, a0 + 1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    a = ap.parse_args()
    assert not torch.cuda.is_available(), "GPU must be hidden for a CPU benchmark"
    cfg = load_config()
    run_dir = start_run(cfg, "benchmark_cpu")
    models, out = resolve_path(cfg, "models"), resolve_path(cfg, "results")
    onnx_dir = models / "onnx"
    th = json.loads((out / "thresholds.json").read_text())
    imgsz = cfg["detector"]["image_size"]
    pad = cfg["pipeline"]["crop_padding"]

    # ---- export ----
    clf, meta = load_checkpoint(models / "classifier_best.pt", "cpu")
    clf.eval()
    size, gray = meta["image_size"], meta.get("grayscale", False)
    cls_onnx = export_module(clf, torch.zeros(1, 3, size, size), onnx_dir / "classifier.onnx")
    det_onnx = export_yolo(models / "detector_best.pt", onnx_dir / "detector.onnx", imgsz)
    cls_m, det_m = OnnxModel(cls_onnx), OnnxModel(det_onnx)

    # ---- inputs: SYNTHETIC VAL frames and their VAL crops only ----
    syn = resolve_path(cfg, "synthetic_data")
    frame_paths = sorted((syn / "images" / "val").glob("*.jpg"))[: a.runs]
    frames = [load_rgb(p) for p in frame_paths]
    gt = json.loads((syn / "gt_val.json").read_text())
    crops_raw = [crop_box(f, gt[p.stem][0]["box"], pad) for p, f in zip(frame_paths, frames) if gt[p.stem]]
    crops_raw = crops_raw[: a.runs]
    crops_in = [classifier_input(prepare_crop(c, size, meta["preprocess"], gray)[None]) for c in crops_raw]

    # ---- parity ----
    with torch.no_grad():
        ref = clf(torch.from_numpy(np.concatenate(crops_in[:20]))).numpy()
    onnx_out = np.concatenate([cls_m(x) for x in crops_in[:20]])
    cls_diff = float(np.abs(ref - onnx_out).max())
    cls_agree = float((ref.argmax(1) == onnx_out.argmax(1)).mean())

    def detect(f):
        x, r, p = letterbox_frame(f, imgsz)
        return decode_yolo(det_m(x), th["det_conf"], th["nms_iou"], r, p, f.shape[:2])

    ultralytics_offline()
    from ultralytics import YOLO
    yolo = YOLO(str(models / "detector_best.pt"))
    n_ref = n_onnx = n_match = 0
    for f in frames[:20]:
        rb = yolo.predict(cv2.cvtColor(f, cv2.COLOR_RGB2BGR), conf=th["det_conf"], iou=th["nms_iou"], imgsz=imgsz,
                          device="cpu", verbose=False)[0].boxes.xyxy.numpy()
        ob, os_ = detect(f)
        n_ref, n_onnx = n_ref + len(rb), n_onnx + len(ob)
        n_match += len(greedy_match(ob, os_, rb, 0.9))
    det_agree = n_match / max(n_ref, 1)

    # ---- latency ----
    def prep_and_classify(c):
        return cls_m(classifier_input(prepare_crop(c, size, meta["preprocess"], gray)[None]))

    def full_frame(f):
        boxes, _ = detect(f)
        for b in boxes:
            prep_and_classify(crop_box(f, b, pad))

    def torch_classify(x):
        with torch.no_grad():
            return clf(torch.from_numpy(x))

    mean_dets = float(np.mean([len(detect(f)[0]) for f in frames]))
    lat = {}
    for label, n in (("all_threads", os.cpu_count()), ("1_thread", 1)):
        set_threads(n)
        lat[label] = {
            "threads": n,
            "detector_per_frame_ms": median_ms(detect, frames),
            "classifier_per_crop_forward_ms": median_ms(cls_m, crops_in),
            "crop_preprocess_only_ms": median_ms(lambda c: prepare_crop(c, size, meta["preprocess"], gray), crops_raw),
            "classifier_per_crop_with_preprocess_ms": median_ms(prep_and_classify, crops_raw),
            "full_pipeline_per_frame_ms": median_ms(full_frame, frames),
            "pytorch_classifier_per_crop_forward_ms": median_ms(torch_classify, crops_in),
        }
    set_threads(os.cpu_count())

    def mb(p) -> float:
        return round(Path(p).stat().st_size / 1e6, 2)

    res = {
        "note": "CPU only (GPU hidden), ONNX models, inputs from SYNTHETIC VAL frames/crops; median over runs",
        "cpu": cpu_name(), "logical_cpus": os.cpu_count(), "runtime": cls_m.backend, "torch": torch.__version__,
        "runs": len(frames), "crop_runs": len(crops_raw), "frame_size": list(frames[0].shape[:2]),
        "detector_input": imgsz, "classifier_input": size, "mean_detections_per_frame": mean_dets,
        "sizes_mb": {"classifier_pt": mb(models / "classifier_best.pt"), "classifier_onnx": mb(cls_onnx),
                     "detector_pt": mb(models / "detector_best.pt"), "detector_onnx": mb(det_onnx)},
        "params_m": {"classifier": round(sum(p.numel() for p in clf.parameters()) / 1e6, 2),
                     "detector": round(sum(p.numel() for p in yolo.model.parameters()) / 1e6, 2)},
        "parity": {"classifier_max_abs_logit_diff": cls_diff, "classifier_argmax_agreement": cls_agree,
                   "detector_boxes_ultralytics": n_ref, "detector_boxes_onnx": n_onnx,
                   "detector_match_rate_iou_0.9": det_agree},
        "latency_ms": lat, "run": run_dir.name,
    }
    for d in (out, run_dir):
        (d / "benchmark_cpu.json").write_text(json.dumps(res, indent=2))

    def row(k: str) -> str:
        return f"{lat['all_threads'][k]:.1f} | {lat['1_thread'][k]:.1f}"

    md = "\n".join([
        "<!-- generated by scripts/benchmark_cpu.py; do not edit by hand -->",
        f"_CPU only (GPU hidden): {res['cpu']}, {res['logical_cpus']} logical CPUs, runtime {res['runtime']}. "
        f"Median over {res['runs']} SYNTHETIC VAL frames ({res['frame_size'][0]}x{res['frame_size'][1]} px, mean "
        f"{mean_dets:.1f} detections per frame) and {res['crop_runs']} VAL crops, after 10 warm-up runs. "
        f"Run `{run_dir.name}`._",
        "",
        "| model | params (M) | .pt (MB) | ONNX fp32 (MB) |", "|---|---:|---:|---:|",
        f"| EfficientNet-B0 classifier ({size} px) | {res['params_m']['classifier']} | "
        f"{res['sizes_mb']['classifier_pt']} | {res['sizes_mb']['classifier_onnx']} |",
        f"| YOLOv8n detector ({imgsz} px) | {res['params_m']['detector']} | {res['sizes_mb']['detector_pt']} | "
        f"{res['sizes_mb']['detector_onnx']} |",
        "",
        f"| latency (ms, median) | {os.cpu_count()} threads | 1 thread |", "|---|---:|---:|",
        f"| detector per frame (letterbox + forward + NMS) | {row('detector_per_frame_ms')} |",
        f"| classifier per crop, forward only | {row('classifier_per_crop_forward_ms')} |",
        f"| crop preprocessing only (illumination, NL-means, CLAHE, letterbox) | {row('crop_preprocess_only_ms')} |",
        f"| classifier per crop, preprocessing + forward | {row('classifier_per_crop_with_preprocess_ms')} |",
        f"| **full pipeline per frame** (detector + all crops, batch 1) | {row('full_pipeline_per_frame_ms')} |",
        f"| reference: PyTorch CPU classifier forward per crop | {row('pytorch_classifier_per_crop_forward_ms')} |",
        "",
        f"ONNX parity: classifier max abs logit difference {cls_diff:.1e}, argmax agreement {cls_agree:.0%} "
        f"(20 crops); detector {n_onnx} ONNX vs {n_ref} Ultralytics boxes on 20 VAL frames, "
        f"{det_agree:.1%} of Ultralytics boxes matched at IoU >= 0.9.",
        "",
    ])
    (out / "benchmark_cpu.md").write_text(md, encoding="utf-8")
    readme = out.parent / "README.md"
    text = readme.read_text(encoding="utf-8")
    if "<!-- EMBEDDED:START -->" in text:
        text = re.sub(r"<!-- EMBEDDED:START -->.*<!-- EMBEDDED:END -->",
                      lambda _: "<!-- EMBEDDED:START -->\n" + md + "<!-- EMBEDDED:END -->", text, flags=re.S)
        readme.write_text(text, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
