"""PlanktoScan analysis service: the ONE entry point used by the REST API (api/) and the CLI.

    service = AnalysisService()                      # loads detector + classifier once
    result = service.analyze("frame.jpg")            # validate -> detect -> crop -> classify -> count
    result.to_dict()                                 # canonical JSON-able result (reports use only this)
    paths = save_outputs(result, out_root)           # unique folder: images + CSV/JSON/HTML/PDF

The service adds nothing to the model logic: detection, cropping, preprocessing, classification and
the uncertain/reject rules are those of src/pipeline (thresholds from results/thresholds.json, chosen
on VAL). It adds input validation, typed errors, timings, warnings and metadata.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.common.config import load_config, resolve_path
from src.common.timing import stage
from src.pipeline.counting import REJECT, UNCERTAIN, Detection, count_detections, estimated_counts
from src.pipeline.report import DOMAIN_WARNING as DOMAIN_NOTE
from src.pipeline.statistics import Calibration, classification_intervals, concentration_block
from src.preprocess.enhance import preprocess

STATUS_OK, STATUS_EMPTY, STATUS_FAILED = "ok", "no_detections", "failed"
NO_ORGANISMS = "No organisms detected."
# formats accepted by extension, and the decoder's own name for each (content must match a known image type)
_PIL_FORMATS = {"PNG", "JPEG", "MPO", "TIFF", "BMP"}


# ---------------------------------------------------------------- errors
class AnalysisError(Exception):
    """Base class: every expected failure of an analysis. `code` is stable, `str()` is user-facing."""
    code = "analysis_error"


class InvalidImageError(AnalysisError):
    """Missing, unreadable, empty or corrupted file."""
    code = "invalid_image"


class UnsupportedFormatError(AnalysisError):
    """File type not in service.formats (by extension or by content)."""
    code = "unsupported_format"


class ImageTooSmallError(AnalysisError):
    code = "image_too_small"


class ImageTooLargeError(AnalysisError):
    """File size or pixel count above the configured limits."""
    code = "image_too_large"


class ModelLoadError(AnalysisError):
    """Weights missing or not loadable."""
    code = "model_load_error"


class InferenceError(AnalysisError):
    """The models failed while processing a valid image (e.g. CUDA out of memory)."""
    code = "inference_error"


# ---------------------------------------------------------------- settings / result
@dataclass
class AnalysisSettings:
    """Per-request overrides. None = the VAL-chosen defaults from results/thresholds.json.

    calibration: how much sample the image shows; when given, the result also has concentrations.
    """
    det_conf: float | None = None
    uncertainty: float | None = None
    calibration: Calibration | None = None


@dataclass
class AnalysisResult:
    """Everything one analysis produced. Images are uint8 RGB arrays; everything else is JSON-able."""
    analysis_id: str
    created_at: str
    filename: str
    source_path: str | None
    status: str
    original: np.ndarray
    enhanced: np.ndarray
    annotated: np.ndarray
    detections: list[Detection]
    counts: dict[str, Any]
    estimated_counts: dict[str, Any]
    timings_ms: dict[str, float]
    count_intervals: dict[str, Any] = field(default_factory=dict)
    sensitivity: dict[str, Any] = field(default_factory=dict)
    concentration: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def message(self) -> str:
        return NO_ORGANISMS if self.status == STATUS_EMPTY else f"{self.counts['total_detections']} objects detected."

    def to_dict(self) -> dict[str, Any]:
        """Canonical result: the single source for the JSON, CSV, HTML and PDF reports and for the history."""
        return {
            "analysis_id": self.analysis_id, "created_at": self.created_at, "filename": self.filename,
            "source_path": self.source_path, "status": self.status, "message": self.message,
            "counts": self.counts, "estimated_counts": self.estimated_counts,
            "count_intervals": self.count_intervals, "concentration": self.concentration,
            "threshold_sensitivity": self.sensitivity,
            "detections": [detection_dict(i + 1, d) for i, d in enumerate(self.detections)],
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
            "warnings": list(self.warnings), "metadata": self.metadata,
        }


def detection_dict(i: int, d: Detection) -> dict[str, Any]:
    x0, y0, x1, y1 = d.box
    out = {"id": i, "status": d.status, "label": d.label, "prob": round(d.prob, 4), "det_conf": round(d.det_conf, 4),
           "box": {"x0": round(x0, 1), "y0": round(y0, 1), "x1": round(x1, 1), "y1": round(y1, 1)},
           "probs": {k: round(v, 4) for k, v in d.probs.items()},
           "calibrated_prob": None if d.calibrated_prob is None else round(d.calibrated_prob, 4),
           "unknown": bool(d.unknown), "ood_score": None if d.ood_score is None else round(d.ood_score, 1)}
    if d.calibrated_probs:
        out["calibrated_probs"] = {k: round(v, 4) for k, v in d.calibrated_probs.items()}
    return out


def threshold_sensitivity(dets: list[Detection], organisms: list[str], reject: str, threshold: float,
                          delta: float) -> dict[str, Any]:
    """Confirmed counts at threshold - delta, threshold and threshold + delta (same detections, same labels).

    Shows how much one image's counts depend on the exact uncertainty threshold, which for v2 sits next to a
    confidence ceiling. 'borderline' = detections whose probability lies within +-delta of the threshold.
    """
    from src.pipeline.counting import assign_status
    levels = {}
    for name, t in (("lower", threshold - delta), ("chosen", threshold), ("higher", threshold + delta)):
        t = min(max(t, 0.0), 1.0)
        per = {c: 0 for c in organisms}
        for d in dets:
            if assign_status(d.label, d.prob, t, reject) == "organism":
                per[d.label] += 1
        levels[name] = {"threshold": round(t, 4), "confirmed_total": sum(per.values()), "confirmed": per}
    return {"delta": delta, **levels, "borderline": sum(abs(d.prob - threshold) <= delta for d in dets)}


def structured_counts(dets: list[Detection], organisms: list[str], reject: str) -> tuple[dict, dict]:
    """(confirmed counts, estimated counts) in the canonical nested form, both checked against the box list."""
    s, e = count_detections(dets, organisms), estimated_counts(dets, organisms, reject)
    counts = {"confirmed": {c: s[c] for c in organisms}, "confirmed_total": sum(s[c] for c in organisms),
              "uncertain": s[UNCERTAIN], "rejected": s[REJECT], "total_detections": len(dets),
              "possibly_unknown": sum(d.unknown for d in dets),
              "possibly_unknown_confirmed": sum(d.unknown and d.status == "organism" for d in dets)}
    est = {"per_class": {c: e[c] for c in organisms}, "organisms_total": sum(e[c] for c in organisms),
           "reject": e[reject], "from_uncertain": e["from_uncertain"], "total": e["total"]}
    assert counts["confirmed_total"] + counts["uncertain"] + counts["rejected"] == len(dets)
    assert est["organisms_total"] + est["reject"] == len(dets)
    return counts, est


# ---------------------------------------------------------------- validation
def validate_bytes(data: bytes, filename: str, scfg: dict[str, Any]) -> np.ndarray:
    """Check a file's bytes against the service limits and decode it to uint8 RGB. Raises AnalysisError."""
    from PIL import Image
    ext = Path(filename).suffix.lower()
    if ext not in scfg["formats"]:
        raise UnsupportedFormatError(f"{filename}: unsupported file type '{ext or '(none)'}'. "
                                     f"Supported: {', '.join(scfg['formats'])}.")
    if not data:
        raise InvalidImageError(f"{filename}: the file is empty.")
    if len(data) > scfg["max_file_mb"] * 1024 * 1024:
        raise ImageTooLargeError(f"{filename}: {len(data) / 2**20:.1f} MB is above the {scfg['max_file_mb']} MB limit.")
    try:                                             # header only: size known before the pixels are decoded
        with Image.open(io.BytesIO(data)) as im:
            fmt, (w, h) = im.format, im.size
    except Image.DecompressionBombError as e:
        raise ImageTooLargeError(f"{filename}: image dimensions are too large ({e}).") from e
    except Exception as e:   # UnidentifiedImageError, OSError, ... and Ultralytics' patched Image.open, which
        # tries an optional HEIF plugin (pi_heif) on failure and raises ModuleNotFoundError when it is absent
        raise InvalidImageError(f"{filename}: not a readable image (corrupted or wrong file type).") from e
    if fmt not in _PIL_FORMATS:
        raise UnsupportedFormatError(f"{filename}: content is {fmt}, which is not a supported image type.")
    if w * h > scfg["max_megapixels"] * 1e6:
        raise ImageTooLargeError(f"{filename}: {w}x{h} ({w * h / 1e6:.1f} MP) is above the "
                                 f"{scfg['max_megapixels']} MP limit.")
    if min(w, h) < scfg["min_side_px"]:
        raise ImageTooSmallError(f"{filename}: {w}x{h} px is too small (minimum side {scfg['min_side_px']} px).")
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)   # grey/16-bit/alpha -> 8-bit BGR
    if bgr is None or bgr.shape[0] != h or bgr.shape[1] != w:
        raise InvalidImageError(f"{filename}: the image data is corrupted or truncated.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_image_file(path: str | Path, scfg: dict[str, Any]) -> tuple[np.ndarray, bytes]:
    """(RGB image, raw bytes) for a path; the same checks as validate_bytes plus existence."""
    p = Path(path)
    if not p.is_file():
        raise InvalidImageError(f"{p.name or path}: file not found.")
    ext = p.suffix.lower()
    if ext not in scfg["formats"]:                   # before reading a possibly huge non-image file
        raise UnsupportedFormatError(f"{p.name}: unsupported file type '{ext or '(none)'}'. "
                                     f"Supported: {', '.join(scfg['formats'])}.")
    if p.stat().st_size > scfg["max_file_mb"] * 1024 * 1024:
        raise ImageTooLargeError(f"{p.name}: {p.stat().st_size / 2**20:.1f} MB is above the "
                                 f"{scfg['max_file_mb']} MB limit.")
    try:
        data = p.read_bytes()
    except OSError as e:
        raise InvalidImageError(f"{p.name}: cannot read the file ({e.strerror}).") from e
    return validate_bytes(data, p.name, scfg), data


def enhance_preview(img: np.ndarray, pcfg: dict[str, Any], max_side: int) -> np.ndarray:
    """Display-only frame enhancement: the crop preprocessing chain with frame-level illumination correction.

    The detector runs on the original image and the classifier on preprocessed crops; this preview only
    shows what the enhancement does to the whole frame.
    """
    s = max_side / max(img.shape[:2])
    small = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA) \
        if s < 1 else img
    frame_cfg = {**pcfg, "illumination": {**pcfg["illumination"], "method": "gaussian_subtract"}}
    return preprocess(small, frame_cfg)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def new_analysis_id() -> str:
    """Sortable and unique: UTC time + 8 random hex digits."""
    return f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------- service
class AnalysisService:
    """Loads the models once; `analyze` / `analyze_bytes` / `analyze_array` process one image each."""

    def __init__(self, cfg: dict[str, Any] | None = None, pipeline: Any | None = None,
                 detector_path: Path | None = None, classifier_path: Path | None = None, warmup: bool | None = None):
        self.cfg = cfg or load_config()
        self.scfg = self.cfg["service"]
        models = resolve_path(self.cfg, "models")
        self.detector_path = Path(detector_path or models / "detector_best.pt")
        self.classifier_path = Path(classifier_path or models / "classifier_best.pt")
        t0 = time.perf_counter()
        if pipeline is None:
            missing = [str(p) for p in (self.detector_path, self.classifier_path) if not p.is_file()]
            if missing:
                raise ModelLoadError("Model weights not found: " + ", ".join(missing)
                                     + ". See README, section 'Model weights'.")
            try:
                from src.pipeline.pipeline import Pipeline
                pipeline = Pipeline(self.cfg, self.detector_path, self.classifier_path)
            except Exception as e:                   # corrupt/incompatible checkpoint, CUDA init failure, ...
                raise ModelLoadError(f"Could not load the models: {type(e).__name__}: {e}") from e
        self.pipe = pipeline
        self.load_ms = (time.perf_counter() - t0) * 1000
        self.model_info = self._model_info()
        self.load_warnings = self._threshold_check()
        if getattr(self.pipe, "confidence_warning", None):
            self.load_warnings.append(self.pipe.confidence_warning)
        if (self.scfg.get("warmup", True) if warmup is None else warmup):
            try:   # detector AND classifier: a blank frame has no detections, so classify a dummy crop too
                self.pipe.run(np.zeros((self.cfg["detector"]["image_size"],) * 2 + (3,), np.uint8))
                self.pipe.classify([np.full((72, 76, 3), 40, np.uint8)])
            except Exception as e:
                raise ModelLoadError(f"Model warm-up inference failed: {type(e).__name__}: {e}") from e

    def _model_info(self) -> dict[str, Any]:
        meta = getattr(self.pipe, "meta", {}) or {}
        info = {"device": getattr(self.pipe, "device", "unknown"),
                "classifier": {"file": self.classifier_path.name, "train_run": meta.get("run"),
                               "backbone": meta.get("backbone"), "classes": list(getattr(self.pipe, "classes", []))},
                "detector": {"file": self.detector_path.name}}
        for key, p in (("classifier", self.classifier_path), ("detector", self.detector_path)):
            if p.is_file():
                info[key]["sha256"] = file_sha256(p)
        return info

    @property
    def confidence_info(self) -> dict[str, Any] | None:
        cm = getattr(self.pipe, "confidence", None)
        if cm is None:
            return None
        return {"temperature": round(cm.temperature, 4), "ood_cutoff": round(cm.ood_threshold, 1),
                "ood_known_quantile": cm.info.get("ood_known_quantile"), "fitted": cm.info.get("fitted"),
                "note": "status uses the raw probability and the VAL-chosen threshold; calibrated probabilities "
                        "and the unknown flag are additional information"}

    def _threshold_check(self) -> list[str]:
        """Warn if results/thresholds.json was chosen for a different classifier than the one loaded."""
        from src.pipeline.pipeline import load_thresholds
        run = load_thresholds(self.cfg).get("uncertainty_classifier_run")
        loaded = self.model_info["classifier"]["train_run"]
        if run and loaded and run != loaded:
            return [f"The uncertainty threshold was chosen for classifier run {run}, but {loaded} is loaded."]
        return []

    @property
    def defaults(self) -> dict[str, float]:
        return {"det_conf": float(self.pipe.det_conf), "uncertainty": float(self.pipe.uncertainty),
                "nms_iou": float(self.pipe.nms_iou)}

    def analyze(self, image_path: str | Path, settings: AnalysisSettings | None = None) -> AnalysisResult:
        """Validate and analyse an image file. Raises an AnalysisError subclass on failure."""
        t0 = time.perf_counter()
        img, _ = read_image_file(image_path, self.scfg)
        load_ms = (time.perf_counter() - t0) * 1000
        return self.analyze_array(img, Path(image_path).name, settings, str(Path(image_path).resolve()),
                                  {"load_and_validate": load_ms})

    def analyze_bytes(self, data: bytes, filename: str, settings: AnalysisSettings | None = None) -> AnalysisResult:
        """Same as analyze, for an uploaded file's bytes (the REST API)."""
        t0 = time.perf_counter()
        img = validate_bytes(data, filename, self.scfg)
        return self.analyze_array(img, filename, settings, None, {"load_and_validate": (time.perf_counter() - t0) * 1000})

    def analyze_array(self, img: np.ndarray, filename: str, settings: AnalysisSettings | None = None,
                      source_path: str | None = None, timings: dict[str, float] | None = None) -> AnalysisResult:
        """Core analysis of a decoded uint8 RGB image (H, W, 3)."""
        from src.pipeline.pipeline import annotate
        settings = settings or AnalysisSettings()
        if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
            raise InvalidImageError(f"{filename}: expected an 8-bit RGB image, got {img.dtype} {img.shape}.")
        timings = dict(timings or {})
        t_start = time.perf_counter() - sum(timings.values()) / 1000
        d = self.defaults
        det_conf = d["det_conf"] if settings.det_conf is None else float(settings.det_conf)
        unc = d["uncertainty"] if settings.uncertainty is None else float(settings.uncertainty)
        for name, v in (("det_conf", det_conf), ("uncertainty", unc)):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {v}")
        device = getattr(self.pipe, "device", "cpu")
        try:
            dets = self.pipe.run(img, det_conf=det_conf, uncertainty=unc, timings=timings)
        except AnalysisError:
            raise
        except Exception as e:
            msg = "out of GPU memory" if "out of memory" in str(e).lower() else f"{type(e).__name__}: {e}"
            raise InferenceError(f"{filename}: model inference failed ({msg}).") from e
        with stage(timings, "counting", device):
            counts, est = structured_counts(dets, list(self.pipe.organisms), self.pipe.reject)
            calibrated = bool(dets) and all(d.calibrated_probs for d in dets)
            intervals = classification_intervals(dets, list(self.pipe.organisms) + [self.pipe.reject],
                                                 calibrated=calibrated)
            ucfg = self.cfg["uncertainty"]
            sens = threshold_sensitivity(dets, list(self.pipe.organisms), self.pipe.reject, unc,
                                         ucfg.get("sensitivity_delta", 0.01))
            conc = None if settings.calibration is None else                 concentration_block(counts["confirmed"], settings.calibration, img.shape[1], img.shape[0])
        with stage(timings, "annotation", device):
            annotated = annotate(img, dets)
        with stage(timings, "enhancement_preview", device):
            enhanced = enhance_preview(img, self.cfg["preprocess"], self.scfg["enhance_preview_max_side"])
        timings["total"] = (time.perf_counter() - t_start) * 1000

        h, w = img.shape[:2]
        warnings = list(self.load_warnings)
        status = STATUS_OK if dets else STATUS_EMPTY
        if not dets:
            warnings.append(NO_ORGANISMS + " Check that the image is a dark-field plankton image in focus; "
                                           "the detection threshold may also be too high.")
        if max(h, w) > self.scfg["large_side_warn_px"]:
            warnings.append(f"Large image ({w}x{h}): the detector resizes it to "
                            f"{self.cfg['detector']['image_size']} px, so small organisms may be missed.")
        if dets and counts["uncertain"] / len(dets) > self.scfg["high_uncertain_warn_fraction"]:
            warnings.append(f"{counts['uncertain']} of {len(dets)} detections are uncertain: the image may be "
                            "outside the training domain.")
        if (det_conf, unc) != (d["det_conf"], d["uncertainty"]):
            warnings.append(f"Non-default thresholds (detection {det_conf:.2f}, uncertainty {unc:.2f}; defaults "
                            f"{d['det_conf']:.2f} / {d['uncertainty']:.2f}): counts are not comparable "
                            "with the reported evaluation.")
        if counts["possibly_unknown"]:
            warnings.append(f"{counts['possibly_unknown']} of {len(dets)} detections look unlike every trained class "
                            f"(possibly unknown organisms; in the confirmed counts: "
                            f"{counts['possibly_unknown_confirmed']}). Check them by eye.")
        if dets and sens["borderline"] >= 2 and sens["borderline"] / len(dets) > ucfg.get("borderline_warn_fraction", 0.2):
            lo, hi = sens["lower"], sens["higher"]
            warnings.append(f"{sens['borderline']} detections are within {sens['delta']:.2f} of the uncertainty threshold: "
                            f"confirmed organisms would be {lo['confirmed_total']} at {lo['threshold']:.2f} and "
                            f"{hi['confirmed_total']} at {hi['threshold']:.2f}.")
        if conc is not None and counts["confirmed_total"] < 10:
            warnings.append(f"Only {counts['confirmed_total']} confirmed organisms: the concentration interval is wide. "
                            "Analyse several images of the same sample and pool them for a usable estimate.")
        meta = {"image": {"width": w, "height": h},
                "thresholds": {"det_conf": det_conf, "uncertainty": unc, "nms_iou": d["nms_iou"],
                               "defaults": {"det_conf": d["det_conf"], "uncertainty": d["uncertainty"]}},
                "classes": {"organisms": list(self.pipe.organisms), "reject": self.pipe.reject},
                "crop_padding": self.cfg["pipeline"]["crop_padding"],   # how the classified crops were cut
                "models": self.model_info,
                "confidence": self.confidence_info,
                "enhanced_preview": {"width": enhanced.shape[1], "height": enhanced.shape[0],
                                     "note": "display only; detection uses the original image"},
                "disclaimer": DOMAIN_NOTE}
        return AnalysisResult(
            analysis_id=new_analysis_id(), created_at=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            filename=filename, source_path=source_path, status=status, original=img, enhanced=enhanced,
            annotated=annotated, detections=dets, counts=counts, estimated_counts=est, timings_ms=timings,
            count_intervals=intervals, concentration=conc, sensitivity=sens,
            warnings=warnings, metadata=meta)


# ---------------------------------------------------------------- outputs
def safe_stem(filename: str) -> str:
    """File-system-safe short stem of a filename."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).stem).strip("._") or "image"
    return s[:60]


def save_outputs(result: AnalysisResult, out_root: Path, formats: tuple[str, ...] = ("csv", "json", "html", "pdf"),
                 original_bytes: bytes | None = None) -> tuple[Path, dict[str, str]]:
    """Write images + reports into a NEW folder out_root/<stem>_<analysis_id>/ (never overwrites).

    Returns (folder, {kind: path}). `original_bytes`, if given, is stored as the original image file
    (the app does this for uploads; the CLI keeps only the source path).
    """
    from src.pipeline.report import report_csv, report_html, report_json, report_pdf
    formats = tuple(dict.fromkeys(("json",) + tuple(formats)))   # report.json is always written (history reads it)
    unknown = set(formats) - {"csv", "json", "html", "pdf"}
    if unknown:
        raise ValueError(f"unknown report formats: {sorted(unknown)}")
    folder = Path(out_root) / f"{safe_stem(result.filename)}_{result.analysis_id}"
    folder.mkdir(parents=True, exist_ok=False)
    files: dict[str, str] = {}

    def write_png(name: str, rgb: np.ndarray) -> None:
        ok, enc = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise OSError(f"could not encode {name}")
        (folder / name).write_bytes(enc.tobytes())
        files[name.split(".")[0]] = str(folder / name)

    write_png("annotated.png", result.annotated)
    write_png("enhanced.png", result.enhanced)
    if original_bytes is not None:
        p = folder / f"original{Path(result.filename).suffix.lower()}"
        p.write_bytes(original_bytes)
        files["original"] = str(p)
    canonical = result.to_dict()
    writers = {"json": ("report.json", lambda: report_json(canonical)), "csv": ("report.csv", lambda: report_csv(canonical)),
               "html": ("report.html", lambda: report_html(canonical, result.annotated, result.enhanced)),
               "pdf": ("report.pdf", lambda: report_pdf(canonical, result.annotated))}
    for f in formats:
        name, make = writers[f]
        data = make()
        (folder / name).write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
        files[f] = str(folder / name)
    return folder, files


def load_saved_result(folder: Path) -> dict[str, Any]:
    """The canonical result stored by save_outputs (report.json)."""
    return json.loads((Path(folder) / "report.json").read_text(encoding="utf-8"))
