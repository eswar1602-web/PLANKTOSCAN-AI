"""PlanktoScan REST API. A thin HTTP layer over src/service.py (the same service the CLI uses).

    python -m api                       # serve on http://127.0.0.1:8000 (and the built frontend, if present)
    uvicorn api.main:create_app --factory --reload   # development (no app is built at import time)

Endpoints (JSON unless noted; every error is {"error": {"code": ..., "message": ...}}):
  GET    /api/health                              liveness + whether the models loaded
  GET    /api/info                                default thresholds, input limits, classes, model identity
  GET    /api/examples                            synthetic VAL frames offered as examples (if generated)
  GET    /api/examples/{name}                     an example image (image/jpeg)
  POST   /api/analyses                            multipart: file (or example=<name>), det_conf, uncertainty,
                                                   optional calibration: imaged_volume_ml | pixel_size_um + depth_um,
                                                   concentration_factor
  GET    /api/analyses?limit&offset&status        history, newest first
  GET    /api/analyses/{id}                       one analysis: history record + canonical result + links
  GET    /api/analyses/{id}/images/{kind}         original | enhanced | annotated
  GET    /api/analyses/{id}/reports/{fmt}         csv | json | html | pdf (download)
  DELETE /api/analyses/{id}                       remove one analysis and its stored files
  DELETE /api/analyses?confirm=true               clear the history
  GET    /api/analyses/{id}/corrections           this analysis's label corrections + reviewed counts
  PUT    /api/analyses/{id}/detections/{n}/correction   JSON {label, note}: set or replace one correction
  DELETE /api/analyses/{id}/detections/{n}/correction   remove it (answers with the updated view)
  GET    /api/corrections                         all corrections: totals and model agreement
  GET    /api/corrections/export                  zip: crops per corrected label + manifest (for retraining)
  POST   /api/reports/{fmt}                       stateless: multipart result (JSON) + annotated [+ enhanced] image
  POST   /api/review                              stateless: JSON {result, corrections}: validated items + reviewed counts

Storage (PLANKTOSCAN_STORAGE or create_app(storage=...)):
  server   (default) every analysis, its files and corrections are kept in the server's history.
  browser  for stateless hosting (several short-lived instances, no shared disk, e.g. Vercel): POST /api/analyses
           keeps nothing and answers with the result, the annotated and enhanced images (base64) and the review
           view; the browser stores them and builds reports and corrections with the two stateless endpoints.
           The history endpoints answer 409 browser_storage. Uploads are capped at BROWSER_MAX_FILE_MB.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.common.config import PROJECT_ROOT, load_config, resolve_path
from src.corrections import allowed_labels, export_corrections, reviewed_counts, validate_correction
from src.history import AnalysisHistory
from src.pipeline.statistics import Calibration
from src.service import (
    AnalysisError, AnalysisSettings, ImageTooLargeError, InvalidImageError, ModelLoadError, load_saved_result,
    new_analysis_id, safe_stem, save_outputs,
)

HTTP_STATUS = {"invalid_image": 400, "unsupported_format": 415, "image_too_small": 422, "image_too_large": 413,
               "model_load_error": 503, "inference_error": 500}
IMAGE_KINDS = ("original", "enhanced", "annotated")
REPORT_FORMATS = {"csv": "text/csv", "json": "application/json", "html": "text/html", "pdf": "application/pdf"}
N_EXAMPLES = 6
STORAGE_MODES = ("server", "browser")
BROWSER_MAX_FILE_MB = 4          # Vercel refuses request bodies above 4.5 MB (multipart overhead included)
RESPONSE_BUDGET = 4_300_000      # ... and response bodies above 4.5 MB: the browser-mode analysis stays below this
MAX_REPORT_IMAGE_BYTES = 8 * 1024 * 1024


class CorrectionIn(BaseModel):
    label: str = Field(min_length=1, max_length=64)
    note: str | None = Field(None, max_length=500)


class ReviewItemIn(CorrectionIn):
    detection_id: int


class ReviewIn(BaseModel):
    result: dict[str, Any]
    corrections: list[ReviewItemIn] = Field(default_factory=list, max_length=10_000)


def encode_image(rgb, fmt: str = ".png", quality: int = 90) -> bytes:
    import cv2
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == ".jpg" else []
    ok, enc = cv2.imencode(fmt, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), params)
    if not ok:
        raise OSError(f"could not encode an image as {fmt}")
    return enc.tobytes()


def encode_previews(images: dict[str, Any], budget: int) -> dict[str, dict[str, str]]:
    """{kind: {type, data (base64)}} within `budget` base64 bytes: PNG, else JPEG, else JPEG downscaled.

    Downscaling only ever applies to very large frames (the reports then use the smaller copy).
    """
    import cv2

    def build(fmt: str, scale: float) -> dict[str, tuple[str, bytes]]:
        out = {}
        for k, rgb in images.items():
            if scale < 1:
                rgb = cv2.resize(rgb, (max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale))),
                                 interpolation=cv2.INTER_AREA)
            out[k] = ("image/png" if fmt == ".png" else "image/jpeg", encode_image(rgb, fmt))
        return out

    def size(enc: dict[str, tuple[str, bytes]]) -> int:
        return sum(4 * ((len(b) + 2) // 3) for _, b in enc.values())

    enc = build(".png", 1.0)
    scale = 1.0
    if size(enc) > budget:
        enc = build(".jpg", 1.0)
    while size(enc) > budget and scale > 0.05:
        scale *= 0.75
        enc = build(".jpg", scale)
    return {k: {"type": t, "data": base64.b64encode(b).decode("ascii")} for k, (t, b) in enc.items()}


def review_view(result: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"items": items, "labels": allowed_labels(result), "reviewed_counts": reviewed_counts(result, items)}


def error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def create_app(service: Any | None = None, history_root: Path | None = None, cfg: dict | None = None,
               load_models: bool = True, frontend_dist: Path | None = None, storage: str | None = None) -> FastAPI:
    """Build the app. Tests pass a fake `service` and a temporary `history_root`."""
    cfg = cfg or load_config()
    storage = storage or os.environ.get("PLANKTOSCAN_STORAGE") or "server"
    if storage not in STORAGE_MODES:
        raise ValueError(f"storage must be one of {STORAGE_MODES}, not {storage!r}")
    browser = storage == "browser"
    max_file_mb = min(cfg["service"]["max_file_mb"], BROWSER_MAX_FILE_MB) if browser else cfg["service"]["max_file_mb"]
    root = Path(history_root or os.environ.get("PLANKTOSCAN_HISTORY_DIR") or resolve_path(cfg, "history"))
    history = AnalysisHistory(root / "history.sqlite", managed_root=root)
    example_dir = resolve_path(cfg, "synthetic_data") / "images" / "val"
    dist = Path(frontend_dist) if frontend_dist else PROJECT_ROOT / "frontend" / "dist"
    state: dict[str, Any] = {"service": service, "model_error": None}
    lock = threading.Lock()                      # one analysis at a time on the one GPU

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if state["service"] is None and load_models:
            try:
                from src.service import AnalysisService
                state["service"] = await run_in_threadpool(AnalysisService, cfg)
            except ModelLoadError as e:          # keep serving history; analysis answers 503
                state["model_error"] = str(e)
        yield

    app = FastAPI(title="PlanktoScan AI", version="2.0", lifespan=lifespan,
                  description="Detect, classify and count plankton in dark-field microscopy images.")

    # ---------------- errors: one envelope for everything
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, e: ApiError):
        return error(e.status, e.code, e.message)

    @app.exception_handler(AnalysisError)
    async def _analysis_error(_: Request, e: AnalysisError):
        return error(HTTP_STATUS.get(e.code, 400), e.code, str(e))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, e: RequestValidationError):
        first = e.errors()[0] if e.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", []) if p not in ("body", "query", "path"))
        return error(422, "bad_request", f"{where}: {first.get('msg', 'invalid request')}" if where else "Invalid request.")

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, e: StarletteHTTPException):
        return error(e.status_code, "not_found" if e.status_code == 404 else "http_error", str(e.detail))

    def server_history() -> None:
        if browser:
            raise ApiError(409, "browser_storage", "This server keeps no history: the app stores analyses in the browser.")

    def svc():
        if state["service"] is None:
            raise ApiError(503, "model_load_error", state["model_error"] or "The models are not loaded.")
        return state["service"]

    def examples() -> list[Path]:
        return sorted(example_dir.glob("*.jpg"))[:N_EXAMPLES] if example_dir.is_dir() else []

    def links(analysis_id: str, files: dict[str, str]) -> dict[str, dict[str, str]]:
        base = f"/api/analyses/{analysis_id}"
        return {"images": {k: f"{base}/images/{k}" for k in IMAGE_KINDS if k in files},
                "reports": {k: f"{base}/reports/{k}" for k in REPORT_FORMATS if k in files}}

    def record(analysis_id: str) -> dict:
        rec = history.get(analysis_id)
        if rec is None:
            raise ApiError(404, "not_found", f"No analysis with id {analysis_id}.")
        return rec

    def summary(rec: dict) -> dict:
        keys = ("analysis_id", "created_at", "filename", "status", "error_code", "error_message", "n_detections",
                "confirmed_total", "estimated_total", "uncertain", "rejected", "processing_ms", "det_conf",
                "uncertainty", "classifier_run", "n_corrections")
        return {**{k: rec.get(k) for k in keys}, "links": links(rec["analysis_id"], rec.get("files") or {})}

    # ---------------- service information
    @app.get("/api/health")
    def health() -> dict:
        s = state["service"]
        return {"status": "ok" if s is not None else "degraded", "models_loaded": s is not None,
                "error": state["model_error"], "device": s.model_info["device"] if s is not None else None,
                "storage": storage}

    @app.get("/api/info")
    def info() -> dict:
        s = svc()
        sc = cfg["service"]
        from src.pipeline.report import DOMAIN_WARNING
        return {"defaults": s.defaults, "formats": sc["formats"],
                "limits": {"max_file_mb": max_file_mb, "max_megapixels": sc["max_megapixels"],
                           "min_side_px": sc["min_side_px"]},
                "classes": {"organisms": list(s.pipe.organisms), "reject": s.pipe.reject},
                "models": s.model_info, "disclaimer": DOMAIN_WARNING, "storage": storage}

    @app.get("/api/examples")
    def list_examples() -> list[dict]:
        return [{"name": p.name, "url": f"/api/examples/{p.name}",
                 "note": "synthetic validation frame (generated, not a real microscope image)"} for p in examples()]

    @app.get("/api/examples/{name}")
    def example_image(name: str):
        p = next((p for p in examples() if p.name == name), None)
        if p is None:
            raise ApiError(404, "not_found", f"No example named {name}.")
        return FileResponse(p, media_type="image/jpeg")

    # ---------------- analyses
    @app.post("/api/analyses", status_code=201)
    async def create_analysis(file: UploadFile | None = File(None), example: str | None = Form(None),
                              det_conf: float | None = Form(None, ge=0, le=1),
                              uncertainty: float | None = Form(None, ge=0, le=1),
                              imaged_volume_ml: float | None = Form(None, gt=0),
                              pixel_size_um: float | None = Form(None, gt=0),
                              depth_um: float | None = Form(None, gt=0),
                              concentration_factor: float | None = Form(None, gt=0)) -> dict:
        s = svc()
        calibration = None
        if any(v is not None for v in (imaged_volume_ml, pixel_size_um, depth_um, concentration_factor)):
            try:
                calibration = Calibration(imaged_volume_ml, pixel_size_um, depth_um,
                                          1.0 if concentration_factor is None else concentration_factor)
            except ValueError as e:
                raise ApiError(422, "bad_request", f"Calibration: {e}.") from e
        max_bytes = int(max_file_mb * 1024 * 1024)
        if file is not None and file.filename:
            name = Path(file.filename).name[:200] or "upload"
            data = await file.read(max_bytes + 1)               # never read more than the limit
            if len(data) > max_bytes:
                e = ImageTooLargeError(f"{name}: the file is above the {max_file_mb:g} MB limit.")
                if not browser:
                    history.add_failure(new_analysis_id(), name, e.code, str(e))
                raise e
        elif example:
            p = next((p for p in examples() if p.name == example), None)
            if p is None:
                raise ApiError(404, "not_found", f"No example named {example}.")
            name, data = p.name, p.read_bytes()
        else:
            raise InvalidImageError("No image was sent. Attach a file in the 'file' field.")
        settings = AnalysisSettings(det_conf=det_conf, uncertainty=uncertainty, calibration=calibration)
        t0 = time.perf_counter()

        if browser:                                              # nothing is stored: everything goes back
            def work_browser():
                with lock:
                    res = s.analyze_bytes(data, name, settings)
                canonical = res.to_dict()
                budget = RESPONSE_BUDGET - len(json.dumps(canonical)) * 2 - 50_000   # result + review view
                images = encode_previews({"annotated": res.annotated, "enhanced": res.enhanced}, budget)
                return {"result": canonical, "images": images, "review": review_view(canonical, []),
                        "storage": storage}
            return await run_in_threadpool(work_browser)

        def work():
            with lock:
                res = s.analyze_bytes(data, name, settings)
            folder, files = save_outputs(res, root, original_bytes=data)
            history.add_result(res.to_dict(), folder, files)
            return res, files

        try:
            res, files = await run_in_threadpool(work)
        except AnalysisError as e:                               # the failure is part of the history too
            history.add_failure(new_analysis_id(), name, e.code, str(e), None, (time.perf_counter() - t0) * 1000)
            raise
        return {"result": res.to_dict(), "links": links(res.analysis_id, files)}

    @app.get("/api/analyses")
    def list_analyses(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
                      status: str | None = Query(None, pattern="^(ok|no_detections|failed)$")) -> dict:
        server_history()
        return {"items": [summary(r) for r in history.list(limit=limit, offset=offset, status=status)],
                "total": history.count(), "limit": limit, "offset": offset}

    @app.get("/api/analyses/{analysis_id}")
    def get_analysis(analysis_id: str) -> dict:
        server_history()
        rec = record(analysis_id)
        result = None
        if rec["status"] != "failed" and rec.get("output_dir") and (Path(rec["output_dir"]) / "report.json").is_file():
            result = load_saved_result(Path(rec["output_dir"]))
        return {**summary(rec), "warnings": rec.get("warnings") or [], "result": result,
                "files_missing": rec["status"] != "failed" and result is None}

    def stored_file(analysis_id: str, kind: str) -> Path:
        files = record(analysis_id).get("files") or {}
        p = Path(files[kind]) if kind in files else None
        if p is None or not p.is_file():
            raise ApiError(404, "not_found", f"Analysis {analysis_id} has no stored {kind} file.")
        return p

    @app.get("/api/analyses/{analysis_id}/images/{kind}")
    def get_image(analysis_id: str, kind: str):
        server_history()
        if kind not in IMAGE_KINDS:
            raise ApiError(404, "not_found", f"Unknown image kind {kind!r}; use one of {', '.join(IMAGE_KINDS)}.")
        return FileResponse(stored_file(analysis_id, kind))

    @app.get("/api/analyses/{analysis_id}/reports/{fmt}")
    def get_report(analysis_id: str, fmt: str):
        server_history()
        if fmt not in REPORT_FORMATS:
            raise ApiError(404, "not_found", f"Unknown report format {fmt!r}; use one of {', '.join(REPORT_FORMATS)}.")
        p = stored_file(analysis_id, fmt)
        stem = safe_stem(record(analysis_id)["filename"])
        return FileResponse(p, media_type=REPORT_FORMATS[fmt], filename=f"{stem}_{analysis_id}_planktoscan.{fmt}")

    @app.delete("/api/analyses/{analysis_id}", status_code=204)
    def delete_analysis(analysis_id: str) -> Response:
        server_history()
        if not history.delete(analysis_id):
            raise ApiError(404, "not_found", f"No analysis with id {analysis_id}.")
        return Response(status_code=204)

    @app.delete("/api/analyses")
    def clear_analyses(confirm: bool = Query(False)) -> dict:
        server_history()
        if not confirm:
            raise ApiError(400, "confirmation_required", "Add ?confirm=true to delete the whole history.")
        return {"deleted": history.clear()}

    # ---------------- label corrections
    def stored_result(analysis_id: str) -> dict:
        rec = record(analysis_id)
        out = Path(rec["output_dir"]) if rec.get("output_dir") else None
        if rec["status"] == "failed" or out is None or not (out / "report.json").is_file():
            raise ApiError(409, "no_result", f"Analysis {analysis_id} has no stored result to correct.")
        return load_saved_result(out)

    def correction_view(analysis_id: str, result: dict) -> dict:
        return review_view(result, history.corrections(analysis_id))

    @app.get("/api/analyses/{analysis_id}/corrections")
    def get_corrections(analysis_id: str) -> dict:
        server_history()
        return correction_view(analysis_id, stored_result(analysis_id))

    @app.put("/api/analyses/{analysis_id}/detections/{detection_id}/correction")
    def put_correction(analysis_id: str, detection_id: int, body: CorrectionIn) -> dict:
        server_history()
        result = stored_result(analysis_id)
        note = (body.note or "").strip() or None
        try:
            det = validate_correction(result, detection_id, body.label, note)
        except LookupError as e:
            raise ApiError(404, "not_found", str(e)) from e
        except ValueError as e:
            raise ApiError(422, "bad_request", str(e)) from e
        meta = result["metadata"]
        saved = history.set_correction(analysis_id, det, body.label, note, image=meta.get("image"),
                                       crop_padding=meta.get("crop_padding"))
        if saved is None:                                         # deleted while this request ran
            raise ApiError(404, "not_found", f"No analysis with id {analysis_id}.")
        return {"correction": saved, **correction_view(analysis_id, result)}

    @app.delete("/api/analyses/{analysis_id}/detections/{detection_id}/correction")
    def delete_correction(analysis_id: str, detection_id: int) -> dict:
        server_history()
        """Remove one correction; answers with the updated view (one round trip, like PUT)."""
        result = stored_result(analysis_id)
        if not history.delete_correction(analysis_id, detection_id):
            raise ApiError(404, "not_found", f"Detection {detection_id} of {analysis_id} has no correction.")
        return correction_view(analysis_id, result)

    @app.get("/api/corrections")
    def corrections_summary() -> dict:
        server_history()
        rows = history.all_corrections()
        by_label: dict[str, int] = {}
        for r in rows:
            by_label[r["corrected_label"]] = by_label.get(r["corrected_label"], 0) + 1
        return {"total": len(rows), "analyses": len({r["analysis_id"] for r in rows}), "by_label": by_label,
                "agreed_with_model": sum(r["corrected_label"] == r["predicted_label"] for r in rows)}

    @app.get("/api/corrections/export")
    def corrections_export():
        server_history()
        import shutil
        import tempfile
        rows = history.all_corrections()
        if not rows:
            raise ApiError(404, "not_found", "There are no corrections to export yet.")
        tmp = Path(tempfile.mkdtemp(prefix="planktoscan_corrections_"))
        try:
            export_corrections(rows, tmp / "corrections", cfg["pipeline"]["crop_padding"])
            archive = shutil.make_archive(str(tmp / "planktoscan_corrections"), "zip", tmp / "corrections")
        except Exception as e:                                    # never leave the temporary folder behind
            shutil.rmtree(tmp, ignore_errors=True)
            raise ApiError(500, "export_error", f"The export failed: {type(e).__name__}: {e}") from e
        return FileResponse(archive, media_type="application/zip", filename="planktoscan_corrections.zip",
                            background=BackgroundTask(shutil.rmtree, tmp, ignore_errors=True))

    # ---------------- stateless: reports and review computed from what the client sends
    def client_result(value: Any) -> dict:
        try:
            r = json.loads(value) if isinstance(value, str) else value
            if not isinstance(r, dict):
                raise ValueError
            r["metadata"]["classes"]["organisms"], r["detections"], r["analysis_id"], r["filename"]   # noqa: B018
        except (ValueError, KeyError, TypeError) as e:
            raise ApiError(422, "bad_request", "result: not a PlanktoScan analysis result.") from e
        return r

    async def client_image(f: UploadFile | None, what: str):
        if f is None:
            return None
        import cv2
        import numpy as np
        data = await f.read(MAX_REPORT_IMAGE_BYTES + 1)
        if len(data) > MAX_REPORT_IMAGE_BYTES:
            raise ApiError(413, "image_too_large", f"{what}: above {MAX_REPORT_IMAGE_BYTES // 2 ** 20} MB.")
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ApiError(400, "invalid_image", f"{what}: not a readable image.")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @app.post("/api/reports/{fmt}")
    async def build_report(fmt: str, result: str = Form(...), annotated: UploadFile | None = File(None),
                           enhanced: UploadFile | None = File(None)):
        if fmt not in REPORT_FORMATS:
            raise ApiError(404, "not_found", f"Unknown report format {fmt!r}; use one of {', '.join(REPORT_FORMATS)}.")
        r = client_result(result)
        ann = await client_image(annotated, "annotated")
        enh = await client_image(enhanced, "enhanced")
        if fmt in ("html", "pdf") and ann is None:
            raise ApiError(422, "bad_request", f"annotated: the {fmt} report needs the annotated image.")
        from src.pipeline.report import report_csv, report_html, report_json, report_pdf
        make = {"json": lambda: report_json(r), "csv": lambda: report_csv(r),
                "html": lambda: report_html(r, ann, enh), "pdf": lambda: report_pdf(r, ann)}[fmt]
        try:
            data = await run_in_threadpool(make)
        except (KeyError, TypeError, ValueError, AttributeError, IndexError) as e:
            raise ApiError(422, "bad_request",
                           f"result: the {fmt} report could not be built from it ({type(e).__name__}).") from e
        name = f"{safe_stem(r['filename'])}_{safe_stem(str(r['analysis_id']))}_planktoscan.{fmt}"
        return Response(data if isinstance(data, bytes) else data.encode("utf-8"), media_type=REPORT_FORMATS[fmt],
                        headers={"content-disposition": f'attachment; filename="{name}"'})

    @app.post("/api/review")
    def review(body: ReviewIn) -> dict:
        """Validate a whole set of corrections against a result; answers like the correction endpoints."""
        r = client_result(body.result)
        items: dict[int, dict] = {}
        try:
            for c in body.corrections:
                note = (c.note or "").strip() or None
                det = validate_correction(r, c.detection_id, c.label, note)
                items[det["id"]] = {"detection_id": det["id"], "corrected_label": c.label, "note": note,
                                    "predicted_label": det["label"], "predicted_status": det["status"],
                                    "prob": det.get("prob"), "box": det["box"]}
            return review_view(r, [items[k] for k in sorted(items)])
        except LookupError as e:
            raise ApiError(404, "not_found", str(e)) from e
        except ValueError as e:
            raise ApiError(422, "bad_request", str(e)) from e
        except (KeyError, TypeError) as e:
            raise ApiError(422, "bad_request", "result: not a PlanktoScan analysis result.") from e

    # ---------------- the built frontend (production); in development Vite serves it and proxies /api
    if (dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app

