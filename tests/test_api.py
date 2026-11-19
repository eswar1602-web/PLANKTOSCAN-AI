"""REST API (api/main.py) with a fake service and a temporary history: endpoints, errors, downloads, history."""

import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from tests.fakes import fake_service, write_image


@pytest.fixture
def client(tmp_path):
    app = create_app(service=fake_service(tmp_path), history_root=tmp_path / "hist", load_models=False,
                     frontend_dist=tmp_path / "no_dist")
    with TestClient(app) as c:
        yield c


def _png(h=120, w=160) -> bytes:
    img = np.full((h, w, 3), 8, np.uint8)
    img[20:50, 20:60] = 200
    return cv2.imencode(".png", img)[1].tobytes()


def _post(client, data=None, name="frame.png", **form):
    files = {"file": (name, data if data is not None else _png(), "image/png")}
    return client.post("/api/analyses", files=files, data={k: str(v) for k, v in form.items()})


def test_health_and_info(client) -> None:
    h = client.get("/api/health").json()
    assert h == {"status": "ok", "models_loaded": True, "error": None, "device": "cpu"}
    info = client.get("/api/info").json()
    assert set(info["defaults"]) == {"det_conf", "uncertainty", "nms_iou"}
    assert ".png" in info["formats"] and info["limits"]["max_file_mb"] > 0
    assert info["classes"]["reject"] == "reject" and len(info["classes"]["organisms"]) == 7
    assert "NOT validated" in info["disclaimer"]


def test_analyse_upload_returns_canonical_result_and_links(client) -> None:
    r = _post(client)
    assert r.status_code == 201
    body = r.json()
    res = body["result"]
    assert res["status"] == "ok" and res["filename"] == "frame.png"
    c, e = res["counts"], res["estimated_counts"]
    assert (c["confirmed_total"], c["uncertain"], c["rejected"], e["organisms_total"]) == (2, 2, 1, 4)
    assert len(res["detections"]) == 5 and res["timings_ms"]["total"] > 0
    aid = res["analysis_id"]
    assert set(body["links"]["images"]) == {"original", "enhanced", "annotated"}
    assert set(body["links"]["reports"]) == {"csv", "json", "html", "pdf"}
    for url in body["links"]["images"].values():
        img = client.get(url)
        assert img.status_code == 200 and img.headers["content-type"].startswith("image/")
    rep = client.get(f"/api/analyses/{aid}/reports/json")
    assert rep.status_code == 200 and json.loads(rep.content) == res               # download == response
    pdf = client.get(f"/api/analyses/{aid}/reports/pdf")
    assert pdf.content[:4] == b"%PDF" and "attachment" in pdf.headers["content-disposition"]
    assert "frame_" in pdf.headers["content-disposition"] and ".pdf" in pdf.headers["content-disposition"]
    csv = client.get(f"/api/analyses/{aid}/reports/csv").text
    assert "# counts" in csv and "ALL organisms,2,4" in csv


def test_thresholds_are_passed_and_validated(client) -> None:
    r = _post(client, det_conf=0.3, uncertainty=0.6)
    assert r.status_code == 201
    assert r.json()["result"]["metadata"]["thresholds"]["uncertainty"] == 0.6
    bad = _post(client, uncertainty=1.5)
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "bad_request"


@pytest.mark.parametrize("data, name, status, code", [
    (b"not an image", "x.png", 400, "invalid_image"),
    (b"GIF89a", "x.gif", 415, "unsupported_format"),
    (None, "tiny", 422, "image_too_small"),
])
def test_validation_errors_use_one_envelope_and_are_recorded(client, data, name, status, code) -> None:
    if data is None:
        data, name = _png(10, 10), "tiny.png"
    r = _post(client, data, name)
    assert r.status_code == status and r.json()["error"]["code"] == code and r.json()["error"]["message"]
    items = client.get("/api/analyses").json()["items"]
    assert items[0]["status"] == "failed" and items[0]["error_code"] == code


def test_too_large_upload_is_refused(tmp_path) -> None:
    from src.common.config import load_config
    cfg = load_config()
    cfg["service"]["max_file_mb"] = 0.001                     # about 1 KB
    app = create_app(cfg=cfg, service=fake_service(tmp_path), history_root=tmp_path / "h", load_models=False,
                     frontend_dist=tmp_path / "x")
    with TestClient(app) as c:
        r = _post(c, b"\x89PNG" + b"0" * 5000)
        assert r.status_code == 413 and r.json()["error"]["code"] == "image_too_large"
        assert c.get("/api/analyses").json()["items"][0]["error_code"] == "image_too_large"


def test_missing_file_and_unknown_example(client) -> None:
    r = client.post("/api/analyses", data={})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_image"
    r = client.post("/api/analyses", data={"example": "nope.jpg"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


def test_zero_detections_is_201_with_warning(tmp_path) -> None:
    app = create_app(service=fake_service(tmp_path, preds=[]), history_root=tmp_path / "h", load_models=False,
                     frontend_dist=tmp_path / "x")
    with TestClient(app) as c:
        r = _post(c)
    assert r.status_code == 201
    res = r.json()["result"]
    assert res["status"] == "no_detections" and res["message"] == "No organisms detected."
    assert any("No organisms detected" in w for w in res["warnings"])


def test_inference_error_is_500_and_recorded(tmp_path) -> None:
    app = create_app(service=fake_service(tmp_path, error=RuntimeError("CUDA error: out of memory")),
                     history_root=tmp_path / "h", load_models=False, frontend_dist=tmp_path / "x")
    with TestClient(app) as c:
        r = _post(c)
        assert r.status_code == 500 and r.json()["error"]["code"] == "inference_error"
        assert "out of GPU memory" in r.json()["error"]["message"]
        assert c.get("/api/analyses").json()["items"][0]["status"] == "failed"


def test_models_missing_gives_503_but_history_works(tmp_path) -> None:
    cfg_models = tmp_path / "empty_models"
    from src.common.config import load_config
    cfg = load_config()
    cfg["paths"]["models"] = str(cfg_models)                  # absolute path: no weights there
    app = create_app(cfg=cfg, history_root=tmp_path / "h", frontend_dist=tmp_path / "x")
    with TestClient(app) as c:
        h = c.get("/api/health").json()
        assert h["status"] == "degraded" and not h["models_loaded"] and "not found" in h["error"]
        r = _post(c)
        assert r.status_code == 503 and r.json()["error"]["code"] == "model_load_error"
        assert c.get("/api/info").status_code == 503
        assert c.get("/api/analyses").status_code == 200


def test_history_list_detail_delete_clear(client) -> None:
    ids = [_post(client, name=f"f{i}.png").json()["result"]["analysis_id"] for i in range(3)]
    lst = client.get("/api/analyses", params={"limit": 2}).json()
    assert lst["total"] == 3 and [x["analysis_id"] for x in lst["items"]] == ids[::-1][:2]
    assert client.get("/api/analyses", params={"status": "failed"}).json()["items"] == []
    assert client.get("/api/analyses", params={"status": "bogus"}).status_code == 422
    d = client.get(f"/api/analyses/{ids[0]}").json()
    assert d["result"]["analysis_id"] == ids[0] and d["filename"] == "f0.png" and d["files_missing"] is False
    assert d["confirmed_total"] == 2 and set(d["links"]["reports"]) == {"csv", "json", "html", "pdf"}
    assert client.delete(f"/api/analyses/{ids[0]}").status_code == 204
    assert client.get(f"/api/analyses/{ids[0]}").status_code == 404
    assert client.get(f"/api/analyses/{ids[0]}/images/annotated").status_code == 404
    assert client.delete(f"/api/analyses/{ids[0]}").status_code == 404
    assert client.delete("/api/analyses").json()["error"]["code"] == "confirmation_required"
    assert client.delete("/api/analyses", params={"confirm": "true"}).json() == {"deleted": 2}
    assert client.get("/api/analyses").json()["total"] == 0


def test_unknown_kinds_and_ids(client) -> None:
    aid = _post(client).json()["result"]["analysis_id"]
    assert client.get(f"/api/analyses/{aid}/images/raw").json()["error"]["code"] == "not_found"
    assert client.get(f"/api/analyses/{aid}/reports/docx").status_code == 404
    assert client.get("/api/analyses/does-not-exist").status_code == 404
    assert client.get("/api/nothing-here").json()["error"]["code"] == "not_found"


def test_examples_endpoint_lists_only_val_frames(client) -> None:
    ex = client.get("/api/examples").json()
    assert isinstance(ex, list) and len(ex) <= 6
    for e in ex:
        assert e["name"].startswith("val_") and "synthetic" in e["note"]
    if ex:
        assert client.get(ex[0]["url"]).status_code == 200
        r = client.post("/api/analyses", data={"example": ex[0]["name"]})
        assert r.status_code == 201 and r.json()["result"]["filename"] == ex[0]["name"]


def test_serves_built_frontend_when_present(tmp_path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>PlanktoScan AI</title>")
    app = create_app(service=fake_service(tmp_path), history_root=tmp_path / "h", load_models=False, frontend_dist=dist)
    with TestClient(app) as c:
        assert "PlanktoScan AI" in c.get("/").text
        assert c.get("/api/health").json()["status"] == "ok"


def test_cli_and_api_use_the_same_service_code() -> None:
    import inspect

    import api.main as api_main
    import scripts.analyze as cli
    for mod in (api_main, cli):
        src = inspect.getsource(mod)
        assert "Pipeline(" not in src and "predict_logits" not in src and "YOLO(" not in src
