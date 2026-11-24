"""REST API in browser-storage mode (stateless hosting): nothing is stored, everything the browser needs comes back,
and reports and corrections are rebuilt from what the browser sends. Compared with the server mode's own output."""

import base64
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app, encode_previews
from tests.fakes import fake_service


def _png(h=120, w=160) -> bytes:
    img = np.full((h, w, 3), 8, np.uint8)
    img[20:50, 20:60] = 200
    return cv2.imencode(".png", img)[1].tobytes()


def _post(client, data=None, name="frame.png"):
    return client.post("/api/analyses", files={"file": (name, data if data is not None else _png(), "image/png")})


@pytest.fixture
def browser(tmp_path):
    app = create_app(service=fake_service(tmp_path), history_root=tmp_path / "hist", load_models=False,
                     frontend_dist=tmp_path / "no_dist", storage="browser")
    with TestClient(app) as c:
        yield c


@pytest.fixture
def server(tmp_path):
    (tmp_path / "s").mkdir()
    app = create_app(service=fake_service(tmp_path / "s"), history_root=tmp_path / "shist", load_models=False,
                     frontend_dist=tmp_path / "no_dist", storage="server")
    with TestClient(app) as c:
        yield c


def _report(client, fmt, body):
    files = {k: (f"{k}.png", base64.b64decode(v["data"]), v["type"]) for k, v in body["images"].items()}
    return client.post(f"/api/reports/{fmt}", data={"result": json.dumps(body["result"])}, files=files)


def test_mode_is_announced_and_upload_limit_is_lower(browser) -> None:
    assert browser.get("/api/health").json()["storage"] == "browser"
    info = browser.get("/api/info").json()
    assert info["storage"] == "browser" and info["limits"]["max_file_mb"] == 4


def test_unknown_storage_mode_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError):
        create_app(service=fake_service(tmp_path), history_root=tmp_path / "h", load_models=False, storage="disk")


def test_analysis_returns_everything_and_stores_nothing(browser, tmp_path) -> None:
    r = _post(browser)
    assert r.status_code == 201
    body = r.json()
    res = body["result"]
    assert res["status"] == "ok" and len(res["detections"]) == 5
    assert set(body["images"]) == {"annotated", "enhanced"}
    for v in body["images"].values():
        assert v["type"] == "image/png"
        img = cv2.imdecode(np.frombuffer(base64.b64decode(v["data"]), np.uint8), cv2.IMREAD_COLOR)
        assert img is not None and img.shape[:2] == (120, 160)
    rc = body["review"]["reviewed_counts"]
    assert body["review"]["items"] == [] and rc["reviewed"] == 0 and rc["organisms_total"] == 2
    assert "other_organism" in body["review"]["labels"]
    assert not list((tmp_path / "hist").glob("*/report.json"))                 # no output folder written


@pytest.mark.parametrize("method, path", [
    ("get", "/api/analyses"), ("get", "/api/analyses/x"), ("get", "/api/analyses/x/images/original"),
    ("get", "/api/analyses/x/reports/pdf"), ("delete", "/api/analyses/x"), ("delete", "/api/analyses?confirm=true"),
    ("get", "/api/analyses/x/corrections"), ("delete", "/api/analyses/x/detections/1/correction"),
    ("get", "/api/corrections"), ("get", "/api/corrections/export"),
])
def test_history_endpoints_answer_409(browser, method, path) -> None:
    r = getattr(browser, method)(path)
    assert r.status_code == 409 and r.json()["error"]["code"] == "browser_storage"
    r = browser.put("/api/analyses/x/detections/1/correction", json={"label": "diatom"})
    assert r.status_code == 409


def test_errors_are_not_recorded_and_too_large_is_413(browser) -> None:
    r = _post(browser, b"not an image", "x.png")
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_image"
    r = _post(browser, b"\x89PNG" + b"0" * (4 * 1024 * 1024 + 10))
    assert r.status_code == 413 and r.json()["error"]["code"] == "image_too_large" and "4 MB" in r.json()["error"]["message"]


def test_reports_match_the_server_mode(browser, server) -> None:
    body = _post(browser).json()
    ref = _post(server).json()
    aid = ref["result"]["analysis_id"]
    for fmt in ("json", "csv", "html", "pdf"):
        r = _report(browser, fmt, body)
        assert r.status_code == 200, (fmt, r.text[:200])
        cd = r.headers["content-disposition"]
        assert cd.startswith("attachment;") and f"frame_{body['result']['analysis_id']}_planktoscan.{fmt}" in cd
        s = server.get(f"/api/analyses/{aid}/reports/{fmt}")
        assert r.headers["content-type"].split(";")[0] == s.headers["content-type"].split(";")[0]
        if fmt == "json":
            assert json.loads(r.content) == body["result"]
        if fmt == "csv":                                       # same layout, same counts (ids and times differ)
            assert "ALL organisms,2,4" in r.text and [l.split(",")[0] for l in r.text.splitlines()] == \
                [l.split(",")[0] for l in s.text.splitlines()]
        if fmt == "pdf":
            assert r.content[:4] == b"%PDF"
        if fmt == "html":
            assert "<!doctype html>" in r.text.lower() and "data:image/png;base64," in r.text


def test_report_needs_a_valid_result_and_image(browser) -> None:
    body = _post(browser).json()
    r = browser.post("/api/reports/pdf", data={"result": json.dumps(body["result"])})
    assert r.status_code == 422 and "annotated" in r.json()["error"]["message"]
    r = browser.post("/api/reports/csv", data={"result": "{not json"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "bad_request"
    r = browser.post("/api/reports/csv", data={"result": json.dumps({"a": 1})})
    assert r.status_code == 422
    r = browser.post("/api/reports/pdf", data={"result": json.dumps(body["result"])},
                     files={"annotated": ("a.png", b"nope", "image/png")})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_image"
    assert browser.post("/api/reports/docx", data={"result": "{}"}).status_code == 404


def test_review_matches_the_server_corrections(browser, server) -> None:
    body = _post(browser).json()
    ref = _post(server).json()["result"]
    aid = ref["analysis_id"]
    server.put(f"/api/analyses/{aid}/detections/3/correction", json={"label": "rotifer"})
    want = server.put(f"/api/analyses/{aid}/detections/5/correction", json={"label": "other_organism", "note": " x "}).json()
    r = browser.post("/api/review", json={"result": body["result"], "corrections": [
        {"detection_id": 5, "label": "diatom"},                                  # replaced by the later one
        {"detection_id": 3, "label": "rotifer"},
        {"detection_id": 5, "label": "other_organism", "note": " x "},
    ]})
    assert r.status_code == 200
    got = r.json()
    assert got["reviewed_counts"] == want["reviewed_counts"] and got["labels"] == want["labels"]
    assert [(c["detection_id"], c["corrected_label"], c["note"]) for c in got["items"]] == [(3, "rotifer", None), (5, "other_organism", "x")]
    assert got["items"][0]["predicted_label"] == "rotifer" and set(got["items"][0]["box"]) == {"x0", "y0", "x1", "y1"}


def test_review_validates(browser) -> None:
    res = _post(browser).json()["result"]
    r = browser.post("/api/review", json={"result": res, "corrections": [{"detection_id": 99, "label": "diatom"}]})
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    r = browser.post("/api/review", json={"result": res, "corrections": [{"detection_id": 1, "label": "whale"}]})
    assert r.status_code == 422 and "Unknown label" in r.json()["error"]["message"]
    r = browser.post("/api/review", json={"result": {"x": 1}, "corrections": []})
    assert r.status_code == 422
    assert browser.post("/api/review", json={"result": res}).json()["reviewed_counts"]["reviewed"] == 0


def test_previews_stay_within_the_budget() -> None:
    rng = np.random.default_rng(0)
    big = rng.integers(0, 255, (1500, 2000, 3), np.uint8)              # noise: PNG is huge, so JPEG, then smaller
    out = encode_previews({"annotated": big, "enhanced": big[:600, :800]}, budget=1_000_000)
    assert sum(len(v["data"]) for v in out.values()) <= 1_000_000
    assert all(v["type"] == "image/jpeg" for v in out.values())
    small = encode_previews({"annotated": np.zeros((50, 60, 3), np.uint8)}, budget=1_000_000)
    assert small["annotated"]["type"] == "image/png"
