"""Label corrections: validation, reviewed counts, history storage, REST endpoints, export (web zip and CLI)."""

import csv
import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from scripts.export_corrections import main as export_main
from src.corrections import NOT_OBJECT, OTHER, allowed_labels, export_corrections, reviewed_counts, validate_correction
from src.history import AnalysisHistory
from tests.fakes import fake_service, write_image
from tests.test_api import _png, _post

# The fake pipeline gives: 1 copepod (confirmed), 2 diatom (confirmed), 3 rotifer (uncertain),
# 4 reject (rejected), 5 dinobryon (uncertain).


@pytest.fixture
def client(tmp_path):
    app = create_app(service=fake_service(tmp_path), history_root=tmp_path / "hist", load_models=False,
                     frontend_dist=tmp_path / "no_dist")
    with TestClient(app) as c:
        yield c


def _result(tmp_path) -> dict:
    return fake_service(tmp_path).analyze(write_image(tmp_path / "a.png", h=400, w=400)).to_dict()


def _corr(det_id, label):
    return {"detection_id": det_id, "corrected_label": label}


# ---------------- logic
def test_allowed_labels_are_the_classes_plus_two(tmp_path) -> None:
    labels = allowed_labels(_result(tmp_path))
    assert labels[-2:] == [OTHER, NOT_OBJECT] and "reject" in labels and len(labels) == 10


def test_validation(tmp_path) -> None:
    r = _result(tmp_path)
    assert validate_correction(r, 3, "rotifer", None)["label"] == "rotifer"
    with pytest.raises(LookupError):
        validate_correction(r, 99, "rotifer", None)
    with pytest.raises(ValueError):
        validate_correction(r, 3, "jellyfish", None)
    with pytest.raises(ValueError):
        validate_correction(r, 3, "rotifer", "x" * 501)


def test_reviewed_counts_without_corrections_equal_the_model(tmp_path) -> None:
    r = _result(tmp_path)
    rc = reviewed_counts(r, [])
    assert rc["confirmed"] == r["counts"]["confirmed"] and rc["organisms_total"] == r["counts"]["confirmed_total"]
    assert rc["unreviewed_uncertain"] == r["counts"]["uncertain"] and rc["reviewed"] == 0


def test_reviewed_counts_apply_each_kind_of_correction(tmp_path) -> None:
    r = _result(tmp_path)
    rc = reviewed_counts(r, [_corr(1, "copepod"),       # agreed
                             _corr(2, NOT_OBJECT),      # confirmed diatom was a false detection
                             _corr(3, "rotifer"),       # uncertain rotifer confirmed by a person
                             _corr(4, OTHER),           # rejected item was an untrained organism
                             _corr(5, "diatom")])       # uncertain dinobryon was a diatom
    assert rc["confirmed"]["copepod"] == 1 and rc["confirmed"]["diatom"] == 1 and rc["confirmed"]["rotifer"] == 1
    assert rc["confirmed"]["dinobryon"] == 0 and rc["other_organism"] == 1 and rc["organisms_total"] == 4
    assert rc["unreviewed_uncertain"] == 0 and rc["reviewed"] == 5
    assert (rc["agreed"], rc["changed"]) == (2, 3)


# ---------------- history storage
def test_history_set_replace_delete_and_cascade(tmp_path) -> None:
    h = AnalysisHistory(tmp_path / "h.sqlite")
    r = _result(tmp_path)
    h.add_result(r, None, {})
    det = r["detections"][2]
    first = h.set_correction(r["analysis_id"], det, "rotifer", "clear lorica")
    again = h.set_correction(r["analysis_id"], det, "copepod", None)          # replaces, keeps created_at
    assert again["corrected_label"] == "copepod" and again["note"] is None and again["created_at"] == first["created_at"]
    assert again["predicted_label"] == "rotifer" and again["box"] == det["box"]
    assert h.get(r["analysis_id"])["n_corrections"] == 1 and h.list()[0]["n_corrections"] == 1
    assert h.delete_correction(r["analysis_id"], 3) and not h.delete_correction(r["analysis_id"], 3)
    h.set_correction(r["analysis_id"], det, "rotifer", None)
    h.delete(r["analysis_id"])
    assert h.all_corrections() == []                                         # removed with the analysis


def test_history_clear_removes_corrections(tmp_path) -> None:
    h = AnalysisHistory(tmp_path / "h.sqlite")
    r = _result(tmp_path)
    h.add_result(r, None, {})
    h.set_correction(r["analysis_id"], r["detections"][0], "copepod", None)
    h.clear()
    assert h.all_corrections() == []


# ---------------- REST
def test_api_correction_round_trip(client) -> None:
    aid = _post(client, data=_png(400, 400)).json()["result"]["analysis_id"]
    base = f"/api/analyses/{aid}"
    v = client.get(f"{base}/corrections").json()
    assert v["items"] == [] and v["reviewed_counts"]["reviewed"] == 0 and OTHER in v["labels"]
    r = client.put(f"{base}/detections/3/correction", json={"label": "rotifer", "note": " lorica visible "})
    assert r.status_code == 200
    body = r.json()
    assert body["correction"]["corrected_label"] == "rotifer" and body["correction"]["note"] == "lorica visible"
    assert body["reviewed_counts"]["confirmed"]["rotifer"] == 1
    assert client.get("/api/analyses").json()["items"][0]["n_corrections"] == 1
    assert client.get("/api/corrections").json() == {"total": 1, "analyses": 1, "by_label": {"rotifer": 1},
                                                     "agreed_with_model": 1}
    gone = client.delete(f"{base}/detections/3/correction")
    assert gone.status_code == 200 and gone.json()["reviewed_counts"]["reviewed"] == 0
    assert client.delete(f"{base}/detections/3/correction").status_code == 404


def test_api_correction_errors(client, tmp_path) -> None:
    aid = _post(client).json()["result"]["analysis_id"]
    base = f"/api/analyses/{aid}"
    bad = client.put(f"{base}/detections/3/correction", json={"label": "jellyfish"})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "bad_request"
    assert client.put(f"{base}/detections/99/correction", json={"label": "rotifer"}).status_code == 404
    assert client.put(f"{base}/detections/3/correction", json={}).status_code == 422
    assert client.put("/api/analyses/nope/detections/1/correction", json={"label": "rotifer"}).status_code == 404
    failed = _post(client, data=b"not an image", name="x.png")
    assert failed.status_code == 400
    fid = client.get("/api/analyses", params={"status": "failed"}).json()["items"][0]["analysis_id"]
    r = client.put(f"/api/analyses/{fid}/detections/1/correction", json={"label": "rotifer"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "no_result"


def test_api_export_zip(client) -> None:
    assert client.get("/api/corrections/export").status_code == 404
    aid = _post(client, data=_png(400, 400)).json()["result"]["analysis_id"]
    for det, label in ((1, "copepod"), (3, "rotifer"), (5, NOT_OBJECT)):
        client.put(f"/api/analyses/{aid}/detections/{det}/correction", json={"label": label})
    r = client.get("/api/corrections/export")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(z.namelist())
    assert {"manifest.csv", "skipped.csv", "summary.json"} <= names
    rows = list(csv.DictReader(io.StringIO(z.read("manifest.csv").decode())))
    assert [x["corrected_label"] for x in rows] == ["copepod", "rotifer", NOT_OBJECT]
    assert all(x["crop"] in names for x in rows) and rows[1]["predicted_status"] == "uncertain"
    summary = json.loads(z.read("summary.json"))
    assert summary["crops_written"] == 3 and summary["agreed_with_model"] == 2


# ---------------- export details and CLI
def test_export_skips_missing_originals_and_boxes_outside(tmp_path) -> None:
    r = _result(tmp_path)
    img = write_image(tmp_path / "src.png", h=120, w=160)
    base = {"analysis_id": r["analysis_id"], "filename": "src.png", "predicted_label": "copepod",
            "predicted_status": "organism", "prob": 0.97, "note": None, "updated_at": "t", "files": {}}
    rows = [{**base, "detection_id": 1, "corrected_label": "copepod", "source_path": str(img),
             "box": {"x0": 10, "y0": 10, "x1": 50, "y1": 60}},
            {**base, "detection_id": 2, "corrected_label": "copepod", "source_path": str(img),
             "box": {"x0": 300, "y0": 300, "x1": 340, "y1": 350}},
            {**base, "analysis_id": "gone", "detection_id": 1, "corrected_label": "diatom",
             "source_path": str(tmp_path / "missing.png"), "box": {"x0": 1, "y0": 1, "x1": 9, "y1": 9}}]
    s = export_corrections(rows, tmp_path / "out", default_pad=0.1)
    assert (s["crops_written"], s["skipped"]) == (1, 2)
    skipped = list(csv.DictReader(open(tmp_path / "out" / "skipped.csv", encoding="utf-8")))
    assert {x["reason"] for x in skipped} == {"box outside the image",
                                              "original image not stored (CLI run whose source file is gone)"}


def test_cli_export(tmp_path, capsys) -> None:
    h = AnalysisHistory(tmp_path / "h.sqlite")
    r = _result(tmp_path)
    h.add_result({**r, "source_path": str(tmp_path / "a.png")}, None, {})
    assert export_main(["--out", str(tmp_path / "e1"), "--history-db", str(tmp_path / "h.sqlite")]) == 1
    h.set_correction(r["analysis_id"], r["detections"][0], "copepod", None)
    assert export_main(["--out", str(tmp_path / "e2"), "--history-db", str(tmp_path / "h.sqlite")]) == 0
    assert len(list((tmp_path / "e2" / "crops" / "copepod").glob("*.png"))) == 1
    assert export_main(["--out", str(tmp_path / "e2"), "--history-db", str(tmp_path / "h.sqlite")]) == 2
    assert export_main(["--out", str(tmp_path / "e3"), "--history-db", str(tmp_path / "none.sqlite")]) == 2


# ---------------- review fixes: image identity, per-analysis padding, orphans, old databases, export failure
def _row(r, img_path, **kw):
    return {"analysis_id": r["analysis_id"], "detection_id": 1, "filename": "src.png", "predicted_label": "copepod",
            "predicted_status": "organism", "prob": 0.97, "note": None, "updated_at": "t", "files": {},
            "corrected_label": "copepod", "source_path": str(img_path),
            "box": {"x0": 20, "y0": 20, "x1": 60, "y1": 60}, **kw}


def test_export_skips_an_image_that_changed_since_the_analysis(tmp_path) -> None:
    r = _result(tmp_path)
    img = write_image(tmp_path / "src.png", h=120, w=160)
    s = export_corrections([_row(r, img, image_width=400, image_height=300)], tmp_path / "out", default_pad=0.1)
    assert s["crops_written"] == 0 and s["skipped"] == 1
    reason = next(csv.DictReader(open(tmp_path / "out" / "skipped.csv", encoding="utf-8")))["reason"]
    assert "the file changed since the analysis" in reason
    ok = export_corrections([_row(r, img, image_width=160, image_height=120)], tmp_path / "out2", default_pad=0.1)
    assert ok["crops_written"] == 1


def test_export_uses_the_padding_stored_with_each_correction(tmp_path) -> None:
    r = _result(tmp_path)
    img = write_image(tmp_path / "src.png", h=120, w=160)
    s = export_corrections([_row(r, img, crop_padding=0.5), _row(r, img, detection_id=2, crop_padding=None)],
                           tmp_path / "out", default_pad=0.0)
    rows = list(csv.DictReader(open(tmp_path / "out" / "manifest.csv", encoding="utf-8")))
    import cv2
    a = cv2.imread(str(tmp_path / "out" / rows[0]["crop"]))
    b = cv2.imread(str(tmp_path / "out" / rows[1]["crop"]))
    assert a.shape[:2] == (80, 80) and b.shape[:2] == (40, 40)      # 40 px box, +-50% vs no padding
    assert s["crop_paddings"] == [0.0, 0.5]


def test_no_orphan_correction_for_a_deleted_analysis(tmp_path) -> None:
    h = AnalysisHistory(tmp_path / "h.sqlite")
    r = _result(tmp_path)
    assert h.set_correction(r["analysis_id"], r["detections"][0], "copepod", None) is None
    assert h.all_corrections() == []
    h.add_result(r, None, {})
    saved = h.set_correction(r["analysis_id"], r["detections"][0], "copepod", None,
                             image=r["metadata"]["image"], crop_padding=r["metadata"]["crop_padding"])
    assert (saved["image_width"], saved["image_height"], saved["crop_padding"]) == (400, 400, 0.1)


def test_old_corrections_table_gains_the_new_columns(tmp_path) -> None:
    import sqlite3
    db = tmp_path / "old.sqlite"
    con = sqlite3.connect(db)
    con.execute("""CREATE TABLE corrections (analysis_id TEXT NOT NULL, detection_id INTEGER NOT NULL,
                   predicted_label TEXT NOT NULL, predicted_status TEXT NOT NULL, prob REAL,
                   corrected_label TEXT NOT NULL, note TEXT, box_json TEXT NOT NULL, created_at TEXT NOT NULL,
                   updated_at TEXT NOT NULL, PRIMARY KEY (analysis_id, detection_id))""")
    con.commit()
    con.close()
    AnalysisHistory(db)
    cols = {r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(corrections)")}
    assert {"image_width", "image_height", "crop_padding"} <= cols


def test_failed_export_answers_in_the_envelope(client, monkeypatch) -> None:
    import api.main as m
    aid = _post(client, data=_png(400, 400)).json()["result"]["analysis_id"]
    client.put(f"/api/analyses/{aid}/detections/1/correction", json={"label": "copepod"})

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(m, "export_corrections", boom)
    r = client.get("/api/corrections/export")
    assert r.status_code == 500 and r.json()["error"]["code"] == "export_error" and "disk full" in r.json()["error"]["message"]
