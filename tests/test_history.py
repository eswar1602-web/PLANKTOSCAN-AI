"""SQLite history: add / list / get / delete / clear, failures, no binaries, safe file deletion."""

import sqlite3

import pytest

from src.history import AnalysisHistory
from src.service import save_outputs
from tests.fakes import fake_service, write_image


def _saved(tmp_path, root, name="a.png"):
    svc = fake_service(tmp_path)
    r = svc.analyze(write_image(tmp_path / name))
    folder, files = save_outputs(r, root)
    return r.to_dict(), folder, files


def test_add_get_list_roundtrip(tmp_path) -> None:
    root = tmp_path / "hist"
    h = AnalysisHistory(root / "h.sqlite", managed_root=root)
    r1, f1, files1 = _saved(tmp_path, root, "a.png")
    r2, f2, files2 = _saved(tmp_path, root, "b.png")
    h.add_result(r1, f1, files1)
    h.add_result(r2, f2, files2)
    rec = h.get(r1["analysis_id"])
    assert rec["filename"] == "a.png" and rec["status"] == "ok" and rec["n_detections"] == 5
    assert rec["confirmed_total"] == 2 and rec["estimated_total"] == 4 and rec["uncertain"] == 2 and rec["rejected"] == 1
    assert rec["confirmed"] == r1["counts"]["confirmed"] and rec["estimated"] == r1["estimated_counts"]["per_class"]
    assert rec["files"] == files1 and rec["output_dir"] == str(f1) and rec["processing_ms"] > 0
    assert rec["uncertainty"] == r1["metadata"]["thresholds"]["uncertainty"]
    assert [x["analysis_id"] for x in h.list()] == [r2["analysis_id"], r1["analysis_id"]]   # newest first
    assert h.count() == 2 and h.get("missing") is None


def test_failures_are_recorded(tmp_path) -> None:
    h = AnalysisHistory(tmp_path / "h.sqlite")
    h.add_failure("id-1", "bad.png", "invalid_image", "bad.png: corrupted", "/x/bad.png", 3.0)
    rec = h.get("id-1")
    assert rec["status"] == "failed" and rec["error_code"] == "invalid_image" and rec["n_detections"] is None
    assert h.list(status="failed")[0]["analysis_id"] == "id-1" and h.list(status="ok") == []


def test_no_binary_columns_and_duplicate_ids_rejected(tmp_path) -> None:
    h = AnalysisHistory(tmp_path / "h.sqlite")
    with sqlite3.connect(tmp_path / "h.sqlite") as con:
        types = {row[2].upper() for row in con.execute("PRAGMA table_info(analyses)")}
    con.close()
    assert "BLOB" not in types and types <= {"TEXT", "INTEGER", "REAL"}
    h.add_failure("dup", "a.png", "x", "y")
    with pytest.raises(sqlite3.IntegrityError):
        h.add_failure("dup", "b.png", "x", "y")
    assert h.count() == 1                       # the failed insert was rolled back


def test_delete_removes_only_managed_folders(tmp_path) -> None:
    root = tmp_path / "hist"
    h = AnalysisHistory(root / "h.sqlite", managed_root=root)
    r1, f1, files1 = _saved(tmp_path, root, "a.png")                   # inside the managed root
    r2, f2, files2 = _saved(tmp_path, tmp_path / "user_out", "b.png")  # CLI output chosen by the user
    h.add_result(r1, f1, files1)
    h.add_result(r2, f2, files2)
    assert h.delete(r1["analysis_id"]) and not f1.exists()
    assert h.delete(r2["analysis_id"]) and f2.exists()                 # never deletes outside the managed root
    assert not h.delete("missing") and h.count() == 0


def test_clear(tmp_path) -> None:
    root = tmp_path / "hist"
    h = AnalysisHistory(root / "h.sqlite", managed_root=root)
    folders = []
    for n in ("a.png", "b.png", "c.png"):
        r, f, files = _saved(tmp_path, root, n)
        h.add_result(r, f, files)
        folders.append(f)
    assert h.clear() == 3 and h.count() == 0 and not any(f.exists() for f in folders)
    assert (root / "h.sqlite").exists()                                # the database itself is kept


def test_persists_across_instances(tmp_path) -> None:
    AnalysisHistory(tmp_path / "h.sqlite").add_failure("p", "a.png", "x", "y")
    assert AnalysisHistory(tmp_path / "h.sqlite").get("p")["filename"] == "a.png"
