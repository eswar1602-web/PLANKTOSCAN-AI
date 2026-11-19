"""CLI (scripts/analyze.py) with a fake service: single file, folder with failures, exit codes, history."""

import json

import pandas as pd

from scripts.analyze import main
from src.history import AnalysisHistory
from tests.fakes import fake_service, write_image


def _args(inp, out, db, *extra):
    return [str(inp), "--out", str(out), "--history-db", str(db), *extra]


def test_single_image(tmp_path, capsys) -> None:
    img = write_image(tmp_path / "one.png")
    code = main(_args(img, tmp_path / "out", tmp_path / "h.sqlite"), service=fake_service(tmp_path))
    assert code == 0
    dirs = [p for p in (tmp_path / "out").iterdir() if p.is_dir()]
    assert len(dirs) == 1 and {f.name for f in dirs[0].iterdir()} == {
        "annotated.png", "enhanced.png", "report.csv", "report.json", "report.html", "report.pdf"}
    rep = json.loads((dirs[0] / "report.json").read_text())
    assert rep["counts"]["confirmed_total"] == 2 and rep["estimated_counts"]["organisms_total"] == 4
    assert "one.png: ok, confirmed 2, estimated 4, uncertain 2, rejected 1" in capsys.readouterr().out


def test_folder_continues_after_failures(tmp_path) -> None:
    src = tmp_path / "in"
    src.mkdir()
    write_image(src / "a.png"), write_image(src / "b.jpg")
    (src / "broken.png").write_bytes(b"not an image")
    write_image(src / "tiny.png", h=10, w=10)
    (src / "notes.txt").write_text("ignored: not an image extension")
    db = tmp_path / "h.sqlite"
    code = main(_args(src, tmp_path / "out", db), service=fake_service(tmp_path))
    assert code == 1                                         # some failed, the rest was processed
    batch = pd.read_csv(next((tmp_path / "out").glob("batch_*.csv")))
    assert len(batch) == 4
    st = dict(zip(batch["image"].map(lambda s: s.replace("\\", "/").split("/")[-1]), batch["status"]))
    assert st == {"a.png": "ok", "b.jpg": "ok", "broken.png": "failed", "tiny.png": "failed"}
    err = dict(zip(batch["image"].map(lambda s: s.replace("\\", "/").split("/")[-1]), batch["error"]))
    assert err["broken.png"].startswith("invalid_image") and err["tiny.png"].startswith("image_too_small")
    assert batch.loc[batch["status"] == "ok", "confirmed_total"].tolist() == [2, 2]
    h = AnalysisHistory(db)
    assert h.count() == 4 and len(h.list(status="failed")) == 2
    assert len(json.loads(next((tmp_path / "out").glob("batch_*.json")).read_text())) == 4


def test_formats_and_no_history(tmp_path) -> None:
    img = write_image(tmp_path / "one.png")
    db = tmp_path / "h.sqlite"
    assert main(_args(img, tmp_path / "out", db, "--formats", "csv", "--no-history"),
                service=fake_service(tmp_path)) == 0
    d = next(p for p in (tmp_path / "out").iterdir() if p.is_dir())
    assert (d / "report.csv").exists() and (d / "report.json").exists() and not (d / "report.pdf").exists()
    assert not db.exists()


def test_threshold_options_reach_the_service(tmp_path) -> None:
    svc = fake_service(tmp_path)
    main(_args(write_image(tmp_path / "x.png"), tmp_path / "o", tmp_path / "h.sqlite", "--det-conf", "0.3",
               "--uncertainty", "0.6"), service=svc)
    assert svc.pipe.calls == [(0.3, 0.6)]


def test_bad_invocations_exit_2(tmp_path) -> None:
    svc = fake_service(tmp_path)
    (tmp_path / "empty").mkdir()
    assert main(_args(tmp_path / "missing", tmp_path / "o", tmp_path / "h"), service=svc) == 2
    assert main(_args(tmp_path / "empty", tmp_path / "o", tmp_path / "h"), service=svc) == 2
    img = write_image(tmp_path / "x.png")
    assert main(_args(img, tmp_path / "o", tmp_path / "h", "--formats", "docx"), service=svc) == 2
    assert main(_args(img, tmp_path / "o", tmp_path / "h", "--uncertainty", "2"), service=svc) == 2


def test_repeated_runs_never_overwrite(tmp_path) -> None:
    img = write_image(tmp_path / "x.png")
    for _ in range(2):
        assert main(_args(img, tmp_path / "out", tmp_path / "h.sqlite"), service=fake_service(tmp_path)) == 0
    assert len([p for p in (tmp_path / "out").iterdir() if p.is_dir()]) == 2
    assert len(list((tmp_path / "out").glob("batch_*.csv"))) == 2
