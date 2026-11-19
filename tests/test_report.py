"""Reports: all four formats come from the same canonical result and agree on every count."""

import io
import json

import pandas as pd

from src.pipeline.report import (
    DOMAIN_WARNING, counts_frame, detections_frame, report_csv, report_html, report_json, report_pdf,
)
from tests.fakes import fake_service, write_image


def _result(tmp_path, preds=None):
    return fake_service(tmp_path, preds=preds).analyze(write_image(tmp_path / "r&x.png"))


def _csv_sections(text: str) -> dict[str, pd.DataFrame]:
    out, name, buf = {}, None, []
    for line in text.splitlines() + ["# end"]:
        if line.startswith("# "):
            if name:
                out[name] = pd.read_csv(io.StringIO("\n".join(buf)))
            name, buf = line[2:], []
        elif line.strip():
            buf.append(line)
    return out


def test_counts_frame_keeps_confirmed_estimated_uncertain_rejected_apart(tmp_path) -> None:
    r = _result(tmp_path).to_dict()
    t = counts_frame(r).set_index("class")
    assert t.loc["copepod", "confirmed"] == 1 and t.loc["rotifer", "confirmed"] == 0
    assert t.loc["rotifer", "estimated"] == 1 and t.loc["rotifer", "estimated_from_uncertain"] == 1
    assert t.loc["ALL organisms", "confirmed"] == 2 and t.loc["ALL organisms", "estimated"] == 4
    assert t.loc["uncertain", "confirmed"] == 2 and t.loc["rejected", "confirmed"] == 1
    assert t.loc["total detections", "confirmed"] == 5 == t.loc["total detections", "estimated"]
    org = [c for c in t.index if c in r["counts"]["confirmed"]]
    assert t.loc[org, "confirmed"].sum() + t.loc["uncertain", "confirmed"] + t.loc["rejected", "confirmed"] == 5


def test_csv_json_html_pdf_agree(tmp_path) -> None:
    res = _result(tmp_path)
    r = res.to_dict()
    j = json.loads(report_json(r))
    assert j == json.loads(json.dumps(r))
    sec = _csv_sections(report_csv(r))
    assert set(sec) == {"summary", "counts", "detections"}
    csv_counts = sec["counts"].set_index("class")
    for k, v in j["counts"]["confirmed"].items():
        assert csv_counts.loc[k, "confirmed"] == v
        assert csv_counts.loc[k, "estimated"] == j["estimated_counts"]["per_class"][k]
    assert len(sec["detections"]) == len(j["detections"]) == 5
    assert sec["detections"]["status"].tolist() == [d["status"] for d in j["detections"]]
    html = report_html(r, res.annotated, res.enhanced)
    assert html.count('src="data:image/png;base64,') == 2
    assert "r&amp;x.png" in html                               # user-supplied text is escaped
    assert counts_frame(r).to_html(index=False, na_rep="-", escape=True) in html
    pdf = report_pdf(r, res.annotated)
    assert pdf[:4] == b"%PDF" and len(pdf) > 1000


def test_reports_for_zero_detections(tmp_path) -> None:
    res = _result(tmp_path, preds=[])
    r = res.to_dict()
    assert len(detections_frame(r)) == 0 and list(detections_frame(r).columns)[0] == "id"
    assert "No organisms detected" in report_html(r, res.annotated, res.enhanced)
    assert report_pdf(r, res.annotated)[:4] == b"%PDF"
    assert _csv_sections(report_csv(r))["counts"].set_index("class").loc["total detections", "confirmed"] == 0


def test_disclaimer_and_warnings_in_every_human_readable_report(tmp_path) -> None:
    res = _result(tmp_path)
    r = res.to_dict()
    assert "NOT validated" in DOMAIN_WARNING and r["metadata"]["disclaimer"] == DOMAIN_WARNING
    assert r["warnings"] and all(w in report_csv(r) for w in r["warnings"])
    html = report_html(r, res.annotated)
    assert DOMAIN_WARNING[:60] in html and r["warnings"][0][:30] in html
