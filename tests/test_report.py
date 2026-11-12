"""Tests for CSV / HTML / PDF report generation."""

import numpy as np

from src.pipeline.counting import Detection, assign_status
from src.pipeline.report import DOMAIN_WARNING, to_csv, to_html, to_pdf

ORGS = ["copepod", "diatom"]


def _dets() -> list[Detection]:
    out = []
    for label, p in (("copepod", 0.9), ("diatom", 0.4), ("reject", 0.95)):
        out.append(Detection(box=(1, 2, 30, 40), det_conf=0.8, label=label, prob=p,
                             status=assign_status(label, p, 0.6, "reject")))
    return out


def test_csv_has_counts_and_detections() -> None:
    text = to_csv(_dets(), ORGS)
    assert "# counts" in text and "# detections" in text
    assert "uncertain,1" in text and "reject,1" in text and "copepod,1" in text


def test_html_and_pdf_render() -> None:
    img = np.zeros((60, 80, 3), np.uint8)
    html = to_html(_dets(), ORGS, img, "x.png", 0.25, 0.6)
    assert "<img src=\"data:image/png;base64," in html and DOMAIN_WARNING[:40] in html
    pdf = to_pdf(_dets(), ORGS, img, "x.png", 0.25, 0.6)
    assert pdf[:4] == b"%PDF" and len(pdf) > 1000
