"""Exportable reports for one analysed image: CSV (detections + counts), HTML, PDF (reportlab)."""

from __future__ import annotations

import base64
import io
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from jinja2 import Template
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.pipeline.counting import Detection, counts_table

DOMAIN_WARNING = (
    "Models were trained on ZooLake crops (freshwater, Lake Greifensee, dark-field plankton camera) and the "
    "detector only on SYNTHETIC frames. Results on real microscope images come from a different domain and "
    "are NOT validated. 'uncertain' items are shown and counted separately; 'reject' items "
    "(debris / skins / unidentifiable objects) are excluded from organism counts."
)


def detections_frame(dets: list[Detection]) -> pd.DataFrame:
    """One row per detection with box, detector confidence, class, probability, status."""
    cols = ["id", "status", "label", "prob", "det_conf", "x0", "y0", "x1", "y1"]
    rows = [{"id": i + 1, **d.to_row()} for i, d in enumerate(dets)]
    df = pd.DataFrame(rows, columns=cols)
    return df.round({"prob": 4, "det_conf": 4, "x0": 1, "y0": 1, "x1": 1, "y1": 1})


def to_csv(dets: list[Detection], organisms: list[str]) -> str:
    """CSV text: a counts section followed by a detections section."""
    buf = io.StringIO()
    buf.write("# counts\n")
    counts_table(dets, organisms).to_csv(buf, index=False)
    buf.write("\n# detections\n")
    detections_frame(dets).to_csv(buf, index=False)
    return buf.getvalue()


def _png_bytes(img_rgb: np.ndarray) -> bytes:
    ok, enc = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return enc.tobytes()


_HTML = Template("""<!doctype html><html><head><meta charset="utf-8"><title>PlanktoScan report</title>
<style>body{font-family:system-ui,sans-serif;margin:24px;color:#0b0b0b;background:#fcfcfb}
table{border-collapse:collapse;margin:12px 0}td,th{border:1px solid #ccc;padding:4px 8px;text-align:left}
.warn{background:#fff4e5;border-left:4px solid #eb6834;padding:8px 12px;max-width:900px}
img{max-width:100%;border:1px solid #ccc}</style></head><body>
<h1>PlanktoScan AI report</h1>
<p>Image: <b>{{ name }}</b> &middot; generated {{ when }} &middot; detection threshold {{ det_conf }},
uncertainty threshold {{ unc }}</p>
<p class="warn">{{ warning }}</p>
<h2>Counts</h2>{{ counts | safe }}
<h2>Annotated image</h2><img src="data:image/png;base64,{{ img }}" alt="annotated image">
<h2>Detections</h2>{{ dets | safe }}
</body></html>""")


def to_html(dets: list[Detection], organisms: list[str], annotated_rgb: np.ndarray, name: str,
            det_conf: float, uncertainty: float) -> str:
    """Self-contained HTML report (annotated image embedded as base64 PNG)."""
    return _HTML.render(
        name=name, when=datetime.now().strftime("%Y-%m-%d %H:%M"), det_conf=f"{det_conf:.2f}",
        unc=f"{uncertainty:.2f}", warning=DOMAIN_WARNING,
        counts=counts_table(dets, organisms).to_html(index=False, na_rep="-"),
        dets=detections_frame(dets).to_html(index=False),
        img=base64.b64encode(_png_bytes(annotated_rgb)).decode())


def to_pdf(dets: list[Detection], organisms: list[str], annotated_rgb: np.ndarray, name: str,
           det_conf: float, uncertainty: float) -> bytes:
    """PDF report built with reportlab (pure Python)."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.4 * cm, bottomMargin=1.4 * cm)
    ss = getSampleStyleSheet()
    story = [Paragraph("PlanktoScan AI report", ss["Title"]),
             Paragraph(f"Image: <b>{name}</b> &middot; generated {datetime.now():%Y-%m-%d %H:%M} &middot; "
                       f"detection threshold {det_conf:.2f}, uncertainty threshold {uncertainty:.2f}", ss["Normal"]),
             Spacer(1, 6), Paragraph(DOMAIN_WARNING, ss["Italic"]), Spacer(1, 10),
             Paragraph("Counts", ss["Heading2"])]

    def table(df: pd.DataFrame) -> Table:
        data = [list(df.columns)] + df.fillna("-").astype(str).values.tolist()
        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef8")),
                               ("FONTSIZE", (0, 0), (-1, -1), 8)]))
        return t

    story.append(table(counts_table(dets, organisms)))
    story += [Spacer(1, 10), Paragraph("Annotated image", ss["Heading2"])]
    h, w = annotated_rgb.shape[:2]
    width = 17 * cm
    story.append(RLImage(io.BytesIO(_png_bytes(annotated_rgb)), width=width, height=width * h / w))
    story += [Spacer(1, 10), Paragraph("Detections", ss["Heading2"]), table(detections_frame(dets))]
    doc.build(story)
    return buf.getvalue()
