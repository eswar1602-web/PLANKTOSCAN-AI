"""Reports for one analysis: JSON, CSV, HTML and PDF, all rendered from the SAME canonical result.

The canonical result is `AnalysisResult.to_dict()` (src/service.py). Every table below is built by
`counts_frame` / `detections_frame` from that dict, so the four formats cannot disagree on a count.
"""

from __future__ import annotations

import base64
import html
import io
import json
from typing import Any

import cv2
import numpy as np
import pandas as pd
from jinja2 import Environment
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

DOMAIN_WARNING = (
    "Models were trained on ZooLake crops (freshwater, Lake Greifensee, dark-field plankton camera) and the "
    "detector only on SYNTHETIC frames. Results on real microscope images come from a different domain and "
    "are NOT validated. Confirmed counts include only confident organism predictions; 'uncertain' and "
    "'rejected' items (debris / skins / unidentifiable objects) are reported separately. Estimated counts "
    "assign every detection, uncertain ones included, to its most likely class.")

COUNT_COLUMNS = ["class", "confirmed", "estimated", "estimated_from_uncertain", "expected", "interval_low",
                 "interval_high", "note"]
CONC_COLUMNS = ["class", "count", "per_l", "per_l_low", "per_l_high", "per_ml"]
INTERVAL_NOTE = ("expected / interval: classification uncertainty from the classifier probabilities (95%, "
                 "Poisson-binomial; temperature-calibrated when available); missed or false detections are not "
                 "included.")
DET_COLUMNS = ["id", "status", "label", "prob", "calibrated_prob", "possibly_unknown", "det_conf",
               "x0", "y0", "x1", "y1"]


def counts_frame(r: dict[str, Any]) -> pd.DataFrame:
    """One row per organism class, then ALL organisms, rejected, uncertain and total rows."""
    c, e = r["counts"], r["estimated_counts"]
    iv = (r.get("count_intervals") or {}).get("per_class", {})      # absent in results saved before intervals

    def interval(k: str) -> dict:
        v = iv.get(k)
        return {"expected": v["expected"], "interval_low": v["low"], "interval_high": v["high"]} if v else {}

    rows = [{"class": k, "confirmed": c["confirmed"][k], "estimated": e["per_class"][k],
             "estimated_from_uncertain": e["from_uncertain"][k], **interval(k), "note": ""} for k in c["confirmed"]]
    reject = r["metadata"]["classes"]["reject"]
    rows += [
        {"class": "ALL organisms", "confirmed": c["confirmed_total"], "estimated": e["organisms_total"],
         "estimated_from_uncertain": sum(e["from_uncertain"][k] for k in c["confirmed"]), "note": "sum of the rows above"},
        {"class": "rejected", "confirmed": c["rejected"], "estimated": e["reject"],
         "estimated_from_uncertain": e["from_uncertain"][reject], **interval(reject), "note": "debris / skins / unidentifiable; not organisms"},
        {"class": "uncertain", "confirmed": c["uncertain"], "estimated": None, "estimated_from_uncertain": None,
         "note": "below the confidence threshold; not in confirmed counts"},
        {"class": "total detections", "confirmed": c["total_detections"], "estimated": e["total"],
         "estimated_from_uncertain": None, "note": ""},
    ]
    return pd.DataFrame(rows, columns=COUNT_COLUMNS).astype({"estimated": "Int64", "estimated_from_uncertain": "Int64",
                                                            "interval_low": "Int64", "interval_high": "Int64"})


def concentration_frame(r: dict[str, Any]) -> pd.DataFrame | None:
    """Organisms per litre (and per mL) of the original water, with the 95% Poisson interval; None without calibration."""
    conc = r.get("concentration")
    if not conc:
        return None
    rows = [{"class": k, **v} for k, v in conc["per_class"].items()] + [{"class": "ALL organisms", **conc["total"]}]
    df = pd.DataFrame(rows)[CONC_COLUMNS]
    for col in CONC_COLUMNS[2:]:
        df[col] = df[col].map(lambda x: float(f"{x:.4g}"))
    return df


def concentration_lines(r: dict[str, Any]) -> list[str]:
    """Plain-text description of the calibration behind the concentration table."""
    conc = r.get("concentration")
    if not conc:
        return []
    cal = ", ".join(f"{k} {v:g}" for k, v in conc["calibration"].items())
    return [f"Calibration: {cal}. Imaged volume {conc['imaged_volume_ml']:.4g} mL; original water "
            f"{conc['water_volume_ml']:.4g} mL.",
            "Concentration from the confirmed counts; 95% exact Poisson interval (sampling error only, assumes "
            "randomly spread organisms; classification and detection errors are not included)."]


def detections_frame(r: dict[str, Any]) -> pd.DataFrame:
    """One row per detection (1-based id, the same ids as drawn on the annotated image)."""
    rows = [{"id": d["id"], "status": d["status"], "label": d["label"], "prob": d["prob"],
             "calibrated_prob": d.get("calibrated_prob"), "possibly_unknown": "yes" if d.get("unknown") else "",
             "det_conf": d["det_conf"], **d["box"]} for d in r["detections"]]
    return pd.DataFrame(rows, columns=DET_COLUMNS)


def report_json(r: dict[str, Any]) -> str:
    return json.dumps(r, indent=2, ensure_ascii=False)


def sensitivity_lines(r: dict[str, Any]) -> list[tuple[str, str]]:
    """Threshold sensitivity, possibly-unknown count and calibration, for results that have them."""
    out = []
    s = r.get("threshold_sensitivity")
    if s:
        out.append(("Threshold sensitivity", f"confirmed organisms {s['lower']['confirmed_total']} / "
                    f"{s['chosen']['confirmed_total']} / {s['higher']['confirmed_total']} at uncertainty "
                    f"{s['lower']['threshold']:.2f} / {s['chosen']['threshold']:.2f} / {s['higher']['threshold']:.2f}; "
                    f"{s['borderline']} detections within {s['delta']:.2f} of the threshold"))
    if "possibly_unknown" in r["counts"]:
        out.append(("Possibly unknown", f"{r['counts']['possibly_unknown']} detections unlike every trained class "
                    f"({r['counts']['possibly_unknown_confirmed']} in the confirmed counts)"))
    conf = r["metadata"].get("confidence")
    if conf:
        out.append(("Calibration", f"temperature {conf['temperature']}, unknown cut-off {conf['ood_cutoff']} "
                    f"(fitted {conf['fitted']})"))
    return out


def summary_lines(r: dict[str, Any]) -> list[tuple[str, str]]:
    t, m = r["timings_ms"], r["metadata"]
    th = m["thresholds"]
    return [("Analysis ID", r["analysis_id"]), ("Image", r["filename"]),
            ("Image size", f"{m['image']['width']} x {m['image']['height']} px"), ("Created (UTC)", r["created_at"]),
            ("Result", r["message"]),
            ("Thresholds", f"detection {th['det_conf']:.2f}, uncertainty {th['uncertainty']:.2f}, NMS IoU {th['nms_iou']}"),
            ("Classifier", f"{m['models']['classifier']['file']} (run {m['models']['classifier']['train_run']})"),
            ("Detector", m["models"]["detector"]["file"]), ("Device", str(m["models"]["device"])),
            *sensitivity_lines(r),
            ("Processing time", f"{t.get('total', 0):.0f} ms total ("
             + ", ".join(f"{k} {v:.0f}" for k, v in t.items() if k != "total") + ")")]


def report_csv(r: dict[str, Any]) -> str:
    """CSV text in sections: summary, counts, concentration (with a calibration), detections."""
    buf = io.StringIO()
    buf.write("# summary\n")
    pd.DataFrame(summary_lines(r) + [("Warning", w) for w in r["warnings"]]
                 + [("Concentration", line) for line in concentration_lines(r)],
                 columns=["field", "value"]).to_csv(buf, index=False)
    buf.write("\n# counts\n")
    counts_frame(r).to_csv(buf, index=False)
    conc = concentration_frame(r)
    if conc is not None:
        buf.write("\n# concentration (organisms per litre of original water)\n")
        conc.to_csv(buf, index=False)
    buf.write("\n# detections\n")
    detections_frame(r).to_csv(buf, index=False)
    return buf.getvalue()


def png_bytes(img_rgb: np.ndarray) -> bytes:
    ok, enc = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError("PNG encoding failed")
    return enc.tobytes()


_HTML = Environment(autoescape=True).from_string("""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>PlanktoScan report: {{ r.filename }}</title>
<style>body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:24px auto;max-width:1100px;padding:0 16px;
color:#1b1b1b;background:#fff;line-height:1.45}h1{font-size:1.5rem;margin-bottom:.2rem}h2{font-size:1.1rem;margin-top:1.6rem}
table{border-collapse:collapse;margin:8px 0;font-size:.9rem}td,th{border:1px solid #d0d0d0;padding:4px 8px;text-align:left}
th{background:#eef2f7}.note{background:#f6f7f9;border-left:4px solid #6b7a90;padding:8px 12px}
.warn{background:#fff4e5;border-left:4px solid #c9601f;padding:8px 12px;margin:6px 0}
.imgs{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
figure{margin:0}img{width:100%;border:1px solid #ccc}figcaption{font-size:.85rem;color:#555}</style></head><body>
<h1>PlanktoScan AI analysis report</h1>
<table>{% for k, v in summary %}<tr><th>{{ k }}</th><td>{{ v }}</td></tr>{% endfor %}</table>
{% for w in r.warnings %}<p class="warn">{{ w }}</p>{% endfor %}
<p class="note">{{ r.metadata.disclaimer }}</p>
<h2>Counts</h2>{{ counts | safe }}
<p class="note">{{ interval_note }}</p>
{% if conc %}<h2>Concentration (per litre of original water)</h2>{% for l in conc_lines %}<p>{{ l }}</p>{% endfor %}
{{ conc | safe }}{% endif %}
<h2>Images</h2><div class="imgs">
<figure><img src="data:image/png;base64,{{ annotated }}" alt="annotated image"><figcaption>Annotated: blue = confirmed
organism, orange = uncertain, grey = rejected. Numbers match the detection ids below.</figcaption></figure>
{% if enhanced %}<figure><img src="data:image/png;base64,{{ enhanced }}" alt="enhanced image"><figcaption>Enhanced
(display only; detection runs on the original image).</figcaption></figure>{% endif %}</div>
<h2>Detections</h2>{{ dets | safe }}
</body></html>""")


def report_html(r: dict[str, Any], annotated_rgb: np.ndarray, enhanced_rgb: np.ndarray | None = None) -> str:
    """Self-contained HTML report (images embedded as base64 PNG)."""
    return _HTML.render(
        r=r, summary=summary_lines(r),
        counts=counts_frame(r).to_html(index=False, na_rep="-", escape=True), interval_note=INTERVAL_NOTE,
        conc=None if concentration_frame(r) is None else concentration_frame(r).to_html(index=False, escape=True),
        conc_lines=concentration_lines(r),
        dets=detections_frame(r).to_html(index=False, escape=True) if r["detections"]
        else f"<p>{html.escape(r['message'])}</p>",
        annotated=base64.b64encode(png_bytes(annotated_rgb)).decode(),
        enhanced=base64.b64encode(png_bytes(enhanced_rgb)).decode() if enhanced_rgb is not None else None)


def report_pdf(r: dict[str, Any], annotated_rgb: np.ndarray) -> bytes:
    """PDF report built with reportlab (pure Python)."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.4 * cm, bottomMargin=1.4 * cm, title=f"PlanktoScan report {r['filename']}")
    ss = getSampleStyleSheet()

    def esc(s: Any) -> str:                         # reportlab paragraphs parse mini-HTML
        return html.escape(str(s))

    def table(df: pd.DataFrame, header: bool = True) -> Table:
        vals = df.astype(object).where(df.notna(), "-").astype(str).values.tolist()
        data = ([list(df.columns)] if header else []) + vals
        t = Table([[Paragraph(esc(v), ss["BodyText"]) if len(v) > 40 else v for v in row] for row in data],
                  repeatRows=1 if header else 0)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 8),
                               ("VALIGN", (0, 0), (-1, -1), "TOP")]
                              + ([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f7"))] if header else [])))
        return t

    story = [Paragraph("PlanktoScan AI analysis report", ss["Title"]),
             table(pd.DataFrame(summary_lines(r), columns=["field", "value"]), header=False), Spacer(1, 6)]
    story += [Paragraph("Warning: " + esc(w), ss["BodyText"]) for w in r["warnings"]]
    story += [Spacer(1, 4), Paragraph(esc(r["metadata"]["disclaimer"]), ss["Italic"]), Spacer(1, 8),
              Paragraph("Counts", ss["Heading2"]), table(counts_frame(r)),
              Paragraph(esc(INTERVAL_NOTE), ss["Italic"]), Spacer(1, 8)]
    conc = concentration_frame(r)
    if conc is not None:
        story += [Paragraph("Concentration (per litre of original water)", ss["Heading2"])]
        story += [Paragraph(esc(line), ss["BodyText"]) for line in concentration_lines(r)]
        story += [table(conc), Spacer(1, 8)]
    story += [Paragraph("Annotated image (blue = confirmed, orange = uncertain, grey = rejected)", ss["Heading2"])]
    h, w = annotated_rgb.shape[:2]
    width = min(17 * cm, 24 * cm * w / h)
    story.append(RLImage(io.BytesIO(png_bytes(annotated_rgb)), width=width, height=width * h / w))
    story += [Spacer(1, 8), Paragraph("Detections", ss["Heading2"]),
              table(detections_frame(r)) if r["detections"] else Paragraph(esc(r["message"]), ss["BodyText"])]
    doc.build(story)
    return buf.getvalue()
