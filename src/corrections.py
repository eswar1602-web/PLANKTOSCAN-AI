"""Label corrections: a person's verdict on single detections, kept apart from the model's output.

A correction never rewrites a stored result or a report. It is stored per (analysis, detection) in the
history database, gives "reviewed counts" next to the model's counts, and can be exported as a folder of
crops (one sub-folder per corrected label) with a manifest, ready to be checked and used for retraining.
The export never writes into the training data.

Allowed labels: the classifier's classes (organism groups and the reject class) plus
  other_organism  a real organism outside the trained groups (an open-set example)
  not_an_object   a false detection: background, noise, an empty box
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

OTHER, NOT_OBJECT = "other_organism", "not_an_object"
EXTRA_LABELS = (OTHER, NOT_OBJECT)
MAX_NOTE = 500


def allowed_labels(result: dict[str, Any]) -> list[str]:
    """The labels a detection of this result may be corrected to."""
    cls = result["metadata"]["classes"]
    return list(cls["organisms"]) + [cls["reject"], *EXTRA_LABELS]


def validate_correction(result: dict[str, Any], detection_id: int, label: str, note: str | None) -> dict[str, Any]:
    """The stored detection a correction refers to.

    Raises LookupError for an unknown detection id, ValueError for an unknown label or a note that is too long.
    """
    det = next((d for d in result["detections"] if d["id"] == detection_id), None)
    if det is None:
        raise LookupError(f"Analysis {result['analysis_id']} has no detection {detection_id}.")
    labels = allowed_labels(result)
    if label not in labels:
        raise ValueError(f"Unknown label {label!r}; use one of {', '.join(labels)}.")
    if note is not None and len(note) > MAX_NOTE:
        raise ValueError(f"The note is longer than {MAX_NOTE} characters.")
    return det


def reviewed_counts(result: dict[str, Any], corrections: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts after review: a corrected detection counts under its corrected label, the rest as the model said.

    Unreviewed detections keep the model's status (confirmed organisms count, uncertain ones stay
    'unreviewed uncertain', rejected ones are not organisms). 'other_organism' is counted separately.
    """
    cls = result["metadata"]["classes"]
    organisms, reject = list(cls["organisms"]), cls["reject"]
    by_id = {c["detection_id"]: c for c in corrections}
    counts = {c: 0 for c in organisms}
    other = unreviewed_uncertain = 0
    agreed = changed = 0
    for d in result["detections"]:
        c = by_id.get(d["id"])
        if c is None:
            if d["status"] == "organism":
                counts[d["label"]] += 1
            elif d["status"] == "uncertain":
                unreviewed_uncertain += 1
            continue
        label = c["corrected_label"]
        agreed += label == d["label"]
        changed += label != d["label"]
        if label in counts:
            counts[label] += 1
        elif label == OTHER:
            other += 1
        # reject / not_an_object: not an organism
    total = sum(counts.values()) + other
    return {"confirmed": counts, "other_organism": other, "organisms_total": total,
            "unreviewed_uncertain": unreviewed_uncertain, "reviewed": len(by_id), "agreed": agreed,
            "changed": changed, "detections": len(result["detections"]),
            "not_organisms": [reject, NOT_OBJECT]}


def pad_box_crop(img: np.ndarray, box: dict[str, float], pad: float) -> np.ndarray:
    """The crop the pipeline classified: the box widened by `pad` of its size on each side, clipped.

    Boxes are stored rounded to 0.1 px, so an edge can differ by 1 px from the pipeline's own cut.
    """
    from src.classify.frame_crops import pad_crop
    return pad_crop(img, (box["x0"], box["y0"], box["x1"], box["y1"]), pad)


def _original_image(rec: dict[str, Any]) -> tuple[np.ndarray | None, str]:
    """(RGB image, '') from the stored original or the CLI's source path, or (None, reason)."""
    files = rec.get("files") or {}
    for p in (files.get("original"), rec.get("source_path")):
        if p and Path(p).is_file():
            bgr = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
            if bgr is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), ""
            return None, f"cannot decode {p}"
    return None, "original image not stored (CLI run whose source file is gone)"


def export_corrections(rows: list[dict[str, Any]], out_dir: Path, default_pad: float) -> dict[str, Any]:
    """Write crops/<label>/<analysis>_<detection>.png, manifest.csv, skipped.csv and summary.json into out_dir.

    rows: AnalysisHistory.all_corrections() (each correction joined with its analysis record). Each crop uses
    the padding stored with its correction (the analysis's own); `default_pad` only for corrections saved
    before that was recorded. An image whose size differs from the analysed one (a CLI source file replaced
    since) is skipped, never cropped. Only one decoded image is held in memory at a time.
    """
    out_dir = Path(out_dir)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["analysis_id"], r["detection_id"]))
    current: tuple[str | None, np.ndarray | None, str] = (None, None, "")
    written, skipped = [], []
    pads = set()
    for r in rows:
        aid = r["analysis_id"]
        if current[0] != aid:
            img, why = _original_image(r)
            want = (r.get("image_width"), r.get("image_height"))
            if img is not None and all(want) and (img.shape[1], img.shape[0]) != want:
                img, why = None, (f"image is {img.shape[1]}x{img.shape[0]}, the analysed one was "
                                  f"{want[0]}x{want[1]}: the file changed since the analysis")
            current = (aid, img, why)
        _, img, why = current
        if img is None:
            skipped.append({"analysis_id": aid, "detection_id": r["detection_id"], "reason": why})
            continue
        pad = r.get("crop_padding")
        pad = default_pad if pad is None else pad
        pads.add(pad)
        crop = pad_box_crop(img, r["box"], pad)
        if crop.size == 0 or min(crop.shape[:2]) < 2:
            skipped.append({"analysis_id": aid, "detection_id": r["detection_id"], "reason": "box outside the image"})
            continue
        rel = Path("crops") / r["corrected_label"] / f"{aid}_{r['detection_id']:03d}.png"
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        ok, enc = cv2.imencode(".png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        if not ok:
            skipped.append({"analysis_id": aid, "detection_id": r["detection_id"], "reason": "PNG encoding failed"})
            continue
        (out_dir / rel).write_bytes(enc.tobytes())
        written.append({"crop": rel.as_posix(), "analysis_id": aid, "detection_id": r["detection_id"],
                        "image": r["filename"], "corrected_label": r["corrected_label"],
                        "predicted_label": r["predicted_label"], "predicted_status": r["predicted_status"],
                        "prob": r["prob"], "classifier_run": r.get("classifier_run") or "",
                        "x0": r["box"]["x0"], "y0": r["box"]["y0"], "x1": r["box"]["x1"], "y1": r["box"]["y1"],
                        "crop_padding": pad, "note": r.get("note") or "", "corrected_at": r["updated_at"]})

    def write_csv(name: str, items: list[dict], cols: list[str]) -> None:
        with open(out_dir / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(items)

    write_csv("manifest.csv", written, ["crop", "analysis_id", "detection_id", "image", "corrected_label",
                                        "predicted_label", "predicted_status", "prob", "classifier_run",
                                        "x0", "y0", "x1", "y1", "crop_padding", "note", "corrected_at"])
    write_csv("skipped.csv", skipped, ["analysis_id", "detection_id", "reason"])
    pairs = Counter((w["predicted_label"], w["corrected_label"]) for w in written)
    summary = {"corrections": len(rows), "crops_written": len(written), "skipped": len(skipped),
               "by_corrected_label": dict(Counter(w["corrected_label"] for w in written)),
               "agreed_with_model": sum(w["predicted_label"] == w["corrected_label"] for w in written),
               "predicted_to_corrected": [{"predicted": p, "corrected": c, "n": n} for (p, c), n in sorted(pairs.items())],
               "crop_paddings": sorted(pads),
               "note": "Human corrections of single detections. Review before training; never mix with the test split."}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
