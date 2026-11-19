"""Analyse one image or a folder of images from the command line (same service as the app).

Usage:
  python -m scripts.analyze <image_or_folder> --out <dir> [--formats csv,json,html,pdf] [--recursive]
                            [--det-conf 0.5] [--uncertainty 0.89] [--no-history] [--history-db PATH]

For every image: a new folder <out>/<stem>_<analysis_id>/ with annotated.png, enhanced.png, report.json,
report.csv, report.html, report.pdf. For the whole run: <out>/batch_<timestamp>.csv and .json (one row
per image, failures included). A failed image is logged and the rest of the folder is still processed.
Exit code: 0 all images analysed, 1 at least one image failed, 2 nothing could run (bad input or models).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.common.config import load_config, resolve_path
from src.history import AnalysisHistory
from src.service import (
    STATUS_FAILED, AnalysisError, AnalysisService, AnalysisSettings, new_analysis_id, save_outputs,
)


def list_inputs(path: Path, formats: list[str], recursive: bool) -> list[Path]:
    """Image files to process. A single file is returned as is (validated later, so errors are reported)."""
    if path.is_file():
        return [path]
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in path.glob(pattern) if p.is_file() and p.suffix.lower() in formats)


def main(argv: list[str] | None = None, service: AnalysisService | None = None) -> int:
    ap = argparse.ArgumentParser(description="PlanktoScan AI: detect, classify and count plankton in images.")
    ap.add_argument("input", type=Path, help="an image file or a folder of images")
    ap.add_argument("--out", type=Path, required=True, help="output folder (created if missing)")
    ap.add_argument("--formats", default="csv,json,html,pdf", help="comma-separated subset of csv,json,html,pdf")
    ap.add_argument("--recursive", action="store_true", help="also search sub-folders")
    ap.add_argument("--det-conf", type=float, help="detection confidence (default: the VAL-chosen value)")
    ap.add_argument("--uncertainty", type=float, help="uncertainty threshold (default: the VAL-chosen value)")
    ap.add_argument("--history-db", type=Path, help="SQLite history file (default: paths.history/history.sqlite)")
    ap.add_argument("--no-history", action="store_true", help="do not record the analyses in the history")
    a = ap.parse_args(argv)

    cfg = load_config() if service is None else service.cfg
    formats = tuple(f.strip().lower() for f in a.formats.split(",") if f.strip())
    bad = set(formats) - {"csv", "json", "html", "pdf"}
    if bad:
        print(f"error: unknown format(s): {', '.join(sorted(bad))}", file=sys.stderr)
        return 2
    for name, v in (("--det-conf", a.det_conf), ("--uncertainty", a.uncertainty)):
        if v is not None and not 0.0 <= v <= 1.0:
            print(f"error: {name} must be between 0 and 1, got {v}", file=sys.stderr)
            return 2
    if not a.input.exists():
        print(f"error: {a.input} does not exist", file=sys.stderr)
        return 2
    inputs = list_inputs(a.input, cfg["service"]["formats"], a.recursive)
    if not inputs:
        print(f"error: no supported images ({', '.join(cfg['service']['formats'])}) in {a.input}", file=sys.stderr)
        return 2
    try:
        service = service or AnalysisService(cfg)
    except AnalysisError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    history = None
    if not a.no_history:
        root = resolve_path(cfg, "history")
        history = AnalysisHistory(a.history_db or root / "history.sqlite", managed_root=root)
    settings = AnalysisSettings(det_conf=a.det_conf, uncertainty=a.uncertainty)
    a.out.mkdir(parents=True, exist_ok=True)
    d = service.defaults
    dc = d["det_conf"] if a.det_conf is None else a.det_conf
    un = d["uncertainty"] if a.uncertainty is None else a.uncertainty
    print(f"{len(inputs)} image(s); device {service.model_info['device']}; thresholds detection {dc:.2f}, "
          f"uncertainty {un:.2f}; output {a.out}")

    rows = []
    for i, path in enumerate(inputs, 1):
        t0 = time.perf_counter()
        try:
            result = service.analyze(path, settings)
            folder, files = save_outputs(result, a.out, formats)
            r = result.to_dict()
            if history:
                history.add_result(r, folder, files)
            c, e = r["counts"], r["estimated_counts"]
            rows.append({"image": str(path), "status": r["status"], "analysis_id": r["analysis_id"],
                         "confirmed_total": c["confirmed_total"], "estimated_total": e["organisms_total"],
                         "uncertain": c["uncertain"], "rejected": c["rejected"], "detections": c["total_detections"],
                         **{f"confirmed_{k}": v for k, v in c["confirmed"].items()},
                         **{f"estimated_{k}": v for k, v in e["per_class"].items()},
                         "time_ms": round(r["timings_ms"]["total"], 1), "warnings": " | ".join(r["warnings"]),
                         "output_dir": str(folder), "error": ""})
            print(f"[{i}/{len(inputs)}] {path.name}: {r['status']}, confirmed {c['confirmed_total']}, "
                  f"estimated {e['organisms_total']}, uncertain {c['uncertain']}, rejected {c['rejected']} "
                  f"({r['timings_ms']['total']:.0f} ms)")
        except (AnalysisError, OSError, ValueError) as err:     # one bad image never stops the batch
            code = getattr(err, "code", type(err).__name__)
            ms = (time.perf_counter() - t0) * 1000
            aid = new_analysis_id()
            if history:
                history.add_failure(aid, path.name, code, str(err), str(path.resolve()), ms)
            rows.append({"image": str(path), "status": STATUS_FAILED, "analysis_id": aid, "time_ms": round(ms, 1),
                         "error": f"{code}: {err}"})
            print(f"[{i}/{len(inputs)}] {path.name}: FAILED ({code}) {err}", file=sys.stderr)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    df = pd.DataFrame(rows)
    count_cols = [c for c in df.columns if c.startswith(("confirmed", "estimated")) or c in ("uncertain", "rejected", "detections")]
    df[count_cols] = df[count_cols].astype("Int64")             # integers; empty for failed images
    df.to_csv(a.out / f"batch_{stamp}.csv", index=False)
    (a.out / f"batch_{stamp}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    n_fail = int((df["status"] == STATUS_FAILED).sum())
    print(f"done: {len(rows) - n_fail} analysed, {n_fail} failed; summary {a.out / f'batch_{stamp}.csv'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
