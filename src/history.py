"""SQLite analysis history: metadata, counts and output file PATHS only (never image binaries).

One connection per call (safe with the API server's worker threads); every write runs in a transaction.
Deleting a record removes its files only if they are inside `managed_root` (the app's own history
folder), so a CLI run's output folder chosen by the user is never deleted by the history.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    analysis_id      TEXT PRIMARY KEY,
    created_at       TEXT NOT NULL,
    filename         TEXT NOT NULL,
    source_path      TEXT,
    status           TEXT NOT NULL CHECK (status IN ('ok', 'no_detections', 'failed')),
    error_code       TEXT,
    error_message    TEXT,
    n_detections     INTEGER,
    confirmed_total  INTEGER,
    estimated_total  INTEGER,
    uncertain        INTEGER,
    rejected         INTEGER,
    confirmed_json   TEXT,
    estimated_json   TEXT,
    processing_ms    REAL,
    det_conf         REAL,
    uncertainty      REAL,
    classifier_run   TEXT,
    warnings_json    TEXT,
    output_dir       TEXT,
    files_json       TEXT
);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses (created_at);
CREATE TABLE IF NOT EXISTS corrections (
    analysis_id      TEXT NOT NULL,
    detection_id     INTEGER NOT NULL,
    predicted_label  TEXT NOT NULL,
    predicted_status TEXT NOT NULL,
    prob             REAL,
    corrected_label  TEXT NOT NULL,
    note             TEXT,
    box_json         TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    image_width      INTEGER,                -- size of the analysed image, checked again at export
    image_height     INTEGER,
    crop_padding     REAL,                   -- the padding the pipeline used for this analysis
    PRIMARY KEY (analysis_id, detection_id)
);
"""
# columns added after the first release of the corrections table (ALTER TABLE on older history files)
CORRECTION_COLUMNS_ADDED = {"image_width": "INTEGER", "image_height": "INTEGER", "crop_padding": "REAL"}
N_CORR = "(SELECT COUNT(*) FROM corrections c WHERE c.analysis_id = analyses.analysis_id) AS n_corrections"


class AnalysisHistory:
    """Persistent list of analyses in an SQLite file."""

    def __init__(self, db_path: str | Path, managed_root: str | Path | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.managed_root = Path(managed_root).resolve() if managed_root else None
        with closing(self._connect()) as con, con:
            con.executescript(SCHEMA)
            have = {r[1] for r in con.execute("PRAGMA table_info(corrections)")}
            for col, typ in CORRECTION_COLUMNS_ADDED.items():
                if col not in have:
                    con.execute(f"ALTER TABLE corrections ADD COLUMN {col} {typ}")

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=10)
        con.row_factory = sqlite3.Row
        return con

    # ---------------- writes
    def add_result(self, r: dict[str, Any], output_dir: str | Path | None, files: dict[str, str]) -> str:
        """Store a successful analysis (canonical result dict from AnalysisResult.to_dict())."""
        c, e, th = r["counts"], r["estimated_counts"], r["metadata"]["thresholds"]
        row = {"analysis_id": r["analysis_id"], "created_at": r["created_at"], "filename": r["filename"],
               "source_path": r.get("source_path"), "status": r["status"], "error_code": None, "error_message": None,
               "n_detections": c["total_detections"], "confirmed_total": c["confirmed_total"],
               "estimated_total": e["organisms_total"], "uncertain": c["uncertain"], "rejected": c["rejected"],
               "confirmed_json": json.dumps(c["confirmed"]), "estimated_json": json.dumps(e["per_class"]),
               "processing_ms": r["timings_ms"].get("total"), "det_conf": th["det_conf"],
               "uncertainty": th["uncertainty"], "classifier_run": r["metadata"]["models"]["classifier"]["train_run"],
               "warnings_json": json.dumps(r["warnings"]), "output_dir": str(output_dir) if output_dir else None,
               "files_json": json.dumps(files)}
        self._insert(row)
        return row["analysis_id"]

    def add_failure(self, analysis_id: str, filename: str, error_code: str, message: str,
                    source_path: str | None = None, processing_ms: float | None = None) -> str:
        """Store a failed analysis (validation or inference error), so batch runs keep a complete log."""
        self._insert({"analysis_id": analysis_id, "created_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                      "filename": filename, "source_path": source_path, "status": "failed", "error_code": error_code,
                      "error_message": message, "processing_ms": processing_ms, "files_json": "{}"})
        return analysis_id

    def _insert(self, row: dict[str, Any]) -> None:
        cols = ", ".join(row)
        with closing(self._connect()) as con, con:           # `with con` = one transaction (rollback on error)
            con.execute(f"INSERT INTO analyses ({cols}) VALUES ({', '.join('?' * len(row))})", tuple(row.values()))

    def delete(self, analysis_id: str, delete_files: bool = True) -> bool:
        """Remove a record (and its managed output folder). Returns False if the id does not exist."""
        rec = self.get(analysis_id)
        if rec is None:
            return False
        with closing(self._connect()) as con, con:
            con.execute("DELETE FROM analyses WHERE analysis_id = ?", (analysis_id,))
            con.execute("DELETE FROM corrections WHERE analysis_id = ?", (analysis_id,))
        if delete_files:
            self._remove_managed(rec.get("output_dir"))
        return True

    def clear(self, delete_files: bool = True) -> int:
        """Remove every record (and the managed output folders). Returns the number removed."""
        recs = self.list(limit=None)
        with closing(self._connect()) as con, con:
            con.execute("DELETE FROM analyses")
            con.execute("DELETE FROM corrections")
        if delete_files:
            for r in recs:
                self._remove_managed(r.get("output_dir"))
        return len(recs)

    def _remove_managed(self, output_dir: str | None) -> None:
        if not output_dir or self.managed_root is None:
            return
        p = Path(output_dir).resolve()
        if p != self.managed_root and self.managed_root in p.parents and p.is_dir():
            shutil.rmtree(p)

    # ---------------- reads
    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        for k in ("confirmed_json", "estimated_json", "warnings_json", "files_json"):
            d[k.removesuffix("_json")] = json.loads(d.pop(k)) if d.get(k) else None
        return d

    def get(self, analysis_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as con:
            row = con.execute(f"SELECT *, {N_CORR} FROM analyses WHERE analysis_id = ?", (analysis_id,)).fetchone()
        return self._decode(row) if row else None

    def list(self, limit: int | None = 100, offset: int = 0, status: str | None = None) -> list[dict[str, Any]]:
        """Newest first."""
        q, args = f"SELECT *, {N_CORR} FROM analyses", []
        if status:
            q += " WHERE status = ?"
            args.append(status)
        q += " ORDER BY created_at DESC, rowid DESC"          # rowid = insertion order (ties within a second)
        if limit is not None:
            q += " LIMIT ? OFFSET ?"
            args += [limit, offset]
        with closing(self._connect()) as con:
            return [self._decode(r) for r in con.execute(q, args).fetchall()]

    def count(self) -> int:
        with closing(self._connect()) as con:
            return int(con.execute("SELECT COUNT(*) FROM analyses").fetchone()[0])

    # ---------------- corrections (src/corrections.py validates them against the stored result)
    @staticmethod
    def _correction(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["box"] = json.loads(d.pop("box_json"))
        return d

    def set_correction(self, analysis_id: str, detection: dict[str, Any], label: str, note: str | None,
                       image: dict[str, int] | None = None, crop_padding: float | None = None) -> dict[str, Any] | None:
        """Insert or replace the correction of one detection (the canonical detection dict of the result).

        `image` ({width, height}) and `crop_padding` describe the analysis, so the export can check that it
        crops the same image the same way. Returns None (nothing written) if the analysis no longer exists:
        the existence check and the write are one statement, so a concurrent delete leaves no orphan.
        """
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        image = image or {}
        with closing(self._connect()) as con, con:
            n = con.execute(
                """INSERT INTO corrections (analysis_id, detection_id, predicted_label, predicted_status, prob,
                                            corrected_label, note, box_json, created_at, updated_at,
                                            image_width, image_height, crop_padding)
                   SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                   WHERE EXISTS (SELECT 1 FROM analyses WHERE analysis_id = ?)
                   ON CONFLICT (analysis_id, detection_id) DO UPDATE SET
                       corrected_label = excluded.corrected_label, note = excluded.note, updated_at = excluded.updated_at""",
                (analysis_id, detection["id"], detection["label"], detection["status"], detection.get("prob"), label,
                 note or None, json.dumps(detection["box"]), now, now, image.get("width"), image.get("height"),
                 crop_padding, analysis_id)).rowcount
        if not n:
            return None
        return next(c for c in self.corrections(analysis_id) if c["detection_id"] == detection["id"])

    def delete_correction(self, analysis_id: str, detection_id: int) -> bool:
        with closing(self._connect()) as con, con:
            n = con.execute("DELETE FROM corrections WHERE analysis_id = ? AND detection_id = ?",
                            (analysis_id, detection_id)).rowcount
        return n > 0

    def corrections(self, analysis_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as con:
            rows = con.execute("SELECT * FROM corrections WHERE analysis_id = ? ORDER BY detection_id",
                               (analysis_id,)).fetchall()
        return [self._correction(r) for r in rows]

    def all_corrections(self) -> list[dict[str, Any]]:
        """Every correction joined with its analysis (filename, stored files, source path, classifier run)."""
        with closing(self._connect()) as con:
            rows = con.execute(
                """SELECT c.*, a.filename, a.source_path, a.files_json, a.classifier_run, a.output_dir
                   FROM corrections c JOIN analyses a ON a.analysis_id = c.analysis_id
                   ORDER BY c.analysis_id, c.detection_id""").fetchall()   # grouped: the export decodes each image once
        out = []
        for r in rows:
            d = self._correction(r)
            d["files"] = json.loads(d.pop("files_json")) if d.get("files_json") else {}
            out.append(d)
        return out
