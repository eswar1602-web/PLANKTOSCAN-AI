"""Export every label correction in the history as a retraining-ready folder (same as the web app's zip).

Usage:  python -m scripts.export_corrections --out corrections_export [--history-db PATH]
Writes <out>/crops/<corrected_label>/<analysis_id>_<detection>.png, manifest.csv, skipped.csv and
summary.json. The out folder must not exist yet (an export never overwrites). Nothing is written into the
training data; review the crops before using them.
Exit code: 0 exported, 1 nothing to export, 2 bad arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.common.config import load_config, resolve_path
from src.corrections import export_corrections
from src.history import AnalysisHistory


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export label corrections as crops + manifest.")
    ap.add_argument("--out", type=Path, required=True, help="new output folder")
    ap.add_argument("--history-db", type=Path, help="SQLite history file (default: paths.history/history.sqlite)")
    a = ap.parse_args(argv)
    cfg = load_config()
    if a.out.exists():
        print(f"error: {a.out} already exists; choose a new folder", file=sys.stderr)
        return 2
    db = a.history_db or resolve_path(cfg, "history") / "history.sqlite"
    if not Path(db).is_file():
        print(f"error: no history database at {db}", file=sys.stderr)
        return 2
    rows = AnalysisHistory(db).all_corrections()
    if not rows:
        print("nothing to export: no corrections in the history")
        return 1
    summary = export_corrections(rows, a.out, cfg["pipeline"]["crop_padding"])
    print(json.dumps({k: summary[k] for k in ("corrections", "crops_written", "skipped", "by_corrected_label",
                                              "agreed_with_model")}, indent=2))
    print(f"exported to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
