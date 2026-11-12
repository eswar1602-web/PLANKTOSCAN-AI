"""Pick the final classifier among runs whose experiment name starts with --prefix, by VAL macro F1.

Copies its best.pt to models/classifier_best.pt and writes results/classifier_selection.json.
Usage:  python -m scripts.select_classifier --prefix cls_
"""

from __future__ import annotations

import argparse
import json
import shutil

from scripts.collect_ablation import collect
from src.common.config import load_config, resolve_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="cls_")
    a = ap.parse_args()
    cfg = load_config()
    df = collect()
    cand = df[df["experiment"].str.startswith(a.prefix)]
    if cand.empty:
        raise SystemExit(f"no runs with prefix {a.prefix}")
    best = cand.sort_values("val_macro_f1", ascending=False).iloc[0]
    src = resolve_path(cfg, "runs") / best["run"] / "best.pt"
    models = resolve_path(cfg, "models")
    models.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, models / "classifier_best.pt")
    sel = {"selected_run": best["run"], "criterion": "val_macro_f1",
           "candidates": cand[["run", "preprocess", "class_weighting", "val_macro_f1"]].to_dict("records")}
    (resolve_path(cfg, "results") / "classifier_selection.json").write_text(json.dumps(sel, indent=2))
    print(json.dumps(sel, indent=2))


if __name__ == "__main__":
    main()
