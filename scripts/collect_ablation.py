"""Collect VAL results of all classifier runs into results/ablation.csv and results/ablation.md.

Usage:  python -m scripts.collect_ablation
Reads runs/*/val_report.json (written by training). Uses VAL numbers only.
"""

from __future__ import annotations

import json

import pandas as pd

from src.common.config import load_config, resolve_path


def collect() -> pd.DataFrame:
    cfg = load_config()
    rows = []
    for rep_path in sorted(resolve_path(cfg, "runs").glob("*/val_report.json")):
        rep = json.loads(rep_path.read_text())
        name = rep_path.parent.name.split("_", 2)[-1]
        row = {"run": rep_path.parent.name, "experiment": name,
               "preprocess": rep["use_preprocess"], "class_weighting": rep["class_weighting"],
               "epochs_budget": rep.get("epochs_budget"), "best_epoch": rep["best_epoch"],
               "val_accuracy": rep["accuracy"], "val_macro_f1": rep["macro_f1"],
               "val_macro_f1_organisms": rep["macro_f1_organisms"]}
        row.update({f"f1_{c}": v["f1"] for c, v in rep["per_class"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    df = collect()
    out = resolve_path(cfg, "results")
    df.to_csv(out / "ablation.csv", index=False)
    show = df[["experiment", "preprocess", "class_weighting", "epochs_budget", "best_epoch",
               "val_accuracy", "val_macro_f1", "val_macro_f1_organisms"]].copy()
    (out / "ablation.md").write_text(
        "# Classifier ablations (VAL split only, single seed)\n\n" + show.round(4).to_markdown(index=False) + "\n",
        encoding="utf-8")
    print(show.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
