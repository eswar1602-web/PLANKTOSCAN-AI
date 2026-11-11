"""Map classes, make the leakage-safe 70/15/15 split, run the pHash leakage check, write reports.

Usage (from the project root, after scripts.inventory_zoolake):  python -m scripts.make_splits

Outputs
  data/processed/split_manifest.csv   closed-set images with target class and split
  data/processed/open_set.csv         10 excluded classes, open-set evaluation only
  results/class_imbalance.csv / .png  counts, days and shares per class and split
  results/leakage_pairs.csv           cross-split near-duplicate pairs (if any)
  results/split_report.md             human-readable summary with warnings
NOTE: this script only counts images; it computes no model metrics on any split.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.config import load_config, resolve_path
from src.common.experiment import start_run
from src.common.seed import set_seed
from src.data.dedupe import drop_duplicate_files, file_md5
from src.data.leakage import compute_hashes, cross_split_pairs, pixel_correlation
from src.data.mapping import OPEN_SET, apply_mapping
from src.data.split import SPLITS, capture_time, check_no_block_overlap, split_dataset, split_summary

# Reference palette (dataviz skill): light surface, text inks, sequential blue, categorical slots 1-3
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE = "#2a78d6"
SPLIT_COLORS = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}


def plot_imbalance(summary: pd.DataFrame, ratio: float, out_path) -> None:
    """Left: images per class (one hue). Right: train/val/test share per class (100% stacked)."""
    s = summary.sort_values("total_images")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, length=0)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    y = np.arange(len(s))
    ax1.barh(y, s["total_images"], color=BLUE, height=0.62)
    ax1.set_yticks(y, s.index, color=INK)
    for yi, v in zip(y, s["total_images"]):
        ax1.text(v + s["total_images"].max() * 0.01, yi, f"{v:,}", va="center", color=INK2, fontsize=9)
    ax1.set_xlabel("images", color=INK2)
    ax1.set_title(f"Images per class (imbalance ratio {ratio:.1f} : 1)", color=INK, loc="left", fontsize=11)

    left = np.zeros(len(s))
    for sp in SPLITS:
        w = s[f"{sp}_share"].to_numpy()
        ax2.barh(y, w, left=left, color=SPLIT_COLORS[sp], height=0.62,
                 edgecolor=SURFACE, linewidth=2, label=sp)
        left += w
    for xv in (0.70, 0.85):
        ax2.axvline(xv, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax2.set_yticks(y, [""] * len(s))
    ax2.set_xlim(0, 1)
    ax2.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax2.set_title("Split share per class (dashed: 70 / 85 % targets)", color=INK, loc="left", fontsize=11)
    ax2.legend(ncols=3, loc="upper left", bbox_to_anchor=(0, -0.08), frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    seed = cfg["project"]["seed"]
    set_seed(seed, cfg["project"]["deterministic"])
    run_dir = start_run(cfg, "split")
    dcfg, scfg, lcfg = cfg["data"], cfg["data"]["split"], cfg["data"]["leakage"]
    images_root = resolve_path(cfg, "raw_data") / "zoolake" / dcfg["zoolake"]["images_subdir"]
    processed, results = resolve_path(cfg, "processed_data"), resolve_path(cfg, "results")
    processed.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    inv = pd.read_csv(resolve_path(cfg, "interim_data") / "zoolake_inventory.csv")
    inv = inv[inv["readable"]].reset_index(drop=True)
    closed, open_set = apply_mapping(inv, dcfg["class_mapping"], dcfg["open_set"])
    n_mapped = len(closed)
    removed = pd.DataFrame(columns=["raw_class", "target", "path", "reason"])
    if scfg["drop_duplicate_files"]:
        # dedupe over closed + open set together, so a file in both counts as a label conflict
        both = pd.concat([closed, open_set.assign(target=OPEN_SET)], ignore_index=True)
        md5 = both["path"].map(lambda p: file_md5(images_root / p))
        both, removed = drop_duplicate_files(both, md5)
        closed = both[both["target"] != OPEN_SET].reset_index(drop=True)
        open_set = both[both["target"] == OPEN_SET].drop(columns="target").reset_index(drop=True)
        removed[["raw_class", "target", "path", "reason"]].to_csv(results / "removed_duplicates.csv", index=False)
    ratios = {s: scfg[s] for s in SPLITS}
    df, fallback_log = split_dataset(closed, ratios, seed, scfg["block"], scfg["fallback_block"],
                                     scfg["fallback_min_fill"], tuple(scfg["stratify_on"]))
    overlap = check_no_block_overlap(df)
    assert not overlap, f"blocks shared across splits: {overlap[:5]}"

    df.drop(columns=["readable"]).to_csv(processed / "split_manifest.csv", index=False)
    open_set = open_set.assign(time=capture_time(open_set))
    open_set.drop(columns=["readable"]).to_csv(processed / "open_set.csv", index=False)

    summary = split_summary(df)
    ratio = summary["total_images"].max() / summary["total_images"].min()
    summary.to_csv(results / "class_imbalance.csv")
    plot_imbalance(summary, ratio, results / "class_imbalance.png")

    # --- leakage check (pHash) --------------------------------------------------------
    hashes = compute_hashes(df, images_root)
    pairs = cross_split_pairs(df, hashes, lcfg["phash_max_distance"])
    small = lcfg["small_image_px"]
    min_side = df[["width", "height"]].min(axis=1).to_numpy()
    if len(pairs):
        a, b = pairs["idx_a"].to_numpy(), pairs["idx_b"].to_numpy()
        pairs["class_a"], pairs["class_b"] = df["target"].to_numpy()[a], df["target"].to_numpy()[b]
        pairs["path_a"], pairs["path_b"] = df["path"].to_numpy()[a], df["path"].to_numpy()[b]
        pairs["hours_apart"] = (df["time"].to_numpy()[b] - df["time"].to_numpy()[a]) / np.timedelta64(1, "h")
        pairs["hours_apart"] = pairs["hours_apart"].abs().round(2)
        pairs["both_small"] = (min_side[a] < small) & (min_side[b] < small)
        pairs["pixel_corr"] = [round(pixel_correlation(images_root / x, images_root / y), 4)
                               for x, y in zip(pairs["path_a"], pairs["path_b"])]
        pairs["confirmed"] = pairs["pixel_corr"] >= lcfg["confirm_pixel_corr"]
    pairs.to_csv(results / "leakage_pairs.csv", index=False)
    n_confirmed = int(pairs["confirmed"].sum()) if len(pairs) else 0
    raw_share = pd.crosstab(df["raw_class"], df["split"], normalize="index").reindex(columns=list(SPLITS), fill_value=0)
    raw_off = raw_share[(raw_share - pd.Series(ratios)).abs().max(axis=1) > 0.10]
    subtypes = pd.crosstab([df["target"], df["raw_class"]], df["split"]).reindex(columns=list(SPLITS), fill_value=0)

    # --- report -------------------------------------------------------------------------
    warn = summary[summary["test_images"] < scfg["min_test_images_warn"]].index.tolist()
    day = df["time"].dt.strftime("%Y-%m-%d")
    bloom = []
    for cls, g in df.assign(day=day).groupby("target"):
        top_day, n = g["day"].value_counts().index[0], g["day"].value_counts().iloc[0]
        bloom.append((cls, top_day, n, 100 * n / len(g), g.loc[g["day"] == top_day, "split"].iloc[0]))
    bloom_df = pd.DataFrame(bloom, columns=["class", "busiest_day", "images", "pct_of_class", "split"])
    bloom_df = bloom_df.sort_values("pct_of_class", ascending=False)

    lines = [
        "# Split and class-imbalance report",
        "",
        f"Seed {seed}; blocks = whole {scfg['block']}s, class-stratified greedy assignment; "
        f"target {ratios}. Run folder: `{run_dir.name}`.",
        "This report counts images only. No model metric has been computed on any split; "
        "the test split stays untouched until final evaluation.",
        "",
        f"- Mapped: {n_mapped:,} images in {df['target'].nunique()} classes; removed as byte-identical "
        f"files: {len(removed)} ({removed['reason'].value_counts().to_dict() if len(removed) else 'none'}; "
        f"see `results/removed_duplicates.csv`).",
        f"- Closed set used: **{len(df):,} images, {df['target'].nunique()} classes**; "
        f"open set (never trained on, never used for tuning): **{len(open_set):,} images, "
        f"{open_set['raw_class'].nunique()} raw classes**.",
        f"- Stratified on: {', '.join(scfg['stratify_on'])}.",
        f"- Imbalance ratio (largest : smallest class): **{ratio:.1f} : 1** "
        f"({summary.index[0]} {summary['total_images'].iloc[0]:,} : "
        f"{summary.index[-1]} {summary['total_images'].iloc[-1]:,}).",
        f"- Hour-block fallback: **{fallback_log if fallback_log else 'not needed for any class'}**.",
        f"- (class, block) groups shared across splits: **{len(overlap)}**.",
        f"- Classes with < {scfg['min_test_images_warn']} test images: **{', '.join(warn) if warn else 'none'}**.",
        "",
        "## Per-class counts and days per split",
        "",
        summary.to_markdown(),
        "",
        "## Raw subtypes per split",
        "",
        subtypes.to_markdown(),
        "",
        f"Raw classes with any split share more than 10 points from target: "
        f"{', '.join(raw_off.index) if len(raw_off) else 'none'}.",
        "",
        "## Busiest single day per class",
        "",
        bloom_df.round(1).to_markdown(index=False),
        "",
        "**Diatom bloom-day warning.** Diatom images are concentrated in a few bloom days "
        "(see table). With whole-day blocks the bloom day must go to train, so diatom val/test "
        "come from few days and the asterionella/fragilaria mix differs between splits. "
        "Diatom test F1 therefore rests on few independent sampling days and mostly measures "
        "the minority genus; treat it as a less reliable estimate than the other classes.",
        "",
        "## Leakage check",
        "",
        "1. Time blocks: no (class, block) group appears in more than one split (above).",
        f"2. Stage 1, pHash candidates: 64-bit pHash, Hamming distance <= {lcfg['phash_max_distance']}, "
        f"every cross-split pair. Caveat: pHash downsamples to 32x32 and these are small dark-field "
        f"crops (median ~72x76 px; min side < {small}px: {int((min_side < small).sum()):,} of {len(df):,}), "
        f"so unrelated blobs collide often.",
        f"3. Stage 2, confirmation: a candidate is a near-duplicate only if the 64x64 grayscale "
        f"pixel correlation is >= {lcfg['confirm_pixel_corr']}.",
        "",
        f"- pHash candidate pairs across splits: {len(pairs):,}",
        f"- **Confirmed near-duplicates across splits: {n_confirmed}**",
    ]
    if len(pairs):
        lines += [
            f"  - candidates by split pair: {pairs.groupby(['split_a', 'split_b']).size().to_dict()}",
            f"  - candidates of different classes: {int((pairs['class_a'] != pairs['class_b']).sum())}; "
            f"both images small: {int(pairs['both_small'].sum())}; median time gap "
            f"{pairs['hours_apart'].median():.0f} h; max pixel corr {pairs['pixel_corr'].max():.3f}",
            "  - details: `results/leakage_pairs.csv`",
        ]
    (results / "split_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "split_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
