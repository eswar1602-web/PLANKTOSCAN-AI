"""Check the extracted ZooLake data and print every raw class with its count.

Usage (from the project root):  python -m scripts.inventory_zoolake
Writes data/interim/zoolake_inventory.csv and results/raw_class_counts.csv.
"""

from __future__ import annotations

from src.common.config import load_config, resolve_path
from src.data.inventory import build_inventory, class_counts, imbalance_ratio


def main() -> None:
    cfg = load_config()
    zl = cfg["data"]["zoolake"]
    root = resolve_path(cfg, "raw_data") / "zoolake" / zl["images_subdir"]
    inv = build_inventory(root)

    interim = resolve_path(cfg, "interim_data")
    results = resolve_path(cfg, "results")
    interim.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    inv.to_csv(interim / "zoolake_inventory.csv", index=False)

    counts = class_counts(inv)
    counts.to_csv(results / "raw_class_counts.csv")
    n_bad = int((~inv["readable"]).sum())
    n_imgs, n_cls = len(inv), inv["raw_class"].nunique()

    print(f"images found: {n_imgs} (expected {zl['expected_images']})")
    print(f"classes found: {n_cls} (expected {zl['expected_classes']})")
    print(f"unreadable/corrupt: {n_bad}")
    print(f"missing timestamp: {int(inv['timestamp'].isna().sum())}")
    if n_bad:
        print(inv.loc[~inv["readable"], ["raw_class", "path"]].to_string(index=False))
    print()
    print(counts.to_string())
    print(f"\nimbalance ratio (max/min): {imbalance_ratio(counts['images']):.1f}")


if __name__ == "__main__":
    main()
