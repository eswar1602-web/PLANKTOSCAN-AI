"""Download ZooLake data.zip (492 MB), verify SHA-256 and extract to data/raw/zoolake/.

Usage (from the project root):  python -m scripts.download_zoolake
"""

from __future__ import annotations

from src.common.config import load_config, resolve_path
from src.data.download import download, safe_extract


def main() -> None:
    cfg = load_config()
    zl = cfg["data"]["zoolake"]
    raw_dir = resolve_path(cfg, "raw_data") / "zoolake"
    zip_path = download(zl["url"], raw_dir / zl["filename"], zl["sha256"])
    marker = raw_dir / ".extracted"
    if not marker.exists():
        safe_extract(zip_path, raw_dir)
        marker.touch()
    print(f"[download] data ready in {raw_dir}")


if __name__ == "__main__":
    main()
