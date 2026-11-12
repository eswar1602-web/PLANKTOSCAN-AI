"""Download EXACTLY the two user-approved pretrained weight files, verify, and record them.

Usage:  python -m scripts.fetch_pretrained
Allowlist (official sources only; nothing else may be fetched):
  1. yolov8n.pt - Ultralytics release assets (tag v8.4.0, the tag the installed ultralytics uses)
  2. EfficientNet-B0 ImageNet weights - timm's default tag efficientnet_b0.ra_in1k on the timm HF hub
Each file's URL, size and SHA-256 is written to models/pretrained/manifest.json, results/pretrained_weights.json
and the run log. Where the publisher exposes a checksum (GitHub release-asset digest, Hugging Face LFS
SHA-256 header), it is fetched as metadata and compared; a mismatch deletes the file and aborts.
"""

from __future__ import annotations

import json
import urllib.request

from src.common.config import load_config, resolve_path
from src.common.experiment import start_run
from src.data.download import sha256_of

UA = {"User-Agent": "planktoscan-research-client"}
ALLOWLIST = {
    "yolov8n.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
        "source": "Ultralytics GitHub release assets (ultralytics/assets, tag v8.4.0)",
    },
    "efficientnet_b0.ra_in1k.safetensors": {
        "url": "https://huggingface.co/timm/efficientnet_b0.ra_in1k/resolve/main/model.safetensors",
        "source": "timm official Hugging Face hub (timm/efficientnet_b0.ra_in1k, timm's default tag)",
    },
}


def published_sha256(name: str) -> str | None:
    """Publisher's checksum (metadata only): GitHub asset digest or HF LFS sha256 header."""
    try:
        if name == "yolov8n.pt":
            req = urllib.request.Request("https://api.github.com/repos/ultralytics/assets/releases/tags/v8.4.0",
                                         headers={**UA, "Accept": "application/vnd.github+json"})
            rel = json.load(urllib.request.urlopen(req, timeout=30))
            for a in rel.get("assets", []):
                if a["name"] == "yolov8n.pt" and str(a.get("digest", "")).startswith("sha256:"):
                    return a["digest"].split(":", 1)[1]
            return None
        req = urllib.request.Request(ALLOWLIST[name]["url"], method="HEAD", headers=UA)

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **k):
                return None
        opener = urllib.request.build_opener(NoRedirect)
        try:
            resp = opener.open(req, timeout=30)
            headers = resp.headers
        except urllib.error.HTTPError as e:      # 302 lands here; headers still carry the LFS hash
            headers = e.headers
        etag = headers.get("X-Linked-Etag") or headers.get("ETag") or ""
        etag = etag.strip('"').replace("W/", "").strip('"')
        return etag if len(etag) == 64 else None
    except Exception as e:  # metadata unavailable -> recorded as None, file hash still recorded
        print(f"[fetch] could not read published checksum for {name}: {e}")
        return None


def main() -> None:
    cfg = load_config()
    run_dir = start_run(cfg, "fetch_pretrained")
    out = resolve_path(cfg, "models") / "pretrained"
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for name, info in ALLOWLIST.items():
        dest = out / name
        if not dest.exists():
            req = urllib.request.Request(info["url"], headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                f.write(r.read())
        sha = sha256_of(dest)
        pub = published_sha256(name)
        if pub is not None and pub != sha:
            dest.unlink()
            raise SystemExit(f"SHA-256 mismatch for {name}: published {pub}, got {sha}. File deleted.")
        rec = {"file": name, "url": info["url"], "source": info["source"], "size_bytes": dest.stat().st_size,
               "sha256": sha, "published_sha256": pub,
               "verified_against_publisher": pub is not None}
        records.append(rec)
        print(json.dumps(rec, indent=2))
    for p in (out / "manifest.json", resolve_path(cfg, "results") / "pretrained_weights.json",
              run_dir / "pretrained_weights.json"):
        p.write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
