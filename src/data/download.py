"""Download a file with SHA-256 verification and extract zips safely."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

_CHUNK = 1 << 20  # 1 MiB
_USER_AGENT = "planktoscan-research-client"


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, expected_sha256: str) -> Path:
    """Stream url to dest and verify its SHA-256; delete the file and raise on mismatch.

    If dest already exists with the right hash, the download is skipped.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256_of(dest) == expected_sha256:
        print(f"[download] {dest.name} already present, checksum OK")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0)) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in iter(lambda: resp.read(_CHUNK), b""):
                out.write(chunk)
                bar.update(len(chunk))
    actual = sha256_of(tmp)
    if actual != expected_sha256:
        tmp.unlink()
        raise RuntimeError(
            f"SHA-256 mismatch for {dest.name}: expected {expected_sha256}, got {actual}. File deleted."
        )
    tmp.replace(dest)
    print(f"[download] {dest.name} checksum OK ({actual})")
    return dest


def safe_extract(zip_path: Path, out_dir: Path) -> None:
    """Extract a zip, refusing any member whose path escapes out_dir (zip-slip guard)."""
    out_dir = out_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (out_dir / member.filename).resolve()
            if not target.is_relative_to(out_dir):
                raise RuntimeError(f"unsafe path in zip: {member.filename}")
        for member in tqdm(zf.infolist(), desc=f"extract {zip_path.name}"):
            if member.is_dir():
                continue
            target = out_dir / member.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
