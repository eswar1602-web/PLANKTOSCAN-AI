"""Assemble a deployable folder (Hugging Face Space or Vercel): only what the running app needs, nothing else.

    python deploy/huggingface/build_space.py                    # -> deploy/huggingface/space/
    python deploy/huggingface/build_space.py --target vercel    # -> deploy/vercel/app/
    python deploy/huggingface/build_space.py --no-examples

Build the frontend first (cd frontend && npm run build). The folder is then uploaded with huggingface_hub or
`npx vercel deploy` (see the README section "Deployment"). It is git-ignored: it holds the model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODELS = ["detector_best.pt", "classifier_best.pt", "confidence.npz"]
N_EXAMPLES = 6  # api/main.py offers the first 6 VAL frames


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", choices=["huggingface", "vercel"], default="huggingface")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: deploy/huggingface/space or deploy/vercel/app")
    ap.add_argument("--no-examples", action="store_true", help="leave out the synthetic example frames")
    a = ap.parse_args()
    if a.out is None:
        a.out = HERE / "space" if a.target == "huggingface" else ROOT / "deploy" / "vercel" / "app"

    dist = ROOT / "frontend" / "dist"
    if not (dist / "index.html").is_file():
        raise SystemExit("frontend/dist is missing: run `npm run build` in frontend/ first.")
    for m in MODELS:
        if not (ROOT / "models" / m).is_file():
            raise SystemExit(f"models/{m} is missing.")

    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for p in out.iterdir():          # start clean, but keep the Vercel project link (.vercel/)
        if p.name == ".vercel":
            continue
        shutil.rmtree(p) if p.is_dir() else p.unlink()

    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for d in ("api", "src"):
        shutil.copytree(ROOT / d, out / d, ignore=ignore)
    shutil.copytree(dist, out / "frontend" / "dist")
    shutil.copy2(ROOT / "config.yaml", out / "config.yaml")
    (out / "results").mkdir()
    shutil.copy2(ROOT / "results" / "thresholds.json", out / "results" / "thresholds.json")
    (out / "models").mkdir()
    for m in MODELS:
        shutil.copy2(ROOT / "models" / m, out / "models" / m)

    # The manifest records the deployed classifier's hash: make sure it is the one being shipped.
    manifest = json.loads((ROOT / "results" / "model_manifest.json").read_text(encoding="utf-8"))
    expected = {f["file"]: f["sha256"] for f in manifest["files"] if "sha256" in f}
    for m in ("detector_best.pt", "classifier_best.pt"):
        got = sha256(out / "models" / m)
        if m in expected and got != expected[m]:
            raise SystemExit(f"models/{m} does not match results/model_manifest.json ({got[:12]} != {expected[m][:12]}).")

    examples = []
    if not a.no_examples:
        src = ROOT / "data" / "synthetic" / "images" / "val"
        examples = sorted(src.glob("*.jpg"))[:N_EXAMPLES]
        dst = out / "data" / "synthetic" / "images" / "val"
        dst.mkdir(parents=True)
        for p in examples:
            shutil.copy2(p, dst / p.name)

    shutil.copy2(HERE / "requirements.txt", out / "requirements.txt")
    if a.target == "huggingface":
        shutil.copy2(HERE / "Dockerfile", out / "Dockerfile")
        shutil.copy2(HERE / "SPACE_README.md", out / "README.md")
    else:
        shutil.copy2(ROOT / "deploy" / "vercel" / "Dockerfile.vercel", out / "Dockerfile.vercel")
        # An explicit ignore file, so the repo's .gitignore (which excludes models/ and data/) is never applied.
        # .env* would hold the CLI's OIDC token: never upload it.
        (out / ".vercelignore").write_text("__pycache__/\n*.pyc\n.env*\n.vercel/\n", encoding="utf-8")
    (out / ".dockerignore").write_text("__pycache__/\n*.pyc\n.git/\n.vercel/\n.env*\n", encoding="utf-8")

    files = [p for p in out.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files)
    print(f"{a.target} folder: {out}")
    print(f"  {len(files)} files, {size / 1e6:.1f} MB, {len(examples)} example frames")
    for m in MODELS:
        print(f"  models/{m}: {(out / 'models' / m).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
