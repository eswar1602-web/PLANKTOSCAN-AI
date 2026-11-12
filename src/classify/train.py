"""Train the crop classifier on TRAIN, select by VAL macro F1. Never touches the test split."""

from __future__ import annotations

import copy
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.classify.augment import augment, normalize, to_float
from src.classify.data import build_array
from src.classify.metrics import classification_metrics, macro_f1
from src.classify.model import build_model
from src.common.config import resolve_path
from src.common.experiment import start_run
from src.common.seed import get_device, set_seed


def class_weights(counts: np.ndarray, scheme: str) -> torch.Tensor | None:
    """Loss weights from train counts: none | inverse_sqrt_freq (normalised to mean 1)."""
    if scheme == "none":
        return None
    if scheme == "inverse_sqrt_freq":
        w = 1.0 / np.sqrt(counts)
    elif scheme == "inverse_freq":
        w = 1.0 / counts
    else:
        raise ValueError(f"unknown class_weighting: {scheme}")
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def class_list(cfg: dict[str, Any]) -> list[str]:
    """Fixed class order = order of class_mapping in config.yaml."""
    return list(cfg["data"]["class_mapping"].keys())


def load_split_arrays(cfg: dict[str, Any], splits: tuple[str, ...], use_preprocess: bool) -> dict:
    """{split: (uint8 images, int labels, dataframe)} for the requested splits."""
    classes = class_list(cfg)
    idx = {c: i for i, c in enumerate(classes)}
    root = resolve_path(cfg, "raw_data") / "zoolake" / cfg["data"]["zoolake"]["images_subdir"]
    df = pd.read_csv(resolve_path(cfg, "processed_data") / "split_manifest.csv")
    pcfg = cfg["preprocess"] if use_preprocess else None
    size = cfg["classifier"]["image_size"]
    out = {}
    for s in splits:
        d = df[df["split"] == s].reset_index(drop=True)
        x = build_array(d, root, size, pcfg, resolve_path(cfg, "processed_data") / "cache", s)
        out[s] = (x, d["target"].map(idx).to_numpy(), d)
    return out


@torch.no_grad()
def predict_logits(model: nn.Module, x_u8: np.ndarray, device: str, batch: int = 256, amp: bool = True) -> np.ndarray:
    """Logits for a uint8 (N, H, W, 3) array, no augmentation."""
    model.eval()
    outs = []
    for i in range(0, len(x_u8), batch):
        xb = normalize(to_float(torch.from_numpy(x_u8[i:i + batch]).to(device)))
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device == "cuda"):
            outs.append(model(xb).float().cpu())
    return torch.cat(outs).numpy()


def _lr_lambda(total_steps: int, warmup_steps: int):
    def f(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return f


def train_classifier(cfg: dict[str, Any], name: str, resume: Path | None = None) -> Path:
    """Run one training experiment; returns the run directory (best.pt, last.pt, metrics)."""
    cfg = copy.deepcopy(cfg)
    ccfg = cfg["classifier"]
    seed = cfg["project"]["seed"]
    set_seed(seed, cfg["project"]["deterministic"])
    device = get_device(cfg["project"]["device"])
    run_dir = resume if resume else start_run(cfg, name)
    classes = class_list(cfg)

    data = load_split_arrays(cfg, ("train", "val"), ccfg["use_preprocess"])   # NO test split here
    x_tr, y_tr, _ = data["train"]
    x_va, y_va, _ = data["val"]
    x_tr_t, y_tr_t = torch.from_numpy(x_tr), torch.from_numpy(y_tr).long()

    model = build_model(ccfg["backbone"], len(classes), ccfg["pretrained"]).to(device)
    counts = np.bincount(y_tr, minlength=len(classes)).astype(float)
    w = class_weights(counts, ccfg["class_weighting"])
    loss_fn = nn.CrossEntropyLoss(weight=None if w is None else w.to(device),
                                  label_smoothing=ccfg["label_smoothing"])
    opt = torch.optim.AdamW(model.parameters(), lr=ccfg["lr"], weight_decay=ccfg["weight_decay"])
    bs = ccfg["batch_size"]
    steps_per_epoch = math.ceil(len(x_tr) / bs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, _lr_lambda(ccfg["epochs"] * steps_per_epoch, ccfg["warmup_epochs"] * steps_per_epoch))
    use_amp = ccfg["amp"] and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    meta = {"backbone": ccfg["backbone"], "classes": classes, "image_size": ccfg["image_size"],
            "use_preprocess": ccfg["use_preprocess"],
            "preprocess": cfg["preprocess"] if ccfg["use_preprocess"] else None,
            "class_weighting": ccfg["class_weighting"], "run": run_dir.name}

    start_epoch, best_f1, best_epoch, bad = 0, -1.0, -1, 0
    if resume and (run_dir / "last.pt").exists():
        ck = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best_f1, best_epoch, bad = ck["epoch"] + 1, ck["best_f1"], ck["best_epoch"], ck["bad"]
        print(f"resumed at epoch {start_epoch}")

    gen = torch.Generator(device=device)
    log_path = run_dir / "metrics.csv"
    new_log = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_log:
            writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1", "lr", "seconds"])
        for epoch in range(start_epoch, ccfg["epochs"]):
            t0 = time.time()
            model.train()
            gen.manual_seed(seed * 1000 + epoch)
            perm = torch.randperm(len(x_tr_t), generator=torch.Generator().manual_seed(seed + epoch))
            tot, n = 0.0, 0
            for i in range(0, len(perm), bs):
                idx = perm[i:i + bs]
                xb = to_float(x_tr_t[idx].to(device, non_blocking=True))
                xb = normalize(augment(xb, ccfg["augment"], gen))
                yb = y_tr_t[idx].to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    loss = loss_fn(model(xb), yb)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update(); sched.step()
                tot += loss.item() * len(idx); n += len(idx)
            logits = predict_logits(model, x_va, device, amp=use_amp)
            val_loss = float(nn.functional.cross_entropy(torch.from_numpy(logits), torch.from_numpy(y_va).long()))
            pred = logits.argmax(1)
            f1 = macro_f1(y_va, pred, len(classes))
            acc = float((pred == y_va).mean())
            secs = time.time() - t0
            writer.writerow([epoch, round(tot / n, 5), round(val_loss, 5), round(acc, 5), round(f1, 5),
                             f"{sched.get_last_lr()[0]:.2e}", round(secs, 1)])
            f.flush()
            improved = f1 > best_f1
            if improved:
                best_f1, best_epoch, bad = f1, epoch, 0
                torch.save({"model": model.state_dict(), "meta": {**meta, "epoch": epoch, "val_macro_f1": f1}},
                           run_dir / "best.pt")
            else:
                bad += 1
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "epoch": epoch, "best_f1": best_f1,
                        "best_epoch": best_epoch, "bad": bad}, run_dir / "last.pt")
            print(f"epoch {epoch:3d} loss {tot / n:.4f} val_loss {val_loss:.4f} acc {acc:.4f} "
                  f"macroF1 {f1:.4f}{' *' if improved else ''} ({secs:.0f}s)", flush=True)
            if bad >= ccfg["early_stopping_patience"]:
                print(f"early stop at epoch {epoch} (best {best_epoch})")
                break

    # final VAL report with the best checkpoint
    ck = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    logits = predict_logits(model, x_va, device, amp=use_amp)
    rep = classification_metrics(y_va, logits.argmax(1), classes, cfg["data"]["reject_class"])
    rep.update({"split": "val", "best_epoch": best_epoch, "run": run_dir.name,
                "use_preprocess": ccfg["use_preprocess"], "class_weighting": ccfg["class_weighting"],
                "epochs_budget": ccfg["epochs"]})
    (run_dir / "val_report.json").write_text(json.dumps(rep, indent=2))
    np.save(run_dir / "val_logits.npy", logits)
    print(f"best epoch {best_epoch}: val macro F1 {rep['macro_f1']:.4f} "
          f"(organisms {rep['macro_f1_organisms']:.4f}), acc {rep['accuracy']:.4f}")
    return run_dir
