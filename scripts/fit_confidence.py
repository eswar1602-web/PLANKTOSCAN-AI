"""Fit the temperature and the unknown-organism flag for the deployed classifier; report what they do.

Usage:  python -m scripts.fit_confidence
Every choice uses TRAIN (class statistics) and VAL (temperature, cut-off) only. The open-set crops
(10 ZooLake classes never trained on) are scored ONCE at the end, to measure the flag; nothing is tuned
on them. Test data is never loaded. Weights are not changed.

Writes models/confidence.npz (loaded by the pipeline), results/confidence.{json,md} and a run folder.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

from src.classify.data import build_array
from src.classify.model import load_checkpoint
from src.classify.train import load_frame_crop_arrays, load_split_arrays, predict_logits
from src.common.config import load_config, resolve_path
from src.common.experiment import start_run
from src.common.seed import get_device, set_seed
from src.pipeline.confidence import ConfidenceModel, Mahalanobis, ece, fit_temperature, nll
from src.pipeline.pipeline import predict_logits_features
from src.service import file_sha256

GRID = (0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92)


def arrays(cfg, meta, split):
    gray = meta.get("grayscale", False)
    x, y, _ = load_split_arrays(cfg, (split,), meta["use_preprocess"], gray)[split]
    run_cfg = {**cfg, "classifier": {**cfg["classifier"], "frame_crops": meta["frame_crops"]}}
    xf, yf, _ = load_frame_crop_arrays(run_cfg, (split,), meta["use_preprocess"], gray)[split]
    return x, y, xf, yf


def coverage_table(conf: np.ndarray, correct: np.ndarray, grid) -> list[dict]:
    return [{"threshold": t, "coverage": round(float((conf >= t).mean()), 4),
             "accepted_accuracy": round(float(correct[conf >= t].mean()), 4) if (conf >= t).any() else None}
            for t in grid]


def main() -> None:
    cfg = load_config()
    set_seed(cfg["project"]["seed"], cfg["project"]["deterministic"])
    device = get_device(cfg["project"]["device"])
    run_dir = start_run(cfg, "fit_confidence")
    models, results = resolve_path(cfg, "models"), resolve_path(cfg, "results")
    th = json.loads((results / "thresholds.json").read_text())
    ccfg = cfg["confidence"]
    path = models / "classifier_best.pt"
    model, meta = load_checkpoint(path, device)
    assert meta["run"] == th["uncertainty_classifier_run"], "thresholds were chosen for another classifier"
    classes, reject = meta["classes"], cfg["data"]["reject_class"]
    t_raw = th["uncertainty_threshold"]

    # ---------------- features and logits: TRAIN (statistics) and VAL (choices)
    xtr, ytr, xtrf, ytrf = arrays(cfg, meta, "train")
    xva, yva, xvaf, yvaf = arrays(cfg, meta, "val")
    ztr, ftr = predict_logits_features(model, np.concatenate([xtr, xtrf]), device)
    ytr_all = np.concatenate([ytr, ytrf])
    zva, fva = predict_logits_features(model, np.concatenate([xva, xvaf]), device)
    yva_all = np.concatenate([yva, yvaf])
    n_orig = len(yva)
    ref = predict_logits(model, np.concatenate([xva, xvaf]), device)       # the path used so far
    same = {"max_abs_logit_diff": float(np.abs(ref - zva).max()),
            "argmax_changes": int((ref.argmax(1) != zva.argmax(1)).sum()), "n": len(zva)}
    assert same["argmax_changes"] == 0, same

    # ---------------- temperature: out-of-fold report, then fit on all of VAL
    rng = np.random.default_rng(ccfg["temperature_folds_seed"])
    fold = rng.permutation(len(yva_all)) % 2
    oof = np.zeros_like(zva)
    fold_t = []
    for k in (0, 1):
        t_k = fit_temperature(zva[fold != k], yva_all[fold != k])
        fold_t.append(t_k)
        oof[fold == k] = zva[fold == k] / t_k
    T = fit_temperature(zva, yva_all)
    raw_p, oof_p = softmax(zva, 1), softmax(oof, 1)
    calib = {"temperature": round(T, 4), "fold_temperatures": [round(t, 4) for t in fold_t],
             "val_nll_raw": round(nll(zva, yva_all), 4), "val_nll_calibrated_out_of_fold": round(nll(oof, yva_all), 4),
             "val_ece_raw": round(ece(raw_p, yva_all), 4), "val_ece_calibrated_out_of_fold": round(ece(oof_p, yva_all), 4),
             "val_accuracy": round(float((raw_p.argmax(1) == yva_all).mean()), 4)}

    # ---------------- threshold fragility: where the raw confidences sit
    conf_raw = raw_p.max(1)
    correct = raw_p.argmax(1) == yva_all
    cal_p = softmax(zva / T, 1)
    conf_cal = cal_p.max(1)
    # the same VAL rule on calibrated confidence: smallest threshold reaching the target accepted accuracy
    target = cfg["uncertainty"]["target_accepted_accuracy"]
    fine = np.round(np.arange(0.5, 1.0, 0.001), 3)
    ok = [t for t in fine if (conf_cal >= t).any() and correct[conf_cal >= t].mean() >= target]
    t_cal = float(min(ok)) if ok else None
    delta = cfg["uncertainty"]["sensitivity_delta"]
    frag = {
        "raw_ceiling_p99": round(float(np.quantile(conf_raw, 0.99)), 4),
        "raw_max": round(float(conf_raw.max()), 4),
        "label_smoothing_ceiling": round(1 - cfg["classifier"]["label_smoothing"] * (1 - 1 / len(classes)), 4),
        "share_within_delta_of_threshold_raw": round(float((np.abs(conf_raw - t_raw) <= delta).mean()), 4),
        "coverage_raw": coverage_table(conf_raw, correct, GRID),
        "calibrated_threshold_same_rule": t_cal,
        "coverage_at_calibrated_threshold": round(float((conf_cal >= t_cal).mean()), 4) if t_cal else None,
        "status_agreement_raw_vs_calibrated_rule": round(float(((conf_raw >= t_raw) == (conf_cal >= t_cal)).mean()), 4)
        if t_cal else None,
    }

    # ---------------- unknown flag: class statistics from TRAIN, cut-off from KNOWN VAL crops
    maha = Mahalanobis.fit(ftr, ytr_all, len(classes))
    s_val = maha.score(fva)
    cut = float(np.quantile(s_val, ccfg["ood_known_quantile"]))
    unk_val = s_val > cut
    org_idx = [i for i, c in enumerate(classes) if c != reject]
    known = {"cutoff": round(cut, 2), "quantile": ccfg["ood_known_quantile"],
             "val_flagged_all": round(float(unk_val.mean()), 4),
             "val_flagged_original_crops": round(float(unk_val[:n_orig].mean()), 4),
             "val_flagged_frame_crops": round(float(unk_val[n_orig:].mean()), 4),
             "val_flagged_among_confirmed_organisms": round(float(unk_val[(conf_raw >= t_raw)
                                                                         & np.isin(raw_p.argmax(1), org_idx)].mean()), 4),
             "val_accuracy_of_flagged": round(float(correct[unk_val].mean()), 4),
             "val_accuracy_of_unflagged": round(float(correct[~unk_val].mean()), 4)}

    # ---------------- ONE measurement on the open-set crops
    od = pd.read_csv(resolve_path(cfg, "processed_data") / "open_set.csv")
    root = resolve_path(cfg, "raw_data") / "zoolake" / cfg["data"]["zoolake"]["images_subdir"]
    ox = build_array(od, root, meta["image_size"], meta["preprocess"], resolve_path(cfg, "processed_data") / "cache",
                     "openset", meta.get("grayscale", False))
    zo, fo = predict_logits_features(model, ox, device)
    po = softmax(zo, 1)
    unc_o = po.max(1) < t_raw
    org_o = ~unc_o & np.isin(po.argmax(1), org_idx)
    unk_o = maha.score(fo) > cut
    od = od.assign(uncertain=unc_o, counted=org_o, unknown=unk_o)

    def outcome(d: pd.DataFrame) -> dict:
        return {"n": len(d), "uncertain": round(float(d["uncertain"].mean()), 4),
                "unknown_flag": round(float(d["unknown"].mean()), 4),
                "uncertain_or_unknown": round(float((d["uncertain"] | d["unknown"]).mean()), 4),
                "counted_as_organism": round(float(d["counted"].mean()), 4),
                "counted_without_any_flag": round(float((d["counted"] & ~d["unknown"]).mean()), 4)}

    open_set = {"ALL": outcome(od), "ALL excluding maybe_cyano": outcome(od[od["raw_class"] != "maybe_cyano"]),
                **{c: outcome(g) for c, g in od.groupby("raw_class")}}

    # ---------------- save
    info = {"fitted": run_dir.name, "temperature": T, "ood_cutoff": cut, "ood_known_quantile": ccfg["ood_known_quantile"],
            "classifier_sha256": file_sha256(path), "train_features": int(len(ytr_all)), "val_features": int(len(yva_all))}
    ConfidenceModel(T, maha, cut, meta["run"], info).save(models / "confidence.npz")
    out = {"classifier_run": meta["run"], "uncertainty_threshold": t_raw, "head_split_check": same,
           "temperature": calib, "threshold_fragility": frag, "unknown_flag_val": known, "open_set": open_set,
           "run_dir": str(run_dir)}
    (results / "confidence.json").write_text(json.dumps(out, indent=2))
    (run_dir / "summary.json").write_text(json.dumps(out, indent=2))
    lines = ["# Confidence calibration and unknown flag (fitted on TRAIN/VAL; open set measured once)", "",
             f"Temperature T = {T:.3f} (folds {fold_t[0]:.3f}, {fold_t[1]:.3f}). VAL NLL {calib['val_nll_raw']} -> "
             f"{calib['val_nll_calibrated_out_of_fold']} out of fold; ECE {calib['val_ece_raw']} -> "
             f"{calib['val_ece_calibrated_out_of_fold']}.", "",
             f"Raw confidence ceiling: 99th percentile {frag['raw_ceiling_p99']}, label-smoothing ceiling "
             f"{frag['label_smoothing_ceiling']}. VAL crops within +-{delta} of the threshold {t_raw}: "
             f"{frag['share_within_delta_of_threshold_raw']:.1%}.", "",
             "| raw threshold | VAL coverage | accepted accuracy |", "|---|---|---|"]
    lines += [f"| {r['threshold']:.2f} | {r['coverage']:.4f} | {r['accepted_accuracy']} |" for r in frag["coverage_raw"]]
    lines += ["", f"Unknown cut-off {cut:.1f} ({ccfg['ood_known_quantile']:.0%} of known VAL scores). Known VAL flagged: "
                  f"original crops {known['val_flagged_original_crops']:.1%}, frame crops {known['val_flagged_frame_crops']:.1%}, "
                  f"confirmed organisms {known['val_flagged_among_confirmed_organisms']:.1%}.", "",
              "| open-set class | n | uncertain | unknown flag | uncertain or unknown | counted as organism | counted without any flag |",
              "|---|---|---|---|---|---|---|"]
    for c, r in open_set.items():
        lines.append(f"| {c} | {r['n']} | {r['uncertain']:.3f} | {r['unknown_flag']:.3f} | {r['uncertain_or_unknown']:.3f} | "
                     f"{r['counted_as_organism']:.3f} | {r['counted_without_any_flag']:.3f} |")
    (results / "confidence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps({"head_split_check": same, "calibrated_threshold": t_cal,
                      "status_agreement": frag["status_agreement_raw_vs_calibrated_rule"]}))


if __name__ == "__main__":
    main()
