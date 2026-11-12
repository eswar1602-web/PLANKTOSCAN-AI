"""Regenerate every results table and figure, the README results section and final_summary.md.

Usage:  python -m scripts.evaluate
Inputs: models/classifier_best.pt, models/detector_best.pt, results/thresholds.json (chosen on VAL),
        data/processed/{split_manifest,open_set}.csv, data/synthetic/ (frames + gt json), runs/*/val_report.json.
The TEST split / synthetic TEST frames are used here and nowhere else. This script makes no tuning
decisions: thresholds and models are fixed before it runs, so re-running it reproduces the same numbers.
"""

from __future__ import annotations

import json
import re

import cv2
import numpy as np
import pandas as pd
import torch

from scripts.collect_ablation import main as collect_ablation
from src.classify.data import build_array
from src.classify.metrics import classification_metrics
from src.classify.model import load_checkpoint
from src.classify.train import load_split_arrays, predict_logits
from src.common.config import load_config, resolve_path
from src.common.experiment import start_run
from src.common.seed import get_device, set_seed
from src.detect.yolo_utils import detection_prf, greedy_match, ultralytics_offline
from src.evaluation import bootstrap_f1, count_errors, plot_confusion, plot_count_errors, plot_f1_ci
from src.pipeline.counting import REJECT, UNCERTAIN, count_detections

F = lambda v, d=4: "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"  # noqa: E731


def md(df: pd.DataFrame, **kw) -> str:
    return df.to_markdown(**kw)


def main() -> None:
    cfg = load_config()
    seed = cfg["project"]["seed"]
    set_seed(seed, cfg["project"]["deterministic"])
    run_dir = start_run(cfg, "evaluate")
    device = get_device(cfg["project"]["device"])
    out = resolve_path(cfg, "results")
    th = json.loads((out / "thresholds.json").read_text())
    unc_t = th["uncertainty_threshold"]
    reject = cfg["data"]["reject_class"]
    S: dict = {"run": run_dir.name, "thresholds": th}

    # ---------------- 1. classifier: VAL and TEST ----------------
    model, meta = load_checkpoint(resolve_path(cfg, "models") / "classifier_best.pt", device)
    classes = meta["classes"]
    organisms = [c for c in classes if c != reject]
    arrays = load_split_arrays(cfg, ("val", "test"), meta["use_preprocess"], meta.get("grayscale", False))
    rep = {}
    for split in ("val", "test"):
        x, y, d = arrays[split]
        probs = torch.softmax(torch.from_numpy(predict_logits(model, x, device)), 1).numpy()
        pred = probs.argmax(1)
        r = classification_metrics(y, pred, classes, reject)
        conf = probs.max(1)
        acc_mask = conf >= unc_t
        r["uncertain_fraction"] = float(1 - acc_mask.mean())
        r["accepted_accuracy"] = float((pred[acc_mask] == y[acc_mask]).mean())
        rep[split] = r
        (out / f"classifier_{split}_metrics.json").write_text(json.dumps(r, indent=2))
        plot_confusion(np.array(r["confusion_matrix"]), classes,
                       f"Classifier confusion matrix - {split.upper()} (n={r['n']}, row-normalised colour)",
                       out / f"confusion_matrix_{split}.png")
        pd.DataFrame({"path": d["path"], "raw_class": d["raw_class"], "true": d["target"],
                      "pred": [classes[i] for i in pred], "prob": conf.round(5)}).to_csv(
            out / f"classifier_{split}_predictions.csv", index=False)
        if split == "test":
            groups = pd.to_datetime(d["time"]).dt.strftime("%Y-%m-%d").to_numpy()
            per, mac = bootstrap_f1(y, pred, groups, len(classes), cfg["evaluation"]["bootstrap_resamples"], seed)
            ci = pd.DataFrame({
                "f1": [r["per_class"][c]["f1"] for c in classes],
                "ci_low": np.nanpercentile(per, 2.5, axis=0), "ci_high": np.nanpercentile(per, 97.5, axis=0),
                "support": [r["per_class"][c]["support"] for c in classes],
                "test_days": [len(np.unique(groups[y == i])) for i in range(len(classes))],
            }, index=classes)
            ci.loc["MACRO (8 classes)"] = [r["macro_f1"], np.percentile(mac, 2.5), np.percentile(mac, 97.5),
                                           r["n"], len(np.unique(groups))]
            ci.index.name = "class"
            ci.to_csv(out / "classifier_test_f1_ci.csv")
            plot_f1_ci(ci, "Per-class F1 on TEST with 95% day-block bootstrap CI", out / "f1_ci_test.png")
            sub = d.assign(pred=[classes[i] for i in pred])
            dia = sub[sub["target"] == "diatom"].groupby("raw_class").apply(
                lambda g: pd.Series({"test_images": len(g), "recall_as_diatom": (g["pred"] == "diatom").mean()}),
                include_groups=False)
            dia.to_csv(out / "diatom_test_subtypes.csv")
            S["diatom_subtypes"] = dia
            S["ci"] = ci

    # ---------------- 2. open set ----------------
    root = resolve_path(cfg, "raw_data") / "zoolake" / cfg["data"]["zoolake"]["images_subdir"]
    od = pd.read_csv(resolve_path(cfg, "processed_data") / "open_set.csv")
    ox = build_array(od, root, meta["image_size"], meta["preprocess"],
                     resolve_path(cfg, "processed_data") / "cache", "openset", meta.get("grayscale", False))
    oprob = torch.softmax(torch.from_numpy(predict_logits(model, ox, device)), 1).numpy()
    od = od.assign(pred=[classes[i] for i in oprob.argmax(1)], prob=oprob.max(1))
    od["outcome"] = np.where(od["prob"] < unc_t, UNCERTAIN, od["pred"])
    tab = pd.crosstab(od["raw_class"], od["outcome"], normalize="index").mul(100).round(1)
    tab.insert(0, "images", od["raw_class"].value_counts())
    tab = tab.sort_values("images", ascending=False)
    total = pd.crosstab(np.ones(len(od)), od["outcome"], normalize="index").mul(100).round(1)
    total.index = ["ALL open-set"]
    total.insert(0, "images", len(od))
    tab = pd.concat([tab, total]).fillna(0)
    tab.index.name = "raw_class"
    tab.to_csv(out / "open_set_outcomes.csv")
    excl_cyano = od[od["raw_class"] != "maybe_cyano"]
    S["open_set"] = {"n": len(od), "uncertain_pct": float((od["outcome"] == UNCERTAIN).mean() * 100),
                     "reject_pct": float((od["outcome"] == reject).mean() * 100),
                     "organism_pct": float(od["outcome"].isin(organisms).mean() * 100),
                     "n_excl_cyano": len(excl_cyano),
                     "uncertain_pct_excl_cyano": float((excl_cyano["outcome"] == UNCERTAIN).mean() * 100),
                     "organism_pct_excl_cyano": float(excl_cyano["outcome"].isin(organisms).mean() * 100)}

    # ---------------- 3. detector on SYNTHETIC frames ----------------
    ultralytics_offline()
    from ultralytics import YOLO
    from src.pipeline.pipeline import Pipeline
    synth = resolve_path(cfg, "synthetic_data")
    det = YOLO(str(resolve_path(cfg, "models") / "detector_best.pt"))
    drows = []
    for split in ("val", "test"):
        m = det.val(data=str(synth / "data.yaml"), split=split, iou=th["nms_iou"], conf=0.001,
                    imgsz=cfg["detector"]["image_size"], batch=cfg["detector"]["batch_size"],
                    plots=False, verbose=False, workers=0)
        drows.append({"split": f"synthetic {split}", "mAP@0.5": float(m.box.map50), "mAP@0.5:0.95": float(m.box.map)})
    dmet = pd.DataFrame(drows).set_index("split")

    # ---------------- 4. end-to-end pipeline on SYNTHETIC TEST frames ----------------
    pipe = Pipeline(cfg)
    gt = json.loads((synth / "gt_test.json").read_text())
    cols = organisms + [REJECT]
    gt_counts, pr_counts = [], []
    tp = n_pred = n_gt = 0
    matched_total = matched_correct = matched_uncertain = 0
    for name, objs in gt.items():
        img = cv2.cvtColor(cv2.imread(str(synth / "images" / "test" / f"{name}.jpg")), cv2.COLOR_BGR2RGB)
        dets = pipe.run(img)
        c = count_detections(dets, organisms)
        pr_counts.append({**{k: c[k] for k in cols}, UNCERTAIN: c[UNCERTAIN]})
        g = pd.Series([o["target"] for o in objs]).value_counts()
        gt_counts.append({k: int(g.get(k, 0)) for k in cols})
        gb = np.array([o["box"] for o in objs]).reshape(-1, 4)
        pb = np.array([d.box for d in dets]).reshape(-1, 4)
        ps = np.array([d.det_conf for d in dets])
        pairs = greedy_match(pb, ps, gb, cfg["detector"]["match_iou"])
        tp += len(pairs); n_pred += len(dets); n_gt += len(objs)
        for i, j in pairs:
            matched_total += 1
            if dets[i].status == UNCERTAIN:
                matched_uncertain += 1
            elif dets[i].label == objs[j]["target"]:
                matched_correct += 1
    P = pd.DataFrame(pr_counts)
    G = pd.DataFrame(gt_counts)
    errs = count_errors(P[cols], G[cols])
    tot_pred = P[organisms].sum(1)
    tot_true = G[organisms].sum(1)
    errs.loc["ALL organisms"] = [tot_true.mean(), tot_pred.mean(), (tot_pred - tot_true).abs().mean(),
                                 (tot_pred - tot_true).mean()]
    errs.index.name = "class"
    errs.to_csv(out / "count_errors_test.csv")
    plot_count_errors(errs, "End-to-end count error per frame - SYNTHETIC TEST frames", out / "count_errors_test.png")
    dp, dr, df1 = detection_prf(tp, n_pred, n_gt)
    e2e = {"frames": len(gt), "gt_objects": n_gt, "detections": n_pred, "det_precision": dp, "det_recall": dr,
           "det_f1": df1, "uncertain_per_frame": float(P[UNCERTAIN].mean()),
           "matched_objects": matched_total,
           "matched_uncertain_pct": 100 * matched_uncertain / max(1, matched_total),
           "matched_confident_accuracy": matched_correct / max(1, matched_total - matched_uncertain)}
    (out / "end_to_end_test.json").write_text(json.dumps(e2e, indent=2))

    # ---------------- 5. ablation + tables ----------------
    collect_ablation()
    abl = pd.read_csv(out / "ablation.csv")
    imb = pd.read_csv(out / "class_imbalance.csv", index_col=0)
    syn = pd.read_csv(out / "synthetic_stats.csv")
    val, test = rep["val"], rep["test"]
    S.update(val=val, test=test, dmet=dmet, e2e=e2e, errs=errs, abl=abl)

    per = pd.DataFrame({
        "val P": [val["per_class"][c]["precision"] for c in classes],
        "val R": [val["per_class"][c]["recall"] for c in classes],
        "val F1": [val["per_class"][c]["f1"] for c in classes],
        "test P": [test["per_class"][c]["precision"] for c in classes],
        "test R": [test["per_class"][c]["recall"] for c in classes],
        "test F1": [test["per_class"][c]["f1"] for c in classes],
        "test F1 95% CI": [f"{S['ci'].loc[c, 'ci_low']:.3f}-{S['ci'].loc[c, 'ci_high']:.3f}" for c in classes],
        "test n": [test["per_class"][c]["support"] for c in classes],
        "test days": [int(S["ci"].loc[c, "test_days"]) for c in classes],
    }, index=pd.Index(classes, name="class")).round(3)
    summ = pd.DataFrame({
        "accuracy": [val["accuracy"], test["accuracy"]],
        "macro F1 (8 classes)": [val["macro_f1"], test["macro_f1"]],
        "macro F1 (7 organisms)": [val["macro_f1_organisms"], test["macro_f1_organisms"]],
        "uncertain fraction": [val["uncertain_fraction"], test["uncertain_fraction"]],
        "accuracy of accepted": [val["accepted_accuracy"], test["accepted_accuracy"]],
    }, index=pd.Index(["val", "test"], name="split")).round(4)
    cls_init = "ImageNet-pretrained" if meta.get("pretrained") else "from scratch"
    det_run = (resolve_path(cfg, "models") / "detector_run.txt").read_text().strip()
    abl_det = pd.read_csv(out / "ablation_detector.csv")
    det_init = abl_det.set_index("run").loc[det_run, "init"]
    det_init = "COCO-pretrained" if det_init.startswith("COCO") else "from scratch"
    abl_show = abl[["experiment", "init", "grayscale", "preprocess", "class_weighting", "epochs_budget", "best_epoch",
                    "val_accuracy", "val_macro_f1", "val_macro_f1_organisms"]].round(4)
    mac_ci = S["ci"].loc["MACRO (8 classes)"]

    lines = [
        "<!-- generated by scripts/evaluate.py; do not edit by hand -->",
        f"_Generated by `python -m scripts.evaluate` (run `{run_dir.name}`). All numbers come from files in `results/`._",
        "",
        "### Data",
        md(imb[["train_images", "val_images", "test_images", "total_images"]].rename_axis("class")),
        "",
        f"Imbalance ratio (largest : smallest class): {imb['total_images'].max() / imb['total_images'].min():.1f} : 1.",
        "",
        "### Classifier ablations (VAL only, single seed, EfficientNet-B0)",
        md(abl_show, index=False),
        "",
        "### Detector ablation: from scratch vs COCO-pretrained (SYNTHETIC VAL frames, Ultralytics default NMS IoU 0.7)",
        md(abl_det.drop(columns=["run"]).round(4), index=False),
        "",
        "The from-scratch detector run was stopped by the user after 7 epochs; compare at the common epoch as well as at best.",
        "",
        f"Final classifier: `{meta['run']}` ({cls_init}, grayscale: {meta.get('grayscale', False)}, preprocessing: {meta['use_preprocess']}, class weighting: "
        f"{meta['class_weighting']}, best epoch {meta['epoch']}).",
        "",
        "### Classifier: overall",
        md(summ),
        "",
        f"Uncertainty threshold **{unc_t:.2f}** (max softmax), chosen on VAL as the smallest threshold whose "
        f"accepted predictions reach {th['uncertainty_target_accuracy']:.0%} accuracy.",
        f"Test macro F1 (8 classes) 95% day-block bootstrap CI: {mac_ci['ci_low']:.3f}-{mac_ci['ci_high']:.3f}.",
        "",
        "### Classifier: per class (val and test)",
        md(per),
        "",
        "**Diatom caveat:** diatom test images come from few days and are mostly *fragilaria*. The asterionella "
        "bloom day (64% of all diatoms) is in train. Recall by subtype on test:",
        "",
        md(S["diatom_subtypes"].round(3)),
        "",
        "### Open set (10 classes never trained on)",
        f"At the val-chosen threshold, {S['open_set']['uncertain_pct']:.1f}% of {S['open_set']['n']} open-set images "
        f"are flagged uncertain, {S['open_set']['reject_pct']:.1f}% are confidently called reject, and "
        f"{S['open_set']['organism_pct']:.1f}% are confidently assigned to one of the 7 organism classes "
        f"(excluding maybe_cyano, whose labels are themselves uncertain: {S['open_set']['uncertain_pct_excl_cyano']:.1f}% "
        f"uncertain, {S['open_set']['organism_pct_excl_cyano']:.1f}% wrongly counted as an organism class).",
        "",
        md(tab),
        "",
        f"### Detector (YOLOv8n, {det_init}, run `{det_run}`; measured on SYNTHETIC frames only)",
        md(dmet.round(4)),
        "",
        f"Tuned on synthetic VAL frames: NMS IoU {th['nms_iou']}, confidence {th['det_conf']:.2f}.",
        "",
        "Synthetic frame sets:",
        md(syn, index=False),
        "",
        "### End-to-end counting (SYNTHETIC TEST frames)",
        f"{e2e['frames']} frames, {e2e['gt_objects']} pasted objects, {e2e['detections']} detections; detection "
        f"P {e2e['det_precision']:.3f} / R {e2e['det_recall']:.3f} / F1 {e2e['det_f1']:.3f} at IoU >= "
        f"{cfg['detector']['match_iou']}. Of the matched objects, {e2e['matched_uncertain_pct']:.1f}% were "
        f"flagged uncertain, and the confident ones were classified with accuracy "
        f"{e2e['matched_confident_accuracy']:.3f}. Mean uncertain items per frame: {e2e['uncertain_per_frame']:.2f}.",
        "",
        md(errs.round(3)),
    ]
    diag_path = out / "frame_crop_diagnostic.csv"
    if diag_path.exists():
        lines += ["", "### Diagnostic: why so many frame detections are uncertain (SYNTHETIC VAL frames, ground-truth boxes)",
                  "Same objects, classified as crops re-cut from the synthetic frames vs as their original ZooLake crops "
                  "(`python -m scripts.diagnose_frame_crops`, VAL only):", "",
                  md(pd.read_csv(diag_path).round(4), index=False)]
    table_md = "\n".join(lines) + "\n"
    (out / "results_tables.md").write_text(table_md, encoding="utf-8")
    (run_dir / "results_tables.md").write_text(table_md, encoding="utf-8")

    readme = resolve_path(cfg, "results").parent / "README.md"
    text = readme.read_text(encoding="utf-8")
    text = re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->",
                  lambda _: "<!-- RESULTS:START -->\n" + table_md + "<!-- RESULTS:END -->", text, flags=re.S)
    readme.write_text(text, encoding="utf-8")

    notes_path = resolve_path(cfg, "results").parent / "docs" / "final_notes.md"
    notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
    headline = [
        "# Final summary",
        "",
        f"_Numbers generated by `python -m scripts.evaluate` (run `{run_dir.name}`); narrative in `docs/final_notes.md`._",
        "",
        "## Headline numbers",
        "",
        f"- Classifier (EfficientNet-B0, {cls_init}), TEST: accuracy {F(test['accuracy'])}, "
        f"macro F1 (8 classes) {F(test['macro_f1'])} [95% CI {mac_ci['ci_low']:.3f}-{mac_ci['ci_high']:.3f}], "
        f"macro F1 (7 organisms) {F(test['macro_f1_organisms'])}.",
        f"- Same classifier on VAL: accuracy {F(val['accuracy'])}, macro F1 {F(val['macro_f1'])}.",
        f"- Uncertainty threshold {unc_t:.2f} (VAL): test uncertain fraction {F(test['uncertain_fraction'], 3)}, "
        f"accuracy of accepted {F(test['accepted_accuracy'])}.",
        f"- Detector (YOLOv8n, {det_init}) on SYNTHETIC test frames: mAP@0.5 "
        f"{F(dmet.loc['synthetic test', 'mAP@0.5'])}, mAP@0.5:0.95 {F(dmet.loc['synthetic test', 'mAP@0.5:0.95'])}.",
        f"- End-to-end on SYNTHETIC test frames: total organism count MAE {F(errs.loc['ALL organisms', 'MAE'], 2)} "
        f"per frame (mean true {F(errs.loc['ALL organisms', 'mean_true_per_frame'], 2)}), mean signed error "
        f"{F(errs.loc['ALL organisms', 'mean_signed_error'], 2)}.",
        f"- Open set: {S['open_set']['uncertain_pct']:.1f}% flagged uncertain, {S['open_set']['organism_pct']:.1f}% "
        f"wrongly counted as an organism class.",
        "",
        "Full tables: `results/results_tables.md` (also embedded in the README).",
        "",
    ]
    (out / "final_summary.md").write_text("\n".join(headline) + "\n" + notes, encoding="utf-8")
    print(table_md)


if __name__ == "__main__":
    main()
