# PlanktoScan AI

Detects, classifies and counts plankton in microscopy images. Computer Vision end-semester project
based on SIH25043 (software and AI part).

**Pipeline:** image → YOLOv8n object detector → crop → preprocessing (illumination correction,
denoising, CLAHE) → EfficientNet-B0 classifier with confidence → per-class counts (low-confidence items
flagged **uncertain**, `reject` items listed separately) → annotated image + CSV / HTML / PDF report.

> **Read the limitations first.** The models were trained on a freshwater dataset of single-organism
> crops. The final classifier is fine-tuned from ImageNet weights and the detector from COCO weights
> (`yolov8n.pt`); from-scratch runs are kept as ablations.
> The detector was trained and evaluated only on **synthetic** frames. Results on real
> microscope images are **not validated**.

## Data and important limitations

- **Source and license.** ZooLake, Eawag, DOI [10.25678/0004DY](https://doi.org/10.25678/0004DY), from
  Kyathanahally et al. (2021), *Deep Learning Classification of Lake Zooplankton*, Frontiers in
  Microbiology 12:746297, [doi:10.3389/fmicb.2021.746297](https://doi.org/10.3389/fmicb.2021.746297).
  Only `data.zip` (492 MB, SHA-256 `ce038e19…1c383d4f`, checked on download) is used.
  **The license is unclear:** the record's license field is empty, but it is marked as open data and the
  paper calls the data "freely available". We use it for non-commercial academic work and cite the
  DOI and the paper. The images are not redistributed in this repo; `python -m scripts.download_zoolake`
  downloads them from the source.
- **Dataset: ZooLake.** It is a **freshwater** (lake) plankton dataset, not marine. SIH25043 is framed
  around marine plankton, so species and imaging conditions will differ from marine samples. A
  model trained here should be retrained or fine-tuned on marine data before real marine use.
- **ZooLake has only single-organism crops, not full frames with bounding boxes.** The classifier is
  trained and tested on real crops. The detector is trained on **synthetic multi-organism frames**
  made by alpha-blending crops onto procedural microscopy-style backgrounds. Each split's frames use
  only that split's crops.
- **Detector mAP is measured on synthetic frames.** It shows how well the detector finds pasted
  organisms. It is not a measure of performance on real full-field microscopy images.

### Classes (Phase 2)

Integrity check: 17,943 images in 35 raw classes, 0 unreadable. Of the 25 mapped raw classes,
14,048 images go into 8 project classes. Five byte-identical files are removed before the split:
two images filed under two conflicting classes (both copies dropped, 4 rows) and one repeated
image (1 row). That leaves **14,043 images**.

| Class | Raw ZooLake classes | Images |
|---|---|---|
| dinobryon | dinobryon | 3,319 |
| rotifer | asplanchna, brachionus, conochilus, kellicottia, keratella_cochlearis, keratella_quadrata, polyarthra, rotifers, synchaeta, trichocerca | 3,278 |
| cladoceran | daphnia, diaphanosoma, bosmina | 1,890 |
| nauplius | nauplius | 1,507 |
| copepod | cyclops, eudiaptomus | 1,403 |
| diatom | asterionella, fragilaria, diatom_chain | 1,377 |
| ceratium | ceratium | 814 |
| **reject** | dirt, unknown, copepod_skins, daphnia_skins | 455 |

- **`reject`** means *not countable as a target organism*: debris (`dirt`), moulted exoskeletons
  (`*_skins`), and objects the taxonomists could not place (`unknown`). Per the dataset README,
  `unknown` may include fragments of zooplankton or algae, so `reject` is **not** a clean
  "non-organism" class.
- **Open set (3,895 images, never trained on or used for tuning):** maybe_cyano, unknown_plankton,
  uroglena, paradileptus, filament, aphanizomenon, fish, leptodora, hydra, chaoborus. It measures
  how the system behaves on organisms it was never taught.
- Imbalance ratio: 332 : 1 across raw classes, 7.3 : 1 after mapping. See
  `results/class_imbalance.png` and `results/split_report.md`.

### Split (leakage-safe)

- 70/15/15 over **whole calendar days**: no day's images are split across train/val/test. Stratified
  jointly on project class **and** raw subtype, seed 42. A class may fall back to hour blocks only if
  its val/test cannot be filled and hour blocks actually help.
- Leakage check: pHash candidates (Hamming distance ≤ 4), then confirmation by pixel correlation
  (≥ 0.98). Result: 0 confirmed near-duplicates across splits. pHash alone is unreliable on these
  small dark-field crops: it flagged about 4,600 candidate pairs, mostly different-class images
  months apart.
- **Diatom caveat:** 64% of diatom images come from one bloom day (2019-06-02, asterionella), which
  must go to train. Diatom test has only 64 images from 9 days, mostly *fragilaria*. Diatom test
  F1 is therefore a weaker estimate than for the other classes.
- The test split is not looked at until the final evaluation (Phase 8).

## Project layout

```
config.yaml          all paths, classes, seed, hyperparameters and decisions
src/common/          config loader, seeding, experiment logging (runs/<timestamp>_<name>/)
src/data/            download, inventory, class mapping, dedupe, day-block split, leakage check
src/preprocess/      illumination correction, denoising, CLAHE
src/classify/        letterbox + cache, GPU augmentation, EfficientNet training, metrics
src/detect/          synthetic frame generator, YOLO helpers (offline mode, matching)
src/pipeline/        detect -> crop -> classify -> count; uncertain/reject logic; CSV/HTML/PDF reports
src/evaluation.py    bootstrap CIs, count errors, figures
app/                 Streamlit UI
scripts/             command-line entry points (python -m scripts.<name>)
tests/               pytest tests
notebooks/           thin Colab backup notebook
docs/viva_notes.md   one-line justification for each design decision and ablation
results/             generated tables and figures (committed, except figures that contain dataset images)
data/ models/ runs/  generated, git-ignored
```

## Setup (Windows, Python 3.12)

```powershell
cd C:\Users\eswar\planktoscan-ai
uv venv --python 3.12 .venv            # or: py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # check pytorch.org for your driver
pip install -r requirements.txt        # exact versions used: requirements-lock.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pytest -q
```

Colab backup: `notebooks/planktoscan_colab.ipynb`. It clones the repo and runs the same scripts.

## Usage

Full reproduction, in order. Every script logs its config, seed, git commit and versions to `runs/`.

```powershell
python -m scripts.download_zoolake        # 492 MB, SHA-256 verified, safe extract
python -m scripts.inventory_zoolake       # integrity check + raw class counts
python -m scripts.make_splits             # mapping, dedupe, day-block split, leakage check, imbalance report
python -m scripts.compare_preprocess      # results/preprocess_comparison.png
# Phase 3 ablation (val macro F1 decides classifier.use_preprocess in config.yaml)
python -m scripts.train_classifier --name abl_pre0 --use-preprocess 0 --weighting none --epochs 25
python -m scripts.train_classifier --name abl_pre1 --use-preprocess 1 --weighting none --epochs 25
# Phase 4 weighting experiment + selection by val macro F1
python -m scripts.train_classifier --name cls_w0 --weighting none
python -m scripts.train_classifier --name cls_w1 --weighting inverse_sqrt_freq
# Pretrained weights (exactly two files, SHA-256 checked) + ImageNet fine-tuning
python -m scripts.fetch_pretrained
python -m scripts.train_classifier --name cls_pt_rgb --pretrained 1 --grayscale 0 --weighting none
python -m scripts.train_classifier --name cls_pt_gray --pretrained 1 --grayscale 1 --weighting none
python -m scripts.select_classifier --prefix cls_
python -m scripts.choose_threshold        # uncertainty threshold on VAL
# Phase 5 detector
python -m scripts.make_synthetic          # 2000/300/300 frames, per-split crops only
python -m scripts.train_detector --pretrained 1 --time-hours 0.667 --name detector_pt   # from yolov8n.pt, 40 min cap
python -m scripts.collect_ablation        # from scratch vs pretrained tables (VAL)
python -m scripts.tune_detector           # NMS IoU + confidence on synthetic VAL frames
# Phase 8: every table and figure + this README's results section
python -m scripts.diagnose_frame_crops   # VAL-only crop vs frame-crop diagnostic
python -m scripts.evaluate
# App
streamlit run app/streamlit_app.py
```

Resume an interrupted classifier run: `python -m scripts.train_classifier --resume runs/<run_dir>`.

## Results

The classifier is judged by per-class F1 and macro F1, because the classes are imbalanced. All tuning
(preprocessing, class weighting, epochs, uncertainty threshold, NMS/confidence) used the **validation**
split only. The test split and the synthetic test frames were used only by `scripts/evaluate.py`.
The section below is generated by that script; do not edit it by hand.

<!-- RESULTS:START -->
<!-- generated by scripts/evaluate.py; do not edit by hand -->
_Generated by `python -m scripts.evaluate` (run `20261112_122056_evaluate`). All numbers come from files in `results/`._

### Data
| class      |   train_images |   val_images |   test_images |   total_images |
|:-----------|---------------:|-------------:|--------------:|---------------:|
| dinobryon  |           2315 |          489 |           515 |           3319 |
| rotifer    |           2471 |          415 |           392 |           3278 |
| cladoceran |           1326 |          281 |           283 |           1890 |
| nauplius   |           1054 |          228 |           225 |           1507 |
| copepod    |            986 |          210 |           207 |           1403 |
| diatom     |           1093 |          220 |            64 |           1377 |
| ceratium   |            568 |          123 |           123 |            814 |
| reject     |            333 |           67 |            55 |            455 |

Imbalance ratio (largest : smallest class): 7.3 : 1.

### Classifier ablations (VAL only, single seed, EfficientNet-B0)
| experiment   | init         | grayscale   | preprocess   | class_weighting   |   epochs_budget |   best_epoch |   val_accuracy |   val_macro_f1 |   val_macro_f1_organisms |
|:-------------|:-------------|:------------|:-------------|:------------------|----------------:|-------------:|---------------:|---------------:|-------------------------:|
| abl_pre0     | from scratch | False       | False        | none              |              25 |           23 |         0.9115 |         0.8608 |                   0.9161 |
| abl_pre1     | from scratch | False       | True         | none              |              25 |           23 |         0.9149 |         0.8621 |                   0.9223 |
| cls_w0       | from scratch | False       | True         | none              |              40 |           37 |         0.9518 |         0.9055 |                   0.9576 |
| cls_w1       | from scratch | False       | True         | inverse_sqrt_freq |              40 |           37 |         0.9469 |         0.9051 |                   0.9549 |
| cls_pt_rgb   | ImageNet     | False       | True         | none              |              40 |           29 |         0.9823 |         0.9593 |                   0.9895 |
| cls_pt_gray  | ImageNet     | True        | True         | none              |              40 |           28 |         0.9784 |         0.959  |                   0.9827 |

### Detector ablation: from scratch vs COCO-pretrained (SYNTHETIC VAL frames, Ultralytics default NMS IoU 0.7)
| init              |   time_cap_h |   epochs_trained | stopped_early_by_user   |   best_epoch |   best_val_mAP50 |   best_val_mAP50_95 |   common_epoch |   val_mAP50_at_common_epoch |   val_mAP50_95_at_common_epoch |
|:------------------|-------------:|-----------------:|:------------------------|-------------:|-----------------:|--------------------:|---------------:|----------------------------:|-------------------------------:|
| from scratch      |        1.25  |                7 | True                    |            7 |           0.937  |              0.6825 |              7 |                      0.937  |                         0.6825 |
| COCO (yolov8n.pt) |        0.667 |               63 | False                   |           63 |           0.9836 |              0.8402 |              7 |                      0.9666 |                         0.7669 |

The from-scratch detector run was stopped by the user after 7 epochs; compare at the common epoch as well as at best.

Final classifier: `20261112_104503_cls_pt_rgb` (ImageNet-pretrained, grayscale: False, preprocessing: True, class weighting: none, best epoch 29).

### Classifier: overall
| split   |   accuracy |   macro F1 (8 classes) |   macro F1 (7 organisms) |   uncertain fraction |   accuracy of accepted |
|:--------|-----------:|-----------------------:|-------------------------:|---------------------:|-----------------------:|
| val     |     0.9823 |                 0.9593 |                   0.9895 |               0.0502 |                 0.9902 |
| test    |     0.9839 |                 0.9623 |                   0.9888 |               0.0376 |                 0.995  |

Uncertainty threshold **0.84** (max softmax), chosen on VAL as the smallest threshold whose accepted predictions reach 99% accuracy.
Test macro F1 (8 classes) 95% day-block bootstrap CI: 0.938-0.986.

### Classifier: per class (val and test)
| class      |   val P |   val R |   val F1 |   test P |   test R |   test F1 | test F1 95% CI   |   test n |   test days |
|:-----------|--------:|--------:|---------:|---------:|---------:|----------:|:-----------------|---------:|------------:|
| copepod    |   0.981 |   1     |    0.991 |    0.99  |    0.986 |     0.988 | 0.976-0.998      |      207 |          47 |
| diatom     |   0.991 |   1     |    0.995 |    0.984 |    0.969 |     0.976 | 0.842-1.000      |       64 |           9 |
| nauplius   |   0.987 |   0.982 |    0.985 |    0.987 |    0.987 |     0.987 | 0.971-0.997      |      225 |          46 |
| cladoceran |   0.989 |   0.993 |    0.991 |    0.99  |    1     |     0.995 | 0.987-1.000      |      283 |          52 |
| rotifer    |   0.972 |   0.993 |    0.982 |    0.97  |    0.99  |     0.98  | 0.956-0.999      |      392 |          73 |
| ceratium   |   0.992 |   0.984 |    0.988 |    1     |    1     |     1     | 1.000-1.000      |      123 |          24 |
| dinobryon  |   0.998 |   0.992 |    0.995 |    0.998 |    0.994 |     0.996 | 0.989-1.000      |      515 |          14 |
| reject     |   0.821 |   0.687 |    0.748 |    0.833 |    0.727 |     0.777 | 0.654-0.923      |       55 |           8 |

**Diatom caveat:** diatom test images come from few days and are mostly *fragilaria*. The asterionella bloom day (64% of all diatoms) is in train. Recall by subtype on test:

| raw_class    |   test_images |   recall_as_diatom |
|:-------------|--------------:|-------------------:|
| asterionella |             8 |                  1 |
| diatom_chain |             2 |                  0 |
| fragilaria   |            54 |                  1 |

### Open set (10 classes never trained on)
At the val-chosen threshold, 71.9% of 3895 open-set images are flagged uncertain, 18.2% are confidently called reject, and 9.9% are confidently assigned to one of the 7 organism classes (excluding maybe_cyano, whose labels are themselves uncertain: 83.5% uncertain, 13.1% wrongly counted as an organism class).

| raw_class        |   images |   ceratium |   cladoceran |   copepod |   diatom |   dinobryon |   nauplius |   reject |   rotifer |   uncertain |
|:-----------------|---------:|-----------:|-------------:|----------:|---------:|------------:|-----------:|---------:|----------:|------------:|
| maybe_cyano      |     1364 |        0.1 |          0   |       0   |      3.5 |         0.4 |        0   |     45.5 |       0   |        50.4 |
| uroglena         |      953 |        0   |          0   |       0   |      0   |         0   |        0   |      0.2 |       5.1 |        94.6 |
| paradileptus     |      424 |        0.2 |          0   |       1.4 |      0   |         0   |        0   |      2.1 |       9.9 |        86.3 |
| filament         |      405 |        0   |          0   |       2.5 |     10.4 |         0   |        0   |      4   |       1.7 |        81.5 |
| aphanizomenon    |      225 |        0.4 |          0   |       0.4 |      0   |         0   |        0   |      4   |      25.3 |        69.8 |
| fish             |      222 |        0   |          1.4 |      13.5 |      1.8 |         0.5 |        0   |      7.7 |       0   |        75.2 |
| leptodora        |      203 |        0   |         17.7 |       6.9 |      0   |         0   |        0   |      7.9 |       0.5 |        67   |
| unknown_plankton |       71 |        0   |          2.8 |       8.5 |      0   |         0   |        1.4 |     25.4 |       5.6 |        56.3 |
| hydra            |       18 |        0   |          0   |      44.4 |      0   |         0   |        0   |      0   |       0   |        55.6 |
| chaoborus        |       10 |        0   |          0   |      50   |      0   |         0   |        0   |      0   |       0   |        50   |
| ALL open-set     |     3895 |        0.1 |          1.1 |       2.1 |      2.4 |         0.2 |        0   |     18.2 |       4.1 |        71.9 |

### Detector (YOLOv8n, COCO-pretrained, run `20261112_111516_detector_pt`; measured on SYNTHETIC frames only)
| split          |   mAP@0.5 |   mAP@0.5:0.95 |
|:---------------|----------:|---------------:|
| synthetic val  |    0.9826 |         0.8438 |
| synthetic test |    0.9782 |         0.8135 |

Tuned on synthetic VAL frames: NMS IoU 0.75, confidence 0.50.

Synthetic frame sets:
| split   |   frames |   objects |   crops_available |   unique_crops_used |
|:--------|---------:|----------:|------------------:|--------------------:|
| train   |     2000 |     17987 |             10146 |                8441 |
| val     |      300 |      2681 |              2033 |                1515 |
| test    |      300 |      2745 |              1864 |                1452 |

### End-to-end counting (SYNTHETIC TEST frames)
300 frames, 2745 pasted objects, 2679 detections; detection P 0.957 / R 0.934 / F1 0.945 at IoU >= 0.5. Of the matched objects, 50.1% were flagged uncertain, and the confident ones were classified with accuracy 0.945. Mean uncertain items per frame: 4.50.

| class         |   mean_true_per_frame |   mean_pred_per_frame |   MAE |   mean_signed_error |
|:--------------|----------------------:|----------------------:|------:|--------------------:|
| copepod       |                 1.013 |                 0.633 | 0.413 |              -0.38  |
| diatom        |                 0.307 |                 0.243 | 0.07  |              -0.063 |
| nauplius      |                 1.133 |                 0.497 | 0.643 |              -0.637 |
| cladoceran    |                 1.52  |                 1.127 | 0.5   |              -0.393 |
| rotifer       |                 1.813 |                 0.9   | 0.92  |              -0.913 |
| ceratium      |                 0.59  |                 0.233 | 0.363 |              -0.357 |
| dinobryon     |                 2.53  |                 0.467 | 2.063 |              -2.063 |
| reject        |                 0.243 |                 0.333 | 0.31  |               0.09  |
| ALL organisms |                 8.907 |                 4.1   | 4.813 |              -4.807 |

### Diagnostic: why so many frame detections are uncertain (SYNTHETIC VAL frames, ground-truth boxes)
Same objects, classified as crops re-cut from the synthetic frames vs as their original ZooLake crops (`python -m scripts.diagnose_frame_crops`, VAL only):

| input                   |    n |   accuracy |   uncertain_fraction |   accepted_accuracy |   median_max_prob |
|:------------------------|-----:|-----------:|---------------------:|--------------------:|------------------:|
| frame crop, padding 0.0 | 2681 |     0.7162 |               0.5528 |              0.9083 |            0.8202 |
| frame crop, padding 0.1 | 2681 |     0.7602 |               0.4931 |              0.9169 |            0.8428 |
| frame crop, padding 0.3 | 2681 |     0.7452 |               0.5323 |              0.8939 |            0.8254 |
| frame crop, padding 0.6 | 2681 |     0.5983 |               0.6882 |              0.7943 |            0.7321 |
| frame crop, padding 1.0 | 2681 |     0.3626 |               0.8239 |              0.553  |            0.599  |
| original ZooLake crop   | 2681 |     0.9836 |               0.047  |              0.9902 |            0.9047 |
<!-- RESULTS:END -->

## Reproducibility
- Fixed seed 42 (`project.seed`) for Python, NumPy, PyTorch, the split and the frame generator; cuDNN
  deterministic mode for the classifier. YOLO training uses the same seed with `deterministic=False`
  for speed, so detector numbers can differ slightly between reruns.
- Each run writes `config.yaml` and `environment.json` (seed, git commit, library versions, command) to
  `runs/<timestamp>_<name>/`.
- Single seed per experiment: differences of about 1 macro-F1 point between ablation arms are within noise.

## Known limitations
- **Freshwater data, one lake, one camera.** ZooLake comes from Lake Greifensee and the Dual Scripps
  Plankton Camera (dark field). SIH25043 targets marine plankton. Other microscopes, magnifications,
  bright-field illumination and marine species are all out of domain.
- **Domain shift to real microscope images.** No real multi-organism frames with labels exist in
  ZooLake, so the full pipeline has never been validated on a real microscope image.
- **Synthetic detector evaluation.** Detector mAP and end-to-end count errors are measured on
  synthetic frames that we generated ourselves. They are optimistic upper bounds for real images.
- **Pretrained on natural images.** The final EfficientNet-B0 starts from ImageNet weights
  (`timm/efficientnet_b0.ra_in1k`) and YOLOv8n from COCO weights (`yolov8n.pt`); URLs, sizes and SHA-256 are in
  `results/pretrained_weights.json`. Neither source contains plankton. Pretraining raised val macro F1 from
  0.906 to 0.959; the from-scratch detector run was stopped early, so its comparison is only fair at the common epoch.
- **Threshold target changed once.** The uncertainty target was 97% accepted accuracy; the pretrained model
  already reaches 98.2% on val, which made the threshold 0 (nothing flagged), so the target was raised
  to 99% on VAL before the test split was used.
- **Crop-to-frame gap (main weakness).** On synthetic frames about half of the detected organisms are
  flagged uncertain, so per-class counts undercount. The same objects as original ZooLake crops are
  classified almost perfectly. The classifier is sensitive to the crop's native surroundings, which pasting
  removes. See the diagnostic table in Results. Fix (not done): also train on frame crops.
- **Merged classes.** Classes such as `rotifer` (10 genera), `cladoceran` (3) and `diatom` (3) hide
  within-class variety. `reject` mixes debris, moulted skins and unidentifiable objects, possibly including
  organism fragments.
- **Bloom-day diatom issue.** 64% of diatoms come from one bloom day (in train). Diatom test has few
  images from few days and is mostly *fragilaria*, so its F1 is a weak estimate.
- **Open set.** Organisms from the 10 excluded classes are not known to the classifier. The
  uncertainty flag catches only part of them (see Results).
- **Unclear dataset license** (see Data section).
- **Single seed** per experiment; no repeated runs.
