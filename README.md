# PlanktoScan AI

Detects, classifies and counts plankton in dark-field microscopy images. A React web app talks to a FastAPI REST
backend; a command-line tool uses the same analysis service; every analysis is kept in an SQLite history with
CSV / JSON / HTML / PDF reports. B.Tech Computer Vision project based on
Smart India Hackathon problem SIH25043 (software and AI part).

> **Read the limitations first.** The models were trained on a **freshwater** dataset of single-organism crops
> (ZooLake). The detector was trained and evaluated only on **synthetic** frames. There are no labelled real
> microscope frames, so accuracy on real images is **unknown**. See [Limitations](#limitations).

## Problem statement

Counting and identifying plankton under a microscope is slow manual work, and the counts depend on who does
the counting. PlanktoScan AI takes a microscopy image and finds every object in it. It names each object
as one of 7 plankton groups (or rejects it as debris) and counts them. Low-confidence identifications are
flagged for a human, so they are never silently counted or dropped.

## Features

- **Detection + classification + counting** in one call: YOLOv8n finds objects, EfficientNet-B0 (v2) names them.
- **Four kinds of count, kept separate everywhere:**
  - **confirmed**: confident organism predictions only;
  - **estimated**: every detection under its most likely class, with the part that came from uncertain items shown;
  - **uncertain**: below the confidence threshold;
  - **rejected**: debris, skins, unidentifiable objects.
- **Images:** original, enhanced (illumination correction + denoising + CLAHE; display only) and annotated
  (`#id class probability`, colour = status).
- **Per-stage timing** (validation, detection, classification, counting, annotation, enhancement) with CUDA
  synchronisation, so GPU times are real.
- **Input validation with typed errors:** missing / corrupted / truncated files, unsupported formats (checked by
  extension and content), too small (< 32 px), too large (> 50 MB or > 40 MP, checked before decoding).
  An image with no detections is a successful result with a warning.
- **Warnings:** no organisms, large image, many uncertain detections (likely out of domain), non-default
  thresholds, threshold chosen for a different model.
- **Reports:** CSV, JSON, HTML (self-contained) and PDF, all rendered from one canonical result.
- **SQLite history** of every analysis, including failures. It stores metadata, counts and file paths
  (no image data), and each analysis gets its own output folder.
- **Web app** (React + Vite) on a **REST API** (FastAPI): upload with validation, original / enhanced / annotated views,
  live detection boxes linked to the detection table, counts, per-stage times, report downloads, history and
  previous analyses.
- **CLI** for single images or whole folders. The API and the CLI call the same Python service; no inference code
  is duplicated.

## System architecture

```
 frontend/ (React + Vite)          api/ (FastAPI)                   src/ (Python, AI/ML)
 +----------------------+  REST   +----------------------+        +-----------------------------+
 | upload + validation  | ------> | POST /api/analyses   |        | service.py AnalysisService  |
 | specimen frame       |  JSON   | GET  /api/analyses   | -----> |  validate, time, warn       |
 | counts, detections   | <------ | images, reports      |        |  pipeline/pipeline.py       |
 | history, downloads   |         | errors as JSON       |        |   YOLOv8n -> crop ->        |
 +----------------------+         +----------+-----------+        |   EfficientNet-B0 v2        |
                                             |                    |  pipeline/report.py         |
 scripts/analyze.py (CLI) -------------------+------------------> |  history.py (SQLite)        |
                                                                  +-----------------------------+
```

- Models are loaded **once** per process: the API loads them at start-up (about 20 s with warm-up), the CLI once per run.
- All inference goes through `AnalysisService` and `Pipeline.run`. The API and the CLI only call the service; a test
  checks that neither touches the models directly.
- One analysis runs on the GPU at a time (a lock in the API); uploads are read only up to the size limit.
- Thresholds come from `results/thresholds.json`. They were chosen on the validation split and are never tuned on test data.

| path | role |
|---|---|
| `src/service.py` | `AnalysisService.analyze(path)`, `analyze_bytes(data, name)`; typed errors; `save_outputs` |
| `src/pipeline/pipeline.py` | detector + classifier, crop, status rule, annotation |
| `src/pipeline/counting.py` | confirmed counts (`count_detections`) and `estimated_counts` |
| `src/pipeline/report.py` | the four report formats |
| `src/history.py` | SQLite history (add / list / get / delete / clear) |
| `api/main.py` | REST API (FastAPI); also serves the built frontend |
| `frontend/` | React + Vite web app |
| `scripts/analyze.py` | CLI |

## AI/ML pipeline

1. **Detection.** YOLOv8n, single class "object", input 640 px, confidence 0.50, NMS IoU 0.75 (both chosen on
   synthetic VAL frames).
2. **Crop.** Each box is grown by 10 % (`pad_crop`, the same function used to build the v2 training crops).
3. **Preprocessing per crop.** Border-median illumination correction, NL-means denoising, CLAHE, then letterbox
   to 128 px. This is the same `prepare_crop` used in training.
4. **Classification.** EfficientNet-B0 classifier v2 gives softmax probabilities over 7 organisms + `reject`.
   Crops are classified in a fixed batch of 32, padded if needed: on the RTX 3050, each *new* batch shape costs
   cuDNN 6-12 s of set-up. On 300 VAL frames (2,641 detections) the fixed batch changed 0 labels and 0 statuses
   compared with variable batches (max probability difference 2.9e-3).
5. **Status.** probability < 0.89 → **uncertain**; else `reject` → **rejected**; else **confirmed** organism.
   The 0.89 threshold is the smallest one whose accepted VAL predictions reach 99 % accuracy.
6. **Counts.** Confirmed counts plus `estimated_counts`. Tests check that both always add up to the final
   box list.

## Model details

| file | model | init | trained on | selected by |
|---|---|---|---|---|
| `models/detector_best.pt` | YOLOv8n (3.1 M params) | COCO `yolov8n.pt` | 2,000 synthetic train frames | synthetic VAL mAP (run `20261112_111516_detector_pt`) |
| `models/classifier_best.pt` = `classifier_v2.pt` | EfficientNet-B0 (4.0 M) | ImageNet `efficientnet_b0.ra_in1k` | 10,146 train crops + 10,146 crops re-cut from synthetic train frames | VAL macro F1 on original + frame crops (run `20261119_235518_cls_v2_frames`) |
| `models/classifier_v1.pt` | EfficientNet-B0 | ImageNet | 10,146 train crops | kept for comparison only |

SHA-256, sizes, training runs and library versions are in `results/model_manifest.json`, written by
`python -m scripts.package_models`. Check your copy with `python -m scripts.package_models --check`.

### Model weights

The weights are **not** in git (`models/` is git-ignored, about 39 MB in total). To get them, either:
- copy `models/` from the release archive or shared drive (check with `python -m scripts.package_models --check`), or
- retrain them with the commands in [Training from scratch](#training-from-scratch-reproduction).

If the weights are missing, the app and the CLI stop with a `ModelLoadError` that names the missing file.

## Installation

Needs **Python 3.12** and **Node.js 20 or newer** (for the web app; tested with Node 24). From the project root:

```powershell
# 1. Python backend and AI/ML
py -3.12 -m venv .venv                 # or: uv venv --python 3.12 .venv
.venv\Scripts\Activate.ps1             # Linux/macOS: source .venv/bin/activate
# PyTorch first, from the PyTorch index (pick ONE):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # NVIDIA GPU (CUDA 12.6 build)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu     # CPU only
pip install -r requirements.txt        # exact versions used: requirements-lock.txt
python -m scripts.package_models --check

# 2. Web frontend
cd frontend
npm ci                                 # exact versions from package-lock.json
npm run build                          # writes frontend/dist, which the API server serves
cd ..
```

### CUDA and CPU

- `project.device: auto` in `config.yaml` uses the GPU when `torch.cuda.is_available()` is true, otherwise the CPU.
  To force one, set `cuda` or `cpu`.
- **Check the GPU:** `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`. If it prints
  `False` on an NVIDIA machine, reinstall the CUDA build of torch that matches your driver (see pytorch.org).
- **Measured on this project's laptop** (RTX 3050 Laptop 4 GB, Ryzen 7 7435HS), 640 x 640 synthetic VAL frame:

  | | model load + warm-up | detection | classification |
  |---|---|---|---|
  | GPU | about 20 s | 20-150 ms | 40-110 ms |
  | CPU only | about 12 s | 80-110 ms | 175-215 ms |

  The display-only enhanced preview adds about 0.8 s per image on both (NL-means denoising, capped at 800 px).

## Running the application

**One command** (after `npm run build`): the API server also serves the web app.

```powershell
python -m api                          # http://127.0.0.1:8000  (API reference: /docs)
python -m api --host 0.0.0.0 --port 8080   # other address or port
```

**Development** (hot reload), two terminals:

```powershell
python -m api --reload                 # terminal 1: API on :8000
cd frontend; npm run dev               # terminal 2: web app on http://localhost:5173, /api proxied to :8000
```

If port 5173 is taken, Vite picks the next free port and prints it (`Local: http://localhost:5174/`).

The web app has two views:
- **Analyse:** drop or choose an image (type and size are checked in the browser, then again on the server), or
  pick an example synthetic VAL frame if `data/synthetic` exists; optional thresholds (empty = the VAL-chosen
  defaults). The result shows:
  - the confirmed, estimated, uncertain and rejected counts and the processing time;
  - the image as original, enhanced or annotated, with the detection boxes drawn live (pointing at a box lights its
    row in the detection table, and the reverse);
  - counts by class, every detection with its class probability and detector confidence, per-stage times;
  - PDF / HTML / CSV / JSON downloads.
- **History:** every analysis, failed uploads included; open one to see the full result again, re-download its
  reports or delete it, or clear the history.

Loading, empty and error states are shown everywhere (server offline, models missing, invalid image, empty
history). Analyses are stored in `results/history/` (git-ignored); set `PLANKTOSCAN_HISTORY_DIR` to use another folder.

### REST API

Every error is `{"error": {"code": ..., "message": ...}}`. Interactive reference: `http://127.0.0.1:8000/docs`.

| method and path | does | answers |
|---|---|---|
| `GET /api/health` | liveness, whether the models loaded, device | 200 |
| `GET /api/info` | default thresholds, accepted formats and limits, classes, model identity | 200, 503 if no models |
| `GET /api/examples` | synthetic VAL frames offered as examples | 200 |
| `POST /api/analyses` | analyse a multipart `file` (or `example=<name>`); optional `det_conf`, `uncertainty` | 201 with the result and links |
| `GET /api/analyses?limit&offset&status` | history, newest first | 200 |
| `GET /api/analyses/{id}` | one analysis: summary, canonical result, links | 200, 404 |
| `GET /api/analyses/{id}/images/{original,enhanced,annotated}` | stored image | 200, 404 |
| `GET /api/analyses/{id}/reports/{csv,json,html,pdf}` | report download | 200, 404 |
| `DELETE /api/analyses/{id}` | delete one analysis and its files | 204, 404 |
| `DELETE /api/analyses?confirm=true` | clear the history | 200, 400 without `confirm` |

Error codes and statuses: `invalid_image` 400, `unsupported_format` 415, `image_too_large` 413, `image_too_small` 422,
`bad_request` 422, `not_found` 404, `inference_error` 500, `model_load_error` 503. A failed analysis is also stored
in the history. Zero detections is a normal 201 result with the warning "No organisms detected."

## Command-line usage

```powershell
python -m scripts.analyze path\to\image.jpg --out outputs
python -m scripts.analyze path\to\folder --out outputs [--recursive] [--formats csv,json,html,pdf]
                          [--det-conf 0.5] [--uncertainty 0.89] [--no-history] [--history-db file.sqlite]
```

- **Per image:** a new folder `outputs/<name>_<analysis_id>/` containing `annotated.png`, `enhanced.png`,
  `report.json`, `report.csv`, `report.html` and `report.pdf`.
- **Per run:** `outputs/batch_<timestamp>.csv` and `.json`, one row per image (failures included, with the error code).
- **Failures:** a failed image is logged, and the rest of the folder is still processed.
- **Exit codes:** `0` all analysed, `1` some failed, `2` nothing could run (bad input path, no images, models missing).

Example console output, from a smoke test on synthetic VAL frames and deliberately broken files:

```
7 image(s); device cuda; thresholds detection 0.50, uncertainty 0.89; output ...\cli_out
[1/7] black.png: no_detections, confirmed 0, estimated 0, uncertain 0, rejected 0 (1028 ms)
[2/7] fake.png: FAILED (invalid_image) fake.png: not a readable image (corrupted or wrong file type).
[3/7] tiny.png: FAILED (image_too_small) tiny.png: 20x20 px is too small (minimum side 32 px).
[4/7] truncated.jpg: FAILED (invalid_image) truncated.jpg: the image data is corrupted or truncated.
[5/7] val_00000.jpg: ok, confirmed 3, estimated 3, uncertain 0, rejected 0 (...)
...
done: 4 analysed, 3 failed; summary ...\batch_<timestamp>.csv
```

Each of the three VAL frames has 3 pasted organisms, and the confirmed counts matched the ground truth class by class.

## Report example

Every format is rendered from `AnalysisResult.to_dict()`, so the numbers are identical across formats (this is tested).
The CSV has three sections. The example below is the synthetic VAL frame `val_00001.jpg`:

```
# summary
field,value
Image,val_00001.jpg
Thresholds,"detection 0.50, uncertainty 0.89, NMS IoU 0.75"
Classifier,classifier_best.pt (run 20261119_235518_cls_v2_frames)
Processing time,"962 ms total (load_and_validate 4, detection 154, classification 39, counting 0, annotation 1, enhancement_preview 763)"

# counts
class,confirmed,estimated,estimated_from_uncertain,note
diatom,2,2,0,
dinobryon,1,1,0,
ALL organisms,3,3,0,sum of the rows above
rejected,0,0,0,debris / skins / unidentifiable; not organisms
uncertain,0,,,below the confidence threshold; not in confirmed counts
total detections,3,3,,

# detections
id,status,label,prob,det_conf,x0,y0,x1,y1
1,organism,diatom,0.9002,0.919,356.4,63.6,379.0,90.0
...
```

(Classes with zero counts are omitted here.) Reports carry the domain disclaimer and every warning. The HTML
report embeds the annotated and enhanced images; the PDF embeds the annotated image.

## Project layout

```
config.yaml          paths, classes, seed, hyperparameters, service limits, decisions
src/service.py       analysis service (validation, errors, timings, canonical result, outputs)
src/history.py       SQLite analysis history
src/pipeline/        detect -> crop -> classify -> status; counting; reports
src/classify/        crops, frame crops (v2), augmentation, EfficientNet training, metrics
src/detect/          synthetic frame generator, YOLO helpers (offline mode, matching)
src/preprocess/      illumination correction, denoising, CLAHE
src/data/            download, inventory, class mapping, dedupe, day-block split, leakage check
src/common/          config, seeding, experiment logging, stage timing
src/evaluation.py    bootstrap CIs, count errors, per-frame scoring, figures
src/deploy/          ONNX export + CPU runtime (embedded feasibility)
api/                 REST API (FastAPI): python -m api
frontend/            web app (React + Vite): npm run dev / npm run build; tests in frontend/src/test
scripts/             entry points (python -m scripts.<name>)
tests/               pytest (models mocked by tests/fakes.py; the API tested with FastAPI's TestClient)
docs/viva_notes.md   one-line justification for every design decision
results/             generated tables, figures, manifests (committed); results/history/ is git-ignored
data/ models/ runs/  generated, git-ignored
```

## Dataset

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

## Tests

```powershell
pytest -q                              # Python: service, validation, counting, history, reports, CLI, REST API, ML code
cd frontend; npm test                  # web app: validation, counts, upload, errors, offline, history, delete
```

## Training from scratch (reproduction)

Not needed to use the app. Full reproduction, in order; every script logs its config, seed, git commit and versions to `runs/`.

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
# Optional, after the final evaluation (nothing is tuned by these)
python -m scripts.benchmark_cpu            # ONNX export + CPU-only latency, fills 'Embedded feasibility'
python -m scripts.eval_real --images <folder> [--counts counts.csv]   # real-image pilot
# Backend v2 (Phase A): classifier v2 on original + synthetic-frame crops
python -m scripts.calibrate_box_jitter   # detector box error on VAL frames -> frame-crop jitter
python -m scripts.train_classifier --name cls_v2_frames --pretrained 1 --grayscale 0 --weighting none --frame-crops 1
python -m scripts.select_classifier_v2 --run runs/<timestamp>_cls_v2_frames
python -m scripts.choose_threshold       # v2 threshold on VAL original + frame crops
python -m scripts.make_synthetic --fresh-test-seed 2026
python -m scripts.evaluate_v2 --dry-run-val   # code check on VAL
python -m scripts.evaluate_v2            # ONE-SHOT v1 vs v2 on fresh test frames (refuses to run twice)
python -m scripts.package_models         # model manifest with SHA-256
```

Resume an interrupted classifier run: `python -m scripts.train_classifier --resume runs/<run_dir>`.

## Evaluation results

**Headline numbers (current system: classifier v2, threshold 0.89).** Where each number was measured matters:

| what | number | measured on |
|---|---|---|
| classifier macro F1 (8 classes) / accuracy | 0.961 / 0.983 | ZooLake TEST crops (real single-organism crops) |
| detector mAP@0.5 / mAP@0.5:0.95 | 0.977 / 0.814 | fresh SYNTHETIC test frames |
| detection F1 (IoU 0.5) | 0.946 | fresh SYNTHETIC test frames |
| organism count MAE per frame, confirmed / estimated | 0.95 / 0.54 (mean 8.60 organisms per frame) | fresh SYNTHETIC test frames |
| matched objects flagged uncertain | 8.4 % | fresh SYNTHETIC test frames |
| any metric on real microscope frames | **not available** | no labelled real frames exist |


The classifier is judged by per-class F1 and macro F1, because the classes are imbalanced. All tuning
(preprocessing, class weighting, epochs, uncertainty threshold, NMS/confidence) used the **validation**
split only. The test split and the synthetic test frames were used only by `scripts/evaluate.py`
(v1) and, once, by `scripts/evaluate_v2.py` (v1 vs v2 on fresh test frames).
The generated section below is the **v1** evaluation (classifier `cls_pt_rgb`, threshold 0.84, original
synthetic test frames); a frozen copy of its outputs is in `results/v1/`. The current model is v2: see
[Classifier v2](#classifier-v2-fixing-undercounting) right after it. Do not edit the generated section by hand.

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

### Classifier v2: fixing undercounting

**Problem (v1).** On synthetic frames about half of the correctly detected organisms were flagged uncertain, so
strict counts were far too low (test count MAE 4.81 per frame, mean true 8.91). The cause was a domain gap: the
classifier was trained on native ZooLake crops but sees crops re-cut from frames.

**Fix.** Classifier v2 (`cls_v2_frames`) uses the same recipe as v1 (ImageNet init, RGB, preprocessing on,
unweighted, same LR, epochs and seed). The only change is the training data: it adds one crop cut from a
synthetic frame for every original crop (50/50). The frame crops are cut around the ground-truth box with a
random box jitter (edge std 0.06 of the side, measured from the detector's real box error on VAL frames,
`results/box_jitter_val.json`). They use the same `pad_crop` the pipeline uses. Train crops come only from
train frames and val crops only from val frames; test frames are refused in code.
- **Selection:** macro F1 on VAL original + frame crops gives v2 0.9454 against v1 0.8377
  (`results/classifier_selection_v2.json`).
- **Uncertainty threshold:** re-chosen on the same mixed VAL set with the same 99% rule, giving t = 0.89
  (v1 keeps 0.84).
- **v1 is kept** as `models/classifier_v1.pt`.

**Fresh test frames.** The old synthetic test frames had already shown the problem, so they could no longer
measure the fix fairly. v2 was therefore evaluated on 300 **new** synthetic test frames (generator seed 2026,
`data/synthetic_test_v2/`, 2,674 objects). They are built from the same test-split crops, because no other
labelled crops exist. Before the frames were generated, every file in `data/synthetic/` and `results/` was
hashed; afterwards only new files had appeared.
`python -m scripts.evaluate_v2` ran **once** (run `20261120_004219_evaluate_v2`) after a dry run on VAL frames,
and it refuses to run again. Numbers are from `results/v1_vs_v2.{json,md}`. Both models use the same detector
and are compared at their own VAL-chosen thresholds.

| fresh SYNTHETIC test frames (300) | v1 | v2 |
|---|---|---|
| uncertainty threshold (chosen on VAL) | 0.84 | 0.89 |
| matched objects flagged uncertain | 47.8% | **8.4%** |
| count MAE per frame, strict (confident organisms only) | 4.43 | **0.95** |
| signed count error, strict | -4.42 | **-0.78** |
| count MAE per frame, estimated (argmax, incl. uncertain) | 1.53 | **0.54** |
| signed count error, estimated | -1.34 | **-0.15** |
| detection F1 (IoU 0.5; same detector) | 0.946 | 0.946 |
| accuracy of confident matched items | 0.936 | 0.995 |
| classifier on TEST split crops: macro F1 / accuracy | 0.962 / 0.984 | 0.961 / 0.983 |
| classifier on TEST split crops: uncertain fraction | 0.038 | 0.053 |

Mean true organisms per frame: 8.60. Detector on the fresh frames: mAP@0.5 0.977, mAP@0.5:0.95 0.814.
Per-class count errors: `results/count_errors_test_v2_{v1,v2}_{strict,estimated}.csv`.

- **Strict counts** (`counts`) count only confident organisms; uncertain items are listed separately.
- **Estimated counts** (`estimated_counts`) put every detection under its classifier argmax and show how many
  of each class came from uncertain items (`from_uncertain`).
- **Which to use:** strict is the conservative count; the estimate is the best guess.

**Threshold sensitivity (v2, report only; not used to pick the threshold).** The top softmax is capped near 0.93
by label smoothing, so the uncertain rate climbs steeply just above the chosen value:

| v2 threshold | matched uncertain | count MAE (strict) | signed error (strict) | test crops uncertain |
|---|---|---|---|---|
| 0.85 | 4.3% | 0.71 | -0.47 | 0.026 |
| **0.89 (chosen on VAL)** | 8.4% | 0.95 | -0.78 | 0.053 |
| 0.90 | 21.1% | 1.90 | -1.86 | 0.137 |

A 0.01 change in the threshold (0.89 to 0.90) more than doubles the uncertain rate. On new data, expect
the uncertain rate to be unstable.

**Reject got worse in v2.**
- **TEST crops:** reject F1 is 0.764 (v1 0.777); the organism classes are unchanged or better.
- **VAL frame crops:** of 70 true reject objects, v2 confidently calls 9 an organism (rotifer 4, dinobryon 2,
  nauplius 1, cladoceran 1, diatom 1), against 1 for v1. v2 confidently rejects 18 and flags 43 uncertain;
  v1 confidently rejected 34 and flagged 35.
- **Fresh test frames:** v2 confidently rejects 0.12 items per frame against 0.31 true reject objects.

So in frames, v2 lets more debris through as organisms or uncertain. No reject-specific retraining was done.

## Real-image qualitative testing

No real microscope images were available while this was built (`data/real_samples/` does not exist), so **no real-image observations or numbers are reported**. To look at real images qualitatively (detections, counts and times only, no accuracy claims), run the normal CLI on a folder:

```powershell
python -m scripts.analyze data/real_samples --out results/real_samples_out --no-history
```

Results on real images must be reported as observations, kept apart from the synthetic-frame metrics above. If manual counts exist for some images, the pilot below compares against them.

### Real-image pilot with manual counts (`scripts/eval_real.py`)

A check of the frozen pipeline on real microscope images, which ZooLake does not provide. It is
**not used for tuning**: models and thresholds stay exactly as chosen on VAL, and every output is
tagged "real pilot, not used for tuning".

```
python -m scripts.eval_real --images path/to/real_images --counts path/to/counts.csv
```

- **counts.csv** has one row per image: `image` (file name, with or without extension) and `total_count`,
  plus optional per-class columns named like the classes (`copepod`, `diatom`, `nauplius`, `cladoceran`,
  `rotifer`, `ceratium`, `dinobryon`).
  - Total counts alone are enough. Empty cells are skipped, and other columns are listed and ignored.
  - `--counts` is optional; without it you get predictions only.
- **Predicted total.** Two variants are compared with the manual total:
  - confident organisms + uncertain: the main number, because uncertain items are usually organisms;
  - confident organisms only: a lower bound.

  `reject` items are never counted as organisms.
- **Errors.** Per image: predicted vs manual. Overall: MAE and mean signed error (predicted − manual).
- **Outputs** in `results/real_pilot/`:
  - `per_image.csv`, `detections.csv`, `summary.csv/.json/.md`;
  - `scatter_total.png` (predicted vs manual);
  - `annotated/` (first `--max-examples` images, with a banner).

  The folder is git-ignored because real images may not be ours to publish.
- **Status.** No real labelled images were available while building this, so no real-pilot numbers
  are reported here.

## Embedded feasibility

`python -m scripts.benchmark_cpu` (run with classifier v1; v2 has the same architecture and input size, so the latencies apply):
- exports both final models to ONNX (fp32) and runs them on the **CPU only**, with the GPU hidden;
- times them on synthetic VAL frames and VAL crops, never test data;
- checks ONNX outputs against PyTorch and Ultralytics.

The runtime is OpenCV's DNN module. `onnxruntime` and `onnx` are not installed (no further downloads
were allowed); the script uses onnxruntime automatically if it is present. The single-thread column is
a rough proxy for a small embedded CPU, not a measurement on one.

<!-- EMBEDDED:START -->
<!-- generated by scripts/benchmark_cpu.py; do not edit by hand -->
_CPU only (GPU hidden): AMD Ryzen 7 7435HS, 16 logical CPUs, runtime OpenCV DNN 5.0.0 (CPU). Median over 100 SYNTHETIC VAL frames (640x640 px, mean 8.5 detections per frame) and 100 VAL crops, after 10 warm-up runs. Run `20261112_131506_benchmark_cpu`._

| model | params (M) | .pt (MB) | ONNX fp32 (MB) |
|---|---:|---:|---:|
| EfficientNet-B0 classifier (128 px) | 4.02 | 16.35 | 16.05 |
| YOLOv8n detector (640 px) | 3.14 | 6.51 | 12.63 |

| latency (ms, median) | 16 threads | 1 thread |
|---|---:|---:|
| detector per frame (letterbox + forward + NMS) | 52.9 | 236.6 |
| classifier per crop, forward only | 5.2 | 16.4 |
| crop preprocessing only (illumination, NL-means, CLAHE, letterbox) | 5.7 | 5.8 |
| classifier per crop, preprocessing + forward | 12.0 | 23.2 |
| **full pipeline per frame** (detector + all crops, batch 1) | 177.6 | 464.0 |
| reference: PyTorch CPU classifier forward per crop | 26.0 | 21.9 |

ONNX parity: classifier max abs logit difference 1.5e-05, argmax agreement 100% (20 crops); detector 139 ONNX vs 139 Ultralytics boxes on 20 VAL frames, 100.0% of Ultralytics boxes matched at IoU >= 0.9.
<!-- EMBEDDED:END -->

How to read it:
- **Sizes.** The ONNX detector is about twice the size of its `.pt` file because Ultralytics stores fp16
  weights and the export is fp32. fp16/int8 quantisation was not tried.
- **Per-frame cost.** It is the detector plus one classifier call per detected object, so it grows with the
  number of organisms in the frame.
- **Cheapest speed-ups.** Batching the crops (here each crop is classified alone, batch 1, as on a
  small device) and lowering the detector input size. Both would need re-validation on VAL.
- **What this does not measure.** These are laptop-CPU numbers. Real embedded boards (Raspberry Pi, Jetson)
  were not measured.

## Reproducibility
- Fixed seed 42 (`project.seed`) for Python, NumPy, PyTorch, the split and the frame generator; cuDNN
  deterministic mode for the classifier. YOLO training uses the same seed with `deterministic=False`
  for speed, so detector numbers can differ slightly between reruns.
- Each run writes `config.yaml` and `environment.json` (seed, git commit, library versions, command) to
  `runs/<timestamp>_<name>/`.
- Single seed per experiment: differences of about 1 macro-F1 point between ablation arms are within noise.
- The model files are identified by SHA-256 in `results/model_manifest.json`; analysis reports include the classifier run and both model checksums in their metadata.

## Limitations

- **Synthetic detector data.** The detector was trained on synthetic frames, and all detector and counting
  numbers (mAP, count MAE) come from synthetic frames we generated ourselves. They are optimistic upper bounds,
  not real-world performance.
- **No validated real-world accuracy.** No labelled real multi-organism microscope images were available.
  Accuracy, precision, recall and mAP on real images are unknown; no real-image numbers are claimed.
- **Freshwater domain gap.** The data is ZooLake: one lake (Greifensee), one dark-field plankton camera. SIH25043
  targets marine plankton. Marine species, other microscopes, magnifications and bright-field illumination are
  out of domain.
- **Limited classes.** 7 organism groups plus `reject`. Merged groups (`rotifer` covers 10 genera, `cladoceran` 3,
  `diatom` 3) hide variety. Organisms outside these groups cannot be named; the uncertainty flag catches only part
  of them (open set: 71.9 % flagged uncertain with v1, 9.9 % wrongly counted as an organism).
- **Sensitivity to image quality.** Illumination, focus, noise, resolution and magnification all matter.
  Large images are resized to 640 px for detection, so small organisms may be lost (the service warns above
  2,560 px).
- **Fragile threshold.** v2's uncertainty threshold 0.89 sits next to a confidence cliff. On fresh synthetic test
  frames, raising it to 0.90 moves the uncertain share from 8.4 % to 21.1 %. Expect the uncertain rate to shift
  on new data.
- **Reject is weaker in v2.** In frames, v2 more often lets debris through as an organism (9 of 70 VAL reject frame
  crops, against 1 for v1). Reject test-crop F1 is 0.764 (v1 0.777).
- **Fresh test frames reuse the test crops.** The v2 test frames are new arrangements of the same test-split crops,
  which had been used once before to evaluate v1.
- **Diatom test estimate is weak.** 64 % of diatoms come from one bloom day (in train), so the diatom test F1 rests
  on few days.
- **No video, tracking or concentration.** Counts are per image. There is no tracking across frames and no
  calibration to organisms per litre (that needs the imaged volume).
- **Single seed** per experiment; differences of about 1 macro-F1 point are within noise.
- **Dataset licence unclear** (see Dataset). The images are not redistributed.

This tool is a decision aid for plankton images similar to its training data. It is not a universal plankton
identifier, and its counts are not validated ecological measurements.

## Future improvements

- Label a set of real microscope frames (boxes + classes) and measure detection and counting on them before any
  real use. Then fine-tune the detector on real frames.
- Add marine data and classes for the SIH25043 setting.
- Calibrate confidences (e.g. temperature scaling on VAL) so the uncertainty threshold is not on a cliff.
- Improve the `reject` class (more debris examples, frame-crop reweighting) and the open-set behaviour.
- Tiled inference for large images, so small organisms are not lost in the 640 px resize (must be validated on VAL).
- Calibration to organisms per litre from the imaged volume; video input with tracking to avoid double counts.
- Quantised ONNX models for edge devices (measured on the actual board).
