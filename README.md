# PlanktoScan AI

Detects, classifies and counts plankton in microscopy images. Computer Vision end-semester project
based on SIH25043 (software and AI part).

**Pipeline:** image → YOLOv8n object detector → crop → preprocessing (illumination correction,
denoising, CLAHE) → EfficientNet-B0 classifier with confidence → per-class counts (low-confidence items
flagged **uncertain**, `reject` items listed separately) → annotated image + CSV / HTML / PDF report.

> **Read the limitations first.** The models were trained on a freshwater dataset of single-organism
> crops. Both networks were trained **from scratch** (no pretrained weights were available offline).
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
python -m scripts.select_classifier --prefix cls_
python -m scripts.choose_threshold        # uncertainty threshold on VAL
# Phase 5 detector
python -m scripts.make_synthetic          # 2000/300/300 frames, per-split crops only
python -m scripts.train_detector          # YOLOv8n from scratch, time-capped
python -m scripts.tune_detector           # NMS IoU + confidence on synthetic VAL frames
# Phase 8: every table and figure + this README's results section
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
*Not generated yet. Run `python -m scripts.evaluate`.*
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
- **Trained from scratch.** No pretrained weights were available offline and downloads were not
  allowed, so EfficientNet-B0 and YOLOv8n start from random weights. ImageNet/COCO initialisation
  would likely improve both.
- **Merged classes.** Classes such as `rotifer` (10 genera), `cladoceran` (3) and `diatom` (3) hide
  within-class variety. `reject` mixes debris, moulted skins and unidentifiable objects, possibly including
  organism fragments.
- **Bloom-day diatom issue.** 64% of diatoms come from one bloom day (in train). Diatom test has few
  images from few days and is mostly *fragilaria*, so its F1 is a weak estimate.
- **Open set.** Organisms from the 10 excluded classes are not known to the classifier. The
  uncertainty flag catches only part of them (see Results).
- **Unclear dataset license** (see Data section).
- **Single seed** per experiment; no repeated runs.
