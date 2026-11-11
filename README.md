# PlanktoScan AI

Detects, classifies and counts plankton in microscopy images. Computer Vision end-semester project
based on SIH25043 (software and AI part).

**Pipeline:** image → preprocessing (flat-field, denoise, CLAHE) → YOLOv8 organism detector → crop →
EfficientNet classifier with confidence → per-class counts → annotated image + CSV / HTML / PDF report.

> Status: Phase 1 (project setup). Sections marked *TBD* are filled in by later phases.

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
config.yaml        all paths, classes, seed and hyperparameters
src/common/        config loader, seeding, experiment logging
src/data/          download, class subset, leakage-safe split, imbalance report
src/preprocess/    CLAHE, denoising, flat-field correction
src/classify/      EfficientNet training and inference
src/detect/        synthetic frame generator, YOLOv8 training and inference
src/pipeline/      end-to-end image -> counts
app/               Streamlit UI
scripts/           command-line entry points
tests/             pytest tests
docs/viva_notes.md one-line justification for each design decision
data/ models/ runs/ results/   generated, git-ignored
```

## Setup (Windows, Python 3.12)

```powershell
cd C:\Users\eswar\planktoscan-ai
.venv\Scripts\Activate.ps1
# 1) PyTorch with CUDA. Check the exact command for your driver at https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# 2) Everything else
pip install -r requirements.txt
# 3) Check it works
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pytest
```

Colab backup notebook: *TBD (Phase 4)*.

## Usage
*TBD (Phases 2–7).*

## Results
*TBD (Phase 8).* All numbers will come from `python evaluate.py` and be written to `results/`.
The classifier is judged by per-class F1 and macro F1, because the classes are imbalanced.

## Reproducibility
Fixed seed (`project.seed` in `config.yaml`). Each training or evaluation run writes its resolved
config, seed, git commit and library versions to `runs/<timestamp>_<name>/`.

## Known limitations
- Freshwater training data (see above).
- Detector evaluated only on synthetic frames (see above).
- *More TBD after evaluation.*
