# PlanktoScan AI

Detects, classifies and counts plankton in microscopy images. Computer Vision end-semester project
based on SIH25043 (software and AI part).

**Pipeline:** image → preprocessing (flat-field, denoise, CLAHE) → YOLOv8 organism detector → crop →
EfficientNet classifier with confidence → per-class counts → annotated image + CSV / HTML / PDF report.

> Status: Phase 1 (project setup). Sections marked *TBD* are filled in by later phases.

## Data and important limitations

- **Dataset: ZooLake.** It is a **freshwater** (lake) plankton dataset, not marine. SIH25043 is framed
  around marine plankton, so species and imaging conditions will differ from marine samples. A
  model trained here should be retrained or fine-tuned on marine data before real marine use.
- **ZooLake has only single-organism crops, not full frames with bounding boxes.** The classifier is
  trained and tested on real crops. The detector is trained on **synthetic multi-organism frames**
  made by alpha-blending crops onto procedural microscopy-style backgrounds. Each split's frames use
  only that split's crops.
- **Detector mAP is measured on synthetic frames.** It shows how well the detector finds pasted
  organisms. It is not a measure of performance on real full-field microscopy images.
- Class subset and merge mapping: *TBD (Phase 2)*.

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
