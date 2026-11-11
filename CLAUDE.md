# CLAUDE.md: PlanktoScan AI

Computer Vision end-semester project based on SIH25043 (software/AI part only).

## Goal
Microscopy image of plankton -> enhance -> detect every organism (boxes) -> classify species with
confidence -> count per class -> annotated image + CSV/HTML/PDF report.

## Stack
Python 3.12 (project venv `.venv`), PyTorch, Ultralytics YOLOv8 (n/s), EfficientNet-B0/B2 via timm,
OpenCV, Streamlit (not FastAPI), reportlab + jinja2 for reports. GPU-optional; the default target is a
local RTX 3050 Laptop (4 GB): mixed precision, small batches. A Colab notebook is the backup.

## Data decisions (agreed with the user)
- Dataset: **ZooLake only** to start. It is a **freshwater** lake dataset; state this limitation in the README.
- The source URL must come from the ZooLake paper's data-availability section. Show the URL and size to
  the user before downloading. Never invent URLs. Ask before any download over 1 GB.
- Print the full raw class list with counts first, then propose a merge mapping into
  copepod / diatom / nauplius + 2-5 other classes. The user approves it before it goes into config.yaml.
- Split 70/15/15, leakage-safe (group near-duplicates), fixed seed.
- ZooLake is crops only, so the detector uses SYNTHETIC frames:
  - each split's frames use ONLY that split's crops (train/val/test)
  - alpha-mask blending (no hard seams), random position/scale/rotation, some overlap, noise, uneven illumination
  - YOLO labels written automatically
  - README and results must state that **detector mAP is measured on synthetic frames**.

## Phases (stop after each one for user review)
1. Setup  2. Data  3. Preprocess  4. Classifier  5. Detector  6. Pipeline  7. Streamlit app + report
8. evaluate.py + final README

## Rules
- Fixed seed; log all experiment settings (`src/common/experiment.py::start_run`).
- Modular: src/preprocess, src/classify, src/detect, src/pipeline, app/. Type hints and short docstrings.
- pytest tests for preprocessing, label generation and counting logic.
- Judge by per-class F1 and macro F1, not accuracy alone.
- Report real numbers only, including failures. Never hardcode or fabricate metrics.
- Tell the user the commands to run and the expected output. Run long training only after the user confirms.
- Each design decision gets a one-line viva note in `docs/viva_notes.md`.
- Low-confidence classifications are labelled "uncertain", never silently dropped.
- Git: local repo only. Push to a private GitHub repo only when the user says so.
