---
title: PlanktoScan AI
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Detect, classify and count plankton in microscopy images
---

# PlanktoScan AI

Upload a dark-field microscopy image of freshwater plankton. The app enhances it, detects every organism
(YOLOv8n), classifies each one into 7 organism classes or "reject" (EfficientNet-B0), flags uncertain and
possibly unknown organisms, counts them per class with 95% ranges, and offers CSV, JSON, HTML and PDF reports.

- Trained on the ZooLake dataset, a **freshwater** lake plankton dataset. Not validated on marine samples.
- The example images are **synthetic** frames built from ZooLake validation crops, not real microscope frames.
  Detector accuracy was also measured on synthetic frames only.
- This demo runs on a shared CPU. Analysis history is shared by every visitor and resets when the Space restarts.

Source code, training and evaluation: https://github.com/eswar1602-web/PLANKTOSCAN-AI

Dataset: ZooLake, Eawag, DOI [10.25678/0004DY](https://doi.org/10.25678/0004DY). Kyathanahally et al. (2021),
*Deep Learning Classification of Lake Zooplankton*, Frontiers in Microbiology 12:746297,
[doi:10.3389/fmicb.2021.746297](https://doi.org/10.3389/fmicb.2021.746297). Used for non-commercial academic work.
