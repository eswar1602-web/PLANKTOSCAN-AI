# Viva notes: design decisions

One line per decision. Added to as each phase is built.

## Phase 1: setup
- **Two-stage design (YOLOv8n detector + EfficientNet-B0 classifier):** ZooLake only has labelled single-organism crops, so the classifier trains on real images and the detector only has to learn "organism vs background" from synthetic frames.
- **Single-class detector:** species identity comes from the classifier, which sees real crops at higher resolution; a multi-class detector would learn species only from synthetic pastes.
- **Python 3.12:** the newest version with mature, tested PyTorch and Ultralytics builds, so setup is predictable.
- **Macro F1 as the headline metric:** the classes are imbalanced, so accuracy can look high while rare classes fail; macro F1 weights every class equally.
- **Best checkpoint chosen by val macro F1, not val loss:** the checkpoint is picked by the same metric we report.
- **Inverse-sqrt-frequency class weights:** they help rare classes without the instability that full inverse-frequency weights can cause on very rare classes.
- **All settings in one config.yaml, copied into every run folder with seed, git commit and library versions:** any reported number can be traced back to the exact code and settings that produced it.
- **Leakage-safe split (perceptual-hash near-duplicate grouping):** near-identical images of the same organism in both train and test would inflate scores.
- **Synthetic frames built per split:** train frames use only train crops, val only val, test only test, so the detector is never tested on organisms it has seen.
- **Alpha-blended pasting:** hard paste edges are a shortcut the detector could learn instead of organism shape.
- **"Uncertain" label instead of dropping low-confidence detections:** counts stay complete, and a human can review doubtful organisms.
- **Mixed precision + batch 32/8:** fits the 4 GB RTX 3050 and roughly halves memory use with negligible accuracy cost.
- **reportlab for PDF:** pure Python, so there are no system dependencies to break on Windows or Colab.
