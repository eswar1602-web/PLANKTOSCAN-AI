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
- **Leakage-safe split:** near-identical images of the same organism in both train and test would inflate scores (detailed in Phase 2 below).
- **Synthetic frames built per split:** train frames use only train crops, val only val, test only test, so the detector is never tested on organisms it has seen.
- **Alpha-blended pasting:** hard paste edges are a shortcut the detector could learn instead of organism shape.
- **"Uncertain" label instead of dropping low-confidence detections:** counts stay complete, and a human can review doubtful organisms.
- **Mixed precision + batch 32/8:** fits the 4 GB RTX 3050 and roughly halves memory use with negligible accuracy cost.
- **reportlab for PDF:** pure Python, so there are no system dependencies to break on Windows or Colab.

## Phase 2: data
- **ZooLake only, chosen from the paper's data statement:** a verified source (DOI, published SHA-256) with taxonomist labels; the freshwater-vs-marine mismatch is stated as a limitation.
- **Merging raw classes into 8 project classes:** this cuts the imbalance ratio from 332:1 to 7.3:1 and leaves at least 55 test images per class, so per-class F1 is not just noise.
- **dinobryon included:** it is the most common object in the lake, and leaving it out would make real samples produce many forced or "uncertain" labels.
- **`reject` rather than "non-organism":** the dataset README says "unknown" may contain organism fragments, so the honest name is "not countable as a target organism".
- **Open set kept apart, never trained or tuned on:** it measures what happens when the system meets organisms it was never taught. That is the realistic failure mode in the field.
- **Whole-day blocks, not a random split:** images from the same day share water, lighting and often the same bloom, so a random split leaks; a pure chronological cut would put seasonal classes in only one split.
- **Stratified on class and raw subtype together:** otherwise one split can be all asterionella and another all fragilaria; this cut the number of off-balance subtypes from 15 to 5 of 25.
- **Hour fallback kept only if it helps:** finer blocks raise leakage risk, so they are used only when they actually improve the val/test fill (for diatom it didn't, so it was reverted).
- **Two-stage leakage check (pHash, then pixel correlation):** pHash alone flagged about 4,600 cross-split pairs on these tiny dark-field crops, mostly different classes months apart. Confirmation found 0 real near-duplicates.
- **Byte-identical files removed (5 rows):** two images were filed under two different classes. A contradictory label is worse than no label, so both copies were dropped.
- **Test split untouched until Phase 8:** any decision made after looking at test results turns the test set into a second validation set and inflates the reported numbers.
