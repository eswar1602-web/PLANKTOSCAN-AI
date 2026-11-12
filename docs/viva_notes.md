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

## Phase 3: preprocessing
- **Subtract the background, don't divide (flat-field):** in dark field the background is near-black, so dividing by it amplifies noise; subtracting the background level is the dark-field equivalent of flat-field correction.
- **Border-median background for crops:** a large blur on a 72 px crop is mostly the organism itself, so its border pixels are a safer, parameter-free estimate of the background.
- **CLAHE tile size in pixels, not a fixed grid:** an 8x8 grid on a 16 px crop would create 2 px tiles; tile_px=32 keeps tiles meaningful on any image size.
- **One `prepare_crop` function for training and inference:** a train/inference mismatch in preprocessing silently lowers accuracy in deployment; one shared function removes that risk.
- **Preprocessing ablation result:** val macro F1 0.8621 with vs 0.8608 without (`results/ablation.csv`). Kept on by the rule set beforehand, but the gap is within single-seed noise: the camera software had already enhanced these crops.
- **No frame-level preprocessing for the detector:** it was never evaluated within the time budget, and an untested step in the detection path is a risk without a measured benefit.

## Phase 4: classifier
- **EfficientNet-B0 at 128 px, not 224:** crops are tiny (median 72x76), so 128 px keeps nearly all detail at a third of the compute, which matters on a 4 GB GPU.
- **Letterbox with black padding, not stretching:** stretching distorts shape, a key cue for plankton; black padding matches the dark-field background.
- **Rotations up to 180 degrees and both flips:** organisms drift in any orientation, so orientation carries no class information.
- **GPU batch augmentation, no DataLoader workers:** Windows multiprocessing data loading is fragile; augmenting whole batches on the GPU is fast and deterministic.
- **LR 1e-3 + warmup + cosine + label smoothing 0.1, same for scratch and pretrained:** identical hyperparameters make the init comparison clean; label smoothing curbs over-confidence (it also caps max softmax near 0.93, which bounds the useful threshold range).
- **ImageNet fine-tuning chosen over from scratch:** on the same split and seed, val macro F1 rose from 0.906 to 0.959; small datasets benefit from generic edge/texture features.
- **RGB kept, not grayscale x3:** crops are coloured and colour carries class information (diatoms are distinctly less blue); both were trained and RGB (0.9593) vs gray (0.9590) is a tie, so the rule 'highest val macro F1' picked RGB.
- **Checkpoint chosen by val macro F1:** we select on the metric we report, not on val loss or accuracy.

## Phase 5: detector
- **Single-class detector:** the detector only answers "is there an object"; species identity comes from the classifier trained on real crops.
- **Reject crops are pasted as detectable objects:** debris also appears in real samples; the classifier, not the detector, decides what is rejected.
- **Alpha mask from the crop's own brightness, edges forced transparent:** no rectangular seams for the detector to exploit.
- **Native pixel scale (0.8-1.25x), large crops capped at 192 px:** keeps organism sizes realistic relative to the frame.
- **Tight boxes from the alpha mask, not the crop rectangle:** labels follow the organism, not the padding around it.
- **Per-split frames with an assertion:** a detector tested on organisms it was trained on would report inflated mAP.
- **YOLOv8n fine-tuned from COCO yolov8n.pt, 40 min cap:** 63 epochs reached val mAP@0.5:0.95 0.840 vs 0.683 for the from-scratch run (stopped at 7 epochs; 0.767 for pretrained at the same epoch).
- **Only two allow-listed weight files, SHA-256 checked against the publisher:** the offline guard still blocks every other download.
- **Ultralytics network side effects disabled and blocked:** the project rule forbids downloads, and the library would otherwise fetch a model for its AMP check and a font for plots.
- **NMS IoU chosen by val mAP@0.5:0.95; confidence by val detection F1:** both are tuned on synthetic val frames only.

## Phase 6-7: pipeline and app
- **"uncertain" when max softmax < threshold, chosen on VAL as the smallest threshold with >=99% accepted accuracy (raised from 97%, which gave t=0 for the pretrained model):** low-confidence items are shown and counted separately rather than silently counted wrong or dropped.
- **Low confidence takes priority over the reject label:** an unsure "reject" is still unsure.
- **Crops are cut from the original image, then preprocessed:** identical to how the classifier was trained.
- **reportlab PDF, jinja2 HTML, CSV:** pure-Python exports with no system dependencies.
- **Domain warning in the app, HTML and PDF:** results on real microscope images are not validated, and the user must see that wherever the numbers appear.

## Phase 8: evaluation
- **Test split used once, by one script:** every number in the README is regenerated by `scripts/evaluate.py`, so no metric is typed by hand.
- **Day-block bootstrap for CIs:** images from one day are correlated; resampling single images would give overconfident intervals.
- **Open-set reported with and without maybe_cyano:** its labels are themselves uncertain, so it could flatter or penalise the uncertainty flag.
- **End-to-end count error per class on synthetic test frames:** counting is the actual product output, so it is evaluated directly, not only through mAP and F1.
- **Frame-crop diagnostic on VAL with ground-truth boxes:** it separates detector errors from classifier domain shift; it showed the end-to-end uncertainty comes from the classifier, not the detector.
- **No fix after seeing test:** retraining on frame crops after the test evaluation would turn the test set into a tuning set; it is listed as future work.
